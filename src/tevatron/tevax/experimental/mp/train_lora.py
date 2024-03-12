import os
import logging

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax
from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding

import datasets
from transformers import AutoTokenizer
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field

try:
    from grad_cache import cachex
    _gradcache_available = True
except ImportError:
    _gradcache_available = False

import magix
from magix.models import ENCODER_MODEL_MAPPING
from magix.lora import Lora

from grad_cache import cachex

from ...loss import contrastive_loss
from .loss import contrastive_loss_2dm

#TODO: use tevatron dataset instead
class TrainDataset:
    def __init__(
        self,
        train_data,
        group_size,
        tokenizer,
        query_max_length=32,
        passage_max_length=128
    ):
        self.group_size = group_size
        self.data = train_data
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    def __len__(self):
        return len(self.data)

    def get_example(self, i, epoch):
        example = self.data[i]
        q = example['query']

        pp = example['positive_passages']
        p = pp[epoch % len(pp)]
        p = p['title'] + ' ' + p['text']

        nn = example['negative_passages']
        off = epoch*(self.group_size - 1) % len(nn)
        nn = nn*2
        nn = nn[off: off + self.group_size - 1]
        nn = [p['title'] + ' ' + p['text'] for p in nn]

        return q, [p] + nn

    def get_batch(self, indices, epoch):
        qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
        dd = sum(dd, [])

        return dict(self.tokenizer(qq, max_length=self.query_max_length, padding='max_length', truncation=True, return_tensors='np')), \
         dict(self.tokenizer(dd, max_length=self.passage_max_length, padding='max_length', truncation=True, return_tensors='np'))

def data_loader(rng: jax.random.PRNGKey, dataset: TrainDataset, batch_size: int, epoch: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset.get_batch(idx, epoch)
        yield batch
        
class Batches:
    def __init__(self, rng: jax.random.PRNGKey, dataset: TrainDataset, batch_size: int, epoch: int, shuffle: bool = False):
        steps_per_epoch = len(dataset) // batch_size

        if shuffle:
            batch_idx = jax.random.permutation(rng, len(dataset))
        else:
            batch_idx = jnp.arange(len(dataset))

        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
        
        self.dataset = dataset
        self.batch_idx = batch_idx
        self.epoch = epoch
        
    def __call__(self, step):
        idx = self.batch_idx[step]
        batch = self.dataset.get_batch(idx, self.epoch)
        return batch


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
):
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: (path[-1] != "bias" and 'layernorm' not in path[-2]) for path in flat_params}
    return flax.traverse_util.unflatten_dict(flat_mask)


@dataclass
class TrainArgs:
    train_file: str = 'Tevatron/wikipedia-nq'
    checkpoint_dir: str = 'checkpoints/nq'
    max_query_length: int = 32
    max_passage_length: int = 160
    num_epochs: int = 1
    batch_size: int = 16
    num_target_passages: int = 16
    grad_cache: bool = False
    query_num_chunks: int = 4
    passage_num_chunks: int = 8
    learning_rate: float = 2e-6
    weight_decay: float = 0.0001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    save_steps: int = 50
    seed: int = 42
    pooling: str = 'bos'
    scale_by_dim: bool = True
    
    def __post_init__(self):
        assert self.pooling in ['bos', 'eos', 'cls']
    
@dataclass
class ModelArgs:
    model_type: str = 'llama'
    model_name: str = None
    model_cache_dir: str = None
    mesh_shape: List[int] = list_field(-1, 1)
    fully_shard_params: bool = True
    
    def __post_init__(self):
        if self.model_type not in ENCODER_MODEL_MAPPING:
            raise ValueError(f"model_type must be one of {list(ENCODER_MODEL_MAPPING.keys())}")

def main():
    parser = ArgumentParser()
    parser.add_arguments(TrainArgs, dest="train_args")
    parser.add_arguments(ModelArgs, dest="model_args")
    args = parser.parse_args()
    train_args: TrainArgs = args.train_args
    model_args: ModelArgs = args.model_args
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    # dataset setup
    if train_args.train_file.endswith('.jsonl'):
        train_data = datasets.load_dataset('json', data_files=train_args.train_file)['train']
    else:     
        train_data = datasets.load_dataset(train_args.train_file)['train']
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        add_eos_token=True, use_fast=True, padding_side='right', legacy=False)
    if tokenizer.pad_token_id is None:
        logger.warning("Tokenizer does not have a pad token. Adding eos token as pad token")
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = TrainDataset(
        train_data, train_args.num_target_passages, tokenizer,
        query_max_length=train_args.max_query_length,
        passage_max_length=train_args.max_passage_length
    )
    
    # optimizer setup
    total_train_steps = len(train_dataset) // train_args.batch_size * train_args.num_epochs
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_args.batch_size,
        train_args.num_epochs,
        int(total_train_steps*0.1),
        train_args.learning_rate,
    )
    optimizer = optax.adamw(
        linear_decay_lr_schedule_fn,
        mask=decay_mask_fn,
        weight_decay=train_args.weight_decay,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_args.max_grad_norm),
        optimizer
    )
    optimizer = optax.apply_if_finite(optimizer, 10)
    
    lora = Lora(
        alpha=30,
        rules={
            'layers/.*/kernel': 32,
        }
    )
    
    # initalize model parameters and optimizer state
    checkpoint_manager = magix.get_chckpoint_manager(train_args.checkpoint_dir, train_args.save_steps, items=['lora', 'optimizer'])    
    is_new_train = checkpoint_manager.latest_step() is None

    mesh = magix.create_device_mesh(model_args.mesh_shape)    
    _model_cls = ENCODER_MODEL_MAPPING[model_args.model_type]
    sharding_config = _model_cls.partition_rules
    if model_args.fully_shard_params:
        model_sharding_config = sharding_config
    else:
        model_sharding_config = magix.spmd_utils.duplicate_over(sharding_config, 'data')
    
    model, params = magix.load_model_hub(_model_cls, model_args.model_name, model_sharding_config, mesh, half=True)

    rng = jax.random.key(train_args.seed)
    dropout_rng, data_rng, lora_rng = jax.random.split(rng, 3)
    
    def create_lora_and_opt_states(rng, params):
        lora_params = lora.init_params(rng, params)
        opt_state = optimizer.init(lora_params)
        return lora_params, opt_state
    
    lora_shapes, opt_shapes = jax.eval_shape(create_lora_and_opt_states, lora_rng, params)
    lora_sharding = magix.lora.create_lora_sharding(model_sharding_config, mesh, lora_shapes)
    opt_sharding = magix.lora.create_lora_sharding(sharding_config, mesh, opt_shapes)

    if is_new_train:
        lora_params = jax.jit(lora.init_params, out_shardings=lora_sharding) (lora_rng, params)
        opt_state = jax.jit(optimizer.init, out_shardings=opt_sharding) (lora_params)
    else:
        loaded = magix.checkpoint_utils.load_by_sharding(
            checkpoint_manager,
            items=['lora', 'optimizer'],
            dummies=[lora_shapes, opt_shapes],
            shardings=[lora_sharding, opt_sharding]
        )
        lora_params, opt_state = loaded['lora'], loaded['optimizer']

    
    def train_step_cached(params, lora_params, opt_state, queries, passages, dropout_rng, query_num_chunks=4, passage_num_chunks=8):        
        q_rngs, p_rngs = jax.random.split(dropout_rng, 2)
        
        if train_args.grad_cache:
            queries = cachex.tree_chunk(queries, query_num_chunks)
            passages = cachex.tree_chunk(passages, passage_num_chunks)

            q_rngs = jax.random.split(q_rngs, query_num_chunks)
            p_rngs = jax.random.split(p_rngs, passage_num_chunks)

        queries['dropout_rng'] = q_rngs
        passages['dropout_rng'] = p_rngs

        def fwd_fn(params, batch):
            out = model(**batch, params=params, train=True)[0]
            if train_args.pooling in ['bos', 'cls']:
                return out[:, 0]
            elif train_args.pooling == 'eos':
                mask = batch['attention_mask']
                eos_indices = mask.sum(axis=1) - 1
                @jax.vmap
                def gather_eos(x, eos_idx):
                    return x[eos_idx]
                return gather_eos(out, eos_indices)
            else:
                raise ValueError(f"Pooling {train_args.pooling} not supported")
        
        if train_args.grad_cache:
            if not _gradcache_available:
                raise ValueError("gradcache is not available. Please install gradcache")
            fwd_fn = cachex.grad_cached(fwd_fn, None)
        
        
        def compute_loss(lora_params, params, qq, pp):
            params = lora.apply(params, lora_params)
            hq = fwd_fn(params, qq)
            hp = fwd_fn(params, pp)
            return contrastive_loss(hq, hp, scale_by_dim=train_args.scale_by_dim).mean()


        loss, grads = jax.value_and_grad(compute_loss, argnums=0) (lora_params, params, queries, passages)
        metrics = {"loss": loss}

        updates, new_opt_state = optimizer.update(grads, opt_state, lora_params)
        new_lora_params = optax.apply_updates(lora_params, updates)

        return new_lora_params, new_opt_state, metrics

    train_step_cached = partial(
        train_step_cached,
        query_num_chunks=train_args.query_num_chunks,
        passage_num_chunks=train_args.passage_num_chunks
    )
    p_train_step = jax.jit(
        train_step_cached,
        donate_argnums=(1,2),
        out_shardings=(magix.item_sharding(lora_params), magix.item_sharding(opt_state), None),
    )
    
    # train loop
    lastest_step = checkpoint_manager.latest_step()
    if lastest_step is None:
        lastest_step = -1
        
    train_metrics = []

    def combine_metrics(list_of_dicts):
        return {key: jnp.array([d[key] for d in list_of_dicts]) for key in list_of_dicts[0]}
    
    
    epochs = tqdm(range(train_args.num_epochs), desc=f"Epoch ... (1/{train_args.num_epochs})", position=0)
    
    logger.info("Starting training loop...")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", train_args.num_epochs)
    logger.info("  Instantaneous batch size = %d", train_args.batch_size)
    
    
    with mesh:
        for epoch in epochs:
            # ======================== Training ================================

            # Create sampling rng
            input_rng = jax.random.fold_in(data_rng, epoch)

            batch_loader = Batches(input_rng, train_dataset, train_args.batch_size, epoch, shuffle=True)
            steps_per_epoch = len(train_dataset) // train_args.batch_size
            # train
            for step in trange(steps_per_epoch):
                cur_step = epoch * (len(train_dataset) // train_args.batch_size) + step
                if lastest_step >= cur_step:
                    continue
                elif lastest_step == cur_step:
                    logger.info('Resuming training from step %d', cur_step)
                
                qq, pp = batch_loader(step)
                
                dropout_rngs = jax.random.fold_in(dropout_rng, cur_step)
                lora_params, opt_state, metrics = p_train_step(params, lora_params, opt_state, qq, pp, dropout_rngs)
                
                is_last_step = (cur_step + 1) == total_train_steps
                checkpoint_manager.save(
                    cur_step,
                    items={'lora': lora_params, 'optimizer': opt_state},
                    force=is_last_step
                )
                train_metrics.append(metrics)
                
                if cur_step % 100 == 0 and cur_step > 0:
                    print(
                        f"Step... ({cur_step} | Loss: {combine_metrics(train_metrics)['loss'].mean()}, Learning Rate: {linear_decay_lr_schedule_fn(cur_step)})",
                        flush=True,
                    )
                    train_metrics = []

            epochs.write(
                    f"Epoch... ({epoch + 1}/{train_args.num_epochs})"
                )

if __name__ == '__main__':
    main()