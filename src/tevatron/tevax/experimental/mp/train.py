import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm, trange
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax

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

from ...loss import contrastive_loss

# TODO: maybe use tevatron TrainDataset instead
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
        # print(indices)
        qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
        dd = sum(dd, [])

        return dict(self.tokenizer(qq, max_length=self.query_max_length, padding='max_length', truncation=True, return_tensors='np')), \
         dict(self.tokenizer(dd, max_length=self.passage_max_length, padding='max_length', truncation=True, return_tensors='np'))

        
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

# TODO: maybe use tevatron args instead
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
    save_steps: int = 200
    seed: int = 42
    pooling: str = 'bos'
    scale_by_dim: bool = True
    
    def __post_init__(self):
        assert self.pooling in ['bos', 'eos', 'cls']
    
@dataclass
class ModelArgs:
    model_name: str = None
    model_type: str = None
    mesh_shape: List[int] = list_field(-1, 1)

def main():
    parser = ArgumentParser()
    parser.add_arguments(TrainArgs, dest="train_args")
    parser.add_arguments(ModelArgs, dest="model_args")
    args = parser.parse_args()
    train_args: TrainArgs = args.train_args
    model_args: ModelArgs = args.model_args
    
    # logger with date and time
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
    tokenizer = AutoTokenizer.from_pretrained(
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
    
    # initalize model parameters and optimizer state
    mesh = magix.create_device_mesh(model_args.mesh_shape)
    
    checkpoint_manager = magix.get_chckpoint_manager(train_args.checkpoint_dir, train_args.save_steps)
    is_new_train = checkpoint_manager.latest_step() is None
    _model_cls = ENCODER_MODEL_MAPPING.get(model_args.model_type, None)
    if _model_cls is None:
        raise ValueError(f"Model {model_args.model_name} not supported")
    sharding_config = _model_cls.partition_rules
    if is_new_train:
        logger.info("Loading model from hub")
        model, params = magix.load_model_hub(_model_cls, model_args.model_name, sharding_config, mesh)
        opt_state = magix.initialize_opt_state(optimizer, params, sharding_config, mesh)
    else:
        logger.info("Loading model from checkpoint")
        model, params, opt_state = magix.load_model_and_optimizer_local(
            _model_cls,
            optimizer,
            checkpoint_manager,
            sharding_config,
            mesh,
            model_name=model_args.model_name
        )
    
    def train_step(params, opt_state, queries, passages, dropout_rng, query_num_chunks=4, passage_num_chunks=8):
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
        
        def compute_loss(params, qq, pp):
            hq = fwd_fn(params, qq)
            hp = fwd_fn(params, pp)
            return contrastive_loss(hq, hp, scale_by_dim=train_args.scale_by_dim).mean()

        loss, grads = jax.value_and_grad(compute_loss, argnums=0) (params, queries, passages)
        metrics = {"loss": loss}

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    train_step = partial(
        train_step,
        query_num_chunks=train_args.query_num_chunks,
        passage_num_chunks=train_args.passage_num_chunks
    )
    p_train_step = jax.jit(
        train_step,
        donate_argnums=(0,1,2,3,4),
        out_shardings=(magix.item_sharding(params), magix.item_sharding(opt_state), None)
    )
    
    rng = jax.random.key(train_args.seed)
    dropout_rng, data_rng = jax.random.split(rng)
    
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
                params, opt_state, metrics = p_train_step(params, opt_state, qq, pp, dropout_rngs)
                
                is_last_step = (cur_step + 1) == total_train_steps
                checkpoint_manager.save(
                    cur_step, items={'model': params, 'optimizer': opt_state}, force=is_last_step
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