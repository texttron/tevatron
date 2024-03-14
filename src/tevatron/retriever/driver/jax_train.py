import logging
import os
import sys
from functools import partial

import datasets
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.jax_utils import prefetch_to_device
from flax.training.common_utils import get_metrics, shard
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModel
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.tevax.training import TiedParams, RetrieverTrainState, retriever_train_step, grad_cache_train_step, \
    DualParams

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TevatronTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    try:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    except:
        model = FlaxAutoModel.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype),
            from_pt=True
        )

    if data_args.train_dir:
        data_files = {
            'train': data_args.train_path
        }
    else:
        data_files = None

    train_dataset = \
        datasets.load_dataset(data_args.dataset_name, data_args.dataset_language, cache_dir=model_args.cache_dir,
                              data_files=data_files)[data_args.dataset_split]

    def tokenize_train(example):
        tokenize = partial(tokenizer, return_attention_mask=False, return_token_type_ids=False, padding=False,
                           truncation=True)
        query = example['query']
        pos_psgs = [p['title'] + " " + p['text'] for p in example['positive_passages']]
        neg_psgs = [p['title'] + " " + p['text'] for p in example['negative_passages']]

        example['query_input_ids'] = dict(tokenize(query, max_length=data_args.q_max_len))
        example['pos_psgs_input_ids'] = [dict(tokenize(x, max_length=data_args.p_max_len)) for x in pos_psgs]
        example['neg_psgs_input_ids'] = [dict(tokenize(x, max_length=data_args.p_max_len)) for x in neg_psgs]

        return example

    train_data = train_dataset.map(
        tokenize_train,
        batched=False,
        num_proc=data_args.dataset_proc_num,
        desc="Running tokenizer on train dataset",
    )
    train_data = train_data.filter(
        function=lambda data: len(data["pos_psgs_input_ids"]) >= 1 and \
                              len(data["neg_psgs_input_ids"]) >= data_args.train_n_passages-1, num_proc=64
    )

    class TrainDataset:
        def __init__(self, train_data, group_size, tokenizer):
            self.group_size = group_size
            self.data = train_data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def get_example(self, i, epoch):
            example = self.data[i]
            q = example['query_input_ids']

            pp = example['pos_psgs_input_ids']
            p = pp[0]

            nn = example['neg_psgs_input_ids']
            off = epoch * (self.group_size - 1) % len(nn)
            nn = nn * 2
            nn = nn[off: off + self.group_size - 1]

            return q, [p] + nn

        def get_batch(self, indices, epoch):
            qq, dd = zip(*[self.get_example(i, epoch) for i in map(int, indices)])
            dd = sum(dd, [])
            return dict(tokenizer.pad(qq, max_length=32, padding='max_length', return_tensors='np')), dict(
                tokenizer.pad(dd, max_length=data_args.p_max_len, padding='max_length', return_tensors='np'))

    train_dataset = TrainDataset(train_data, data_args.train_n_passages, tokenizer)

    def create_learning_rate_fn(
            train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int,
            learning_rate: float
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

    def _decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale") for name in ["self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
        ]
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    def decay_mask_fn(params):
        param_nodes, treedef = jax.tree_flatten(params, lambda v: isinstance(v, dict))
        masks = [_decay_mask_fn(param_node) for param_node in param_nodes]
        return jax.tree_unflatten(treedef, masks)

    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        int(training_args.num_train_epochs),
        int(total_train_steps * 0.1),
        training_args.learning_rate,
    )

    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    if model_args.untie_encoder:
        params = DualParams.create(model.params)
    else:
        params = TiedParams.create(model.params)
    state = RetrieverTrainState.create(apply_fn=model.__call__, params=params, tx=adamw)

    if training_args.grad_cache:
        q_n_subbatch = train_batch_size // training_args.gc_q_chunk_size
        p_n_subbatch = train_batch_size * data_args.train_n_passages // training_args.gc_p_chunk_size
        p_train_step = jax.pmap(
            partial(grad_cache_train_step, q_n_subbatch=q_n_subbatch, p_n_subbatch=p_n_subbatch),
            "device"
        )
    else:
        p_train_step = jax.pmap(
            retriever_train_step,
            "device"
        )

    state = jax_utils.replicate(state)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    class IterableTrain(IterableDataset):
        def __init__(self, dataset, batch_idx, epoch):
            super(IterableTrain).__init__()
            self.dataset = dataset
            self.batch_idx = batch_idx
            self.epoch = epoch

        def __iter__(self):
            for idx in self.batch_idx:
                batch = self.dataset.get_batch(idx, self.epoch)
                batch = shard(batch)
                yield batch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_metrics = []
    for epoch in tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0):
        # ======================== Training ================================
        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        steps_per_epoch = len(train_dataset) // train_batch_size

        batch_idx = jax.random.permutation(input_rng, len(train_dataset))
        batch_idx = batch_idx[: steps_per_epoch * train_batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, train_batch_size)).tolist()

        train_loader = prefetch_to_device(
            iter(DataLoader(
                IterableTrain(train_dataset, batch_idx, epoch),
                num_workers=16, prefetch_factor=256, batch_size=None, collate_fn=lambda v: v)
            ), 2)

        # train
        epochs = tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False)
        for step in epochs:
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            batch = next(train_loader)

            loss, state, dropout_rngs = p_train_step(state, *batch, dropout_rngs)
            train_metrics.append({'loss': loss})

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metrics = get_metrics(train_metrics)
                print(
                    f"Step... ({cur_step} | Loss: {train_metrics['loss'].mean()},"
                    f" Learning Rate: {linear_decay_lr_schedule_fn(cur_step)})",
                    flush=True,
                )
                train_metrics = []

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs})"
        )

    params = jax_utils.unreplicate(state.params)

    if model_args.untie_encoder:
        os.makedirs(training_args.output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(training_args.output_dir, 'query_encoder'), params=params.q_params)
        model.save_pretrained(os.path.join(training_args.output_dir, 'passage_encoder'), params=params.p_params)
    else:
        model.save_pretrained(training_args.output_dir, params=params.p_params)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
