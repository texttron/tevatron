import logging
import os
import pickle
import sys

import datasets
import jax
import numpy as np
from flax.training.common_utils import shard
from jax import pmap
from tevatron.arguments import DataArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.arguments import ModelArguments
from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.datasets import HFQueryDataset, HFCorpusDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax.training.train_state import TrainState
from flax import jax_utils
import optax
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          HfArgumentParser, TensorType)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = FlaxAutoModel.from_pretrained(model_args.model_name_or_path, config=config, from_pt=False)

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   tokenizer, max_len=text_max_length)

    # prepare padding batch (for last nonfull batch)
    dataset_size = len(encode_dataset)
    padding_prefix = "padding_"
    total_batch_size = len(jax.devices()) * training_args.per_device_eval_batch_size
    features = list(encode_dataset.encode_data.features.keys())
    padding_batch = {features[0]: [], features[1]: []}
    for i in range(total_batch_size - (dataset_size % total_batch_size)):
        padding_batch["text_id"].append(f"{padding_prefix}{i}")
        padding_batch["text"].append([0])
    padding_batch = datasets.Dataset.from_dict(padding_batch)
    encode_dataset.encode_data = datasets.concatenate_datasets([encode_dataset.encode_data, padding_batch])

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length',
            pad_to_multiple_of=16,
            return_tensors=TensorType.NUMPY,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    # craft a fake state for now to replicate on devices
    adamw = optax.adamw(0.0001)
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

    def encode_step(batch, state):
        embedding = state.apply_fn(**batch, params=state.params, train=False)[0]
        return embedding[:, 0]

    p_encode_step = pmap(encode_step)
    state = jax_utils.replicate(state)

    encoded = []
    lookup_indices = []

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        batch_embeddings = p_encode_step(shard(batch.data), state)
        encoded.extend(np.concatenate(batch_embeddings, axis=0))
    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded[:dataset_size], lookup_indices[:dataset_size]), f)


if __name__ == "__main__":
    main()
