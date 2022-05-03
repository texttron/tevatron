import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import json

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.modeling import EncoderOutput, SpladeModel
from tevatron.datasets import HFQueryDataset, HFCorpusDataset

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

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

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

    model = SpladeModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   tokenizer, max_len=text_max_length)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    collection_file = open(data_args.encoded_save_path, "w")

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch)
                    reps = model_output.q_reps.cpu().detach().numpy()
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    reps = model_output.p_reps.cpu().detach().numpy()
                for rep, id_ in zip(reps, batch_ids):
                    idx = np.nonzero(rep)
                    # then extract values:
                    data = rep[idx]
                    data = np.rint(data * 100).astype(int)
                    dict_splade = dict()
                    for id_token, value_token in zip(idx[0],data):
                        if value_token > 0:
                            real_token = vocab_dict[id_token]
                            dict_splade[real_token] = int(value_token)
                    if len(dict_splade.keys()) == 0:
                        print("empty input =>", id_)
                        dict_splade[vocab_dict[998]] = 1  # in case of empty doc we fill with "[unused993]" token (just to fill
                        # and avoid issues with anserini), in practice happens just a few times ...
                    if not data_args.encode_is_qry:
                        dict_ = dict(id=id_, content="", vector=dict_splade)
                        json_dict = json.dumps(dict_)  
                        collection_file.write(json_dict + "\n")
                    else:
                        string_splade = " ".join(
                            [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()])
                        collection_file.write(str(id_) + "\t" + string_splade + "\n")
    collection_file.close()




if __name__ == "__main__":
    main()
