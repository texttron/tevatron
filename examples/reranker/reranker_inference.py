import logging
import os
import pickle
import sys
from contextlib import nullcontext
from unittest import result

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from tevatron.reranker.data import HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from tevatron.reranker.modeling import RerankerModel

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

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
        cache_dir=model_args.cache_dir
    )

    model = RerankerModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    rerank_dataset = HFRerankDataset(tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rerank_dataset = RerankerInferenceDataset(
        rerank_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
        tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=RerankerInferenceCollator(
            tokenizer,
            max_length=data_args.q_max_len+data_args.p_max_len,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    model = model.to(training_args.device)
    model.eval()
    all_results = {}

    for (batch_query_ids, batch_text_ids, batch) in tqdm(rerank_loader):
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                model_output = model(batch)
                scores = model_output.scores.cpu().detach().numpy()
                for i in range(len(scores)):
                    qid = batch_query_ids[i]
                    docid = batch_text_ids[i]
                    score = scores[i][0]
                    if qid not in all_results:
                        all_results[qid] = []
                    all_results[qid].append((docid, score))

    with open(data_args.encoded_save_path, 'w') as f:
        for qid in all_results:
            results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
            for docid, score in results:
                f.write(f'{qid}\t{docid}\t{score}\n')

if __name__ == "__main__":
    main()
