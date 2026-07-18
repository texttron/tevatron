#!/usr/bin/env python
"""Encode raw BEIR JSONL/JSONL.GZ files without materializing HF Arrow caches."""

import argparse
import gzip
import json
import logging
import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.retriever.modeling import DenseModel
from tevatron.utils.io import ensure_parent_dir


logger = logging.getLogger(__name__)


def iter_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_examples(path, encode_is_query, prefix, num_shards, shard_index):
    for row_index, row in enumerate(iter_jsonl(path)):
        if row_index % num_shards != shard_index:
            continue

        content_id = row.get("query_id") or row.get("docid") or row["_id"]
        if encode_is_query:
            text = row.get("query_text", row.get("query", row.get("text", ""))) or ""
        else:
            text = row.get("text", "") or ""
            title = row.get("title", "") or ""
            if title:
                text = f"{title} {text}"
            text = text.strip()
        yield str(content_id), prefix + text


def batched(iterator, batch_size):
    ids = []
    texts = []
    for content_id, text in iterator:
        ids.append(content_id)
        texts.append(text)
        if len(ids) == batch_size:
            yield ids, texts
            ids = []
            texts = []
    if ids:
        yield ids, texts


def encode_batch(tokenizer, texts, max_length, append_eos_token, pad_to_multiple_of):
    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length - 1 if append_eos_token else max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=True,
    )
    if append_eos_token:
        tokenized["input_ids"] = [
            input_ids + [tokenizer.eos_token_id] for input_ids in tokenized["input_ids"]
        ]
    return tokenizer.pad(
        tokenized,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--append_eos_token", action="store_true")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right")
    parser.add_argument("--encode_is_query", action="store_true")
    parser.add_argument("--query_prefix", default="")
    parser.add_argument("--passage_prefix", default="")
    parser.add_argument("--query_max_len", type=int, default=32)
    parser.add_argument("--passage_max_len", type=int, default=128)
    parser.add_argument("--pad_to_multiple_of", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--dataset_number_of_shards", type=int, default=1)
    parser.add_argument("--dataset_shard_index", type=int, default=0)
    parser.add_argument("--encode_output_path", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = args.padding_side

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "torch_dtype": torch_dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = DenseModel.load(
        args.model_name_or_path,
        pooling=args.pooling,
        normalize=args.normalize,
        lora_name_or_path=args.lora_name_or_path,
        **model_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    prefix = args.query_prefix if args.encode_is_query else args.passage_prefix
    max_length = args.query_max_len if args.encode_is_query else args.passage_max_len
    examples = iter_examples(
        args.input,
        args.encode_is_query,
        prefix,
        args.dataset_number_of_shards,
        args.dataset_shard_index,
    )

    encoded = []
    lookup_indices = []
    amp_context = (
        torch.amp.autocast("cuda")
        if device.type == "cuda" and (args.fp16 or args.bf16)
        else nullcontext()
    )

    for batch_ids, batch_texts in tqdm(
        batched(examples, args.per_device_eval_batch_size),
        desc=os.path.basename(args.input),
    ):
        batch = encode_batch(
            tokenizer,
            batch_texts,
            max_length,
            args.append_eos_token,
            args.pad_to_multiple_of,
        )
        lookup_indices.extend(batch_ids)
        with amp_context:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                if args.encode_is_query:
                    model_output = model(query=batch)
                    reps = model_output.q_reps
                else:
                    model_output = model(passage=batch)
                    reps = model_output.p_reps
                encoded.append(reps.cpu().detach().numpy())

    if not encoded:
        raise ValueError(f"No rows were encoded from {args.input}")

    ensure_parent_dir(args.encode_output_path)
    with open(args.encode_output_path, "wb") as f:
        pickle.dump((np.concatenate(encoded), lookup_indices), f)


if __name__ == "__main__":
    main()
