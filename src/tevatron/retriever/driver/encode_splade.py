"""Standalone SPLADE encoder for Tevatron-v2.

The dense `encode.py` driver hardcodes `DenseModel.load` and dumps a single
dense matrix per shard via pickle. SPLADE produces a *sparse* vocab-space
vector per item, so it needs its own writer. This driver mirrors `encode.py`'s
data path (same `EncodeDataset` / `EncodeCollator`) but swaps in `SpladeModel`
and writes sparse output in two interchangeable formats:

  Corpus (one JSON line per doc):
    {"id": "<docid>", "content": "", "vector": {"<token>": <weight>, ...}}
  Query (one TSV line per query):
    float:  "<qid>\\t{\"<token>\": <weight>, ...}"      (JSON dict)
    int:    "<qid>\\t<token> <token> <token2> ..."      (repeated tokens)

Weight precision is selected by `--splade_weight_format`:
  - float (default): raw float weights, for the PySeismic retrieval path
    (`tevatron.retriever.driver.search_splade`). Preserves precision.
  - int: round(weight * `--splade_quant_factor`) integer weights, for the
    Lucene/Anserini impact-index path. Lossy but Anserini-compatible.

The token strings are the tokenizer's vocab surface forms (same convention as
the legacy encoder and Anserini's impact index), so the two retrieval backends
see identical token keys.

Launch (one shard per GPU for the corpus; single process for queries):

    python -m tevatron.retriever.driver.encode_splade \\
        --model_name_or_path naver/splade-v3 \\
        --dataset_name Tevatron/beir-corpus --dataset_config scifact \\
        --dataset_split train --fp16 --passage_max_len 512 \\
        --per_device_eval_batch_size 32 \\
        --dataset_number_of_shards 8 --dataset_shard_index 0 \\
        --encode_output_path .../scifact/corpus/split0.jsonl
"""

import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, HfArgumentParser

from tevatron.retriever.arguments import (
    DataArguments,
    ModelArguments,
    TevatronTrainingArguments as TrainingArguments,
)
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.modeling import EncoderOutput, SpladeModel, SpladeModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class SpladeEncodeArguments:
    splade_weight_format: str = field(
        default="float",
        metadata={"help": "float (PySeismic, raw weights) or int (Anserini, "
                          "round(w*quant_factor)).", "choices": ["float", "int"]},
    )
    splade_quant_factor: int = field(
        default=100,
        metadata={"help": "Integer quantization multiplier when "
                          "--splade_weight_format int (legacy used 100)."},
    )
    splade_model_type: str = field(
        default="mlm",
        metadata={"help": "Backbone family: 'mlm' (BERT-family AutoModelForMaskedLM, "
                          "the original SPLADE; default) or 'causal' (decoder LM, the "
                          "LACONIC variant).", "choices": ["mlm", "causal"]},
    )
    splade_topk: int = field(
        default=0,
        metadata={"help": "If >0, keep only the top-k highest-weight terms per "
                          "vector before writing (LACONIC used 512 for decoder "
                          "SPLADE, whose dense logits activate far more terms than "
                          "an MLM backbone). 0 = keep all positive weights."},
    )


def _to_token_dict(rep: np.ndarray, id2tok: dict, fmt: str, quant: int, topk: int = 0) -> dict:
    """Sparse vocab vector -> {token_surface: weight}, dropping zeros.

    float: keep raw positive weights.
    int:   round(weight * quant), keep only strictly-positive results.
    topk:  if >0, restrict to the k highest-weight dims before the format/zero
           filtering (the decoder-SPLADE term cap; see ``--splade_topk``).
    """
    if topk and topk < rep.shape[0]:
        keep = np.argpartition(rep, -topk)[-topk:]
        nz = keep[rep[keep] > 0]
    else:
        nz = np.nonzero(rep)[0]
    out = {}
    if fmt == "int":
        for tok_id in nz:
            w = int(np.rint(float(rep[tok_id]) * quant))
            if w > 0:
                out[id2tok[int(tok_id)]] = w
    else:
        for tok_id in nz:
            w = float(rep[tok_id])
            if w > 0.0:
                out[id2tok[int(tok_id)]] = w
    return out


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SpladeEncodeArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, splade_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, splade_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported; shard across processes.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" if data_args.padding_side == "right" else "left"

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if splade_args.splade_model_type == "causal":
        model = SpladeModelForCausalLM.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            is_bidirectional=model_args.is_bidirectional,
            pooling_strategy=model_args.pooling_strategy,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        model = SpladeModel.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )

    encode_dataset = EncodeDataset(data_args=data_args)
    encode_collator = EncodeCollator(data_args=data_args, tokenizer=tokenizer)
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    model = model.to(training_args.device)
    model.eval()

    # vocab id -> surface token (same keys for both retrieval backends)
    id2tok = {v: k for k, v in tokenizer.get_vocab().items()}
    fmt = splade_args.splade_weight_format
    quant = splade_args.splade_quant_factor
    is_query = data_args.encode_is_query

    # empty-vector fallback token (avoids Anserini choking on empty docs)
    fallback_tok = id2tok.get(998, next(iter(id2tok.values())))

    os.makedirs(os.path.dirname(os.path.abspath(data_args.encode_output_path)) or ".", exist_ok=True)
    n_written = n_empty = 0
    with open(data_args.encode_output_path, "w") as fout:
        for batch_ids, batch in encode_loader:
            with torch.amp.autocast("cuda") if (training_args.fp16 or training_args.bf16) else nullcontext():
                with torch.no_grad():
                    batch = {k: v.to(training_args.device) for k, v in batch.items()}
                    if is_query:
                        out: EncoderOutput = model(query=batch)
                        reps = out.q_reps.float().cpu().numpy()
                    else:
                        out: EncoderOutput = model(passage=batch)
                        reps = out.p_reps.float().cpu().numpy()

            for rep, _id in zip(reps, batch_ids):
                vec = _to_token_dict(rep, id2tok, fmt, quant, splade_args.splade_topk)
                if not vec:
                    n_empty += 1
                    vec = {fallback_tok: 1 if fmt == "int" else 1.0}
                if is_query:
                    if fmt == "int":
                        # repeated-token form for Anserini query parsing
                        toks = " ".join(" ".join([t] * w) for t, w in vec.items())
                        fout.write(f"{_id}\t{toks}\n")
                    else:
                        fout.write(f"{_id}\t{json.dumps(vec)}\n")
                else:
                    fout.write(json.dumps({"id": _id, "content": "", "vector": vec}) + "\n")
                n_written += 1

    logger.info("Wrote %d %s vectors to %s (empty->fallback: %d, format=%s)",
                n_written, "query" if is_query else "corpus",
                data_args.encode_output_path, n_empty, fmt)


if __name__ == "__main__":
    main()
