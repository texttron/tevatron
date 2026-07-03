"""HF transformers + DDP rerank backend for the tevatron reranker baseline.

Loads a checkpoint trained via `tevatron.reranker.driver.train`, which is a
sequence-classification model with a single-logit `score` head. Score for
each (query, passage) pair = `model(pair).scores[:, 0]`.

Prompt construction reuses `tevatron.reranker.dataset.format_pair` to keep
the eval and training input formats in sync.

DDP: candidates are sharded across ranks via DistributedSampler. Each rank
writes a per-rank shard of (qid, pid, score). Rank 0 merges, sorts per
query desc, and writes the final TREC text.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.reranker.dataset import format_pair
from tevatron.reranker.modeling import RerankerModel

logger = logging.getLogger(__name__)


@dataclass
class HFRerankConfig:
    model_name_or_path: str
    rerank_input: str
    rerank_output: str
    lora_name_or_path: str | None = None
    tokenizer_name: str | None = None
    rerank_max_len: int = 512
    per_device_eval_batch_size: int = 16
    query_prefix: str = "query:"
    passage_prefix: str = "passage:"
    append_eos_token: bool = False
    pad_to_multiple_of: int | None = None


class _RerankCandidates(Dataset):
    """Reads rerank.jsonl into (qid, pid, formatted_pair) tuples."""

    def __init__(self, path: str, query_prefix: str, passage_prefix: str):
        self.items: list[tuple[str, str, str]] = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                pair = format_pair(
                    query=ex["query"],
                    passage=ex.get("text", ""),
                    title=ex.get("title", ""),
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                )
                self.items.append((str(ex["query_id"]), str(ex["docid"]), pair))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.items[idx]


class _Collator:
    def __init__(self, tokenizer, max_len: int, append_eos_token: bool, pad_to_multiple_of: int | None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.append_eos_token = append_eos_token
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[tuple[str, str, str]]) -> dict:
        qids = [b[0] for b in batch]
        pids = [b[1] for b in batch]
        pairs = [b[2] for b in batch]
        truncate_len = self.max_len - 1 if self.append_eos_token else self.max_len
        enc = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            max_length=truncate_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.append_eos_token:
            enc["input_ids"] = [x + [self.tokenizer.eos_token_id] for x in enc["input_ids"]]
        enc = self.tokenizer.pad(
            enc,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {"qids": qids, "pids": pids, **enc}


def _shard_path(output_path: str, rank: int) -> str:
    return f"{output_path}.rank{rank}.tmp"


def _gather_shards(output_path: str, world_size: int) -> None:
    by_qid: dict[str, list[tuple[str, float]]] = {}
    for r in range(world_size):
        sp = _shard_path(output_path, r)
        with open(sp) as f:
            for line in f:
                qid, pid, score = line.rstrip().split("\t")
                by_qid.setdefault(qid, []).append((pid, float(score)))
        os.remove(sp)

    with open(output_path, "w") as f:
        for qid, hits in by_qid.items():
            hits.sort(key=lambda x: x[1], reverse=True)
            for pid, score in hits:
                f.write(f"{qid}\t{pid}\t{score}\n")


def run(cfg: HFRerankConfig) -> None:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer_name = cfg.tokenizer_name or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.rerank_output)) or ".", exist_ok=True)

    dataset = _RerankCandidates(cfg.rerank_input, cfg.query_prefix, cfg.passage_prefix)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_eval_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=_Collator(tokenizer, cfg.rerank_max_len, cfg.append_eos_token, cfg.pad_to_multiple_of),
        num_workers=2,
        pin_memory=True,
    )

    model = RerankerModel.load(
        cfg.model_name_or_path,
        lora_name_or_path=cfg.lora_name_or_path,
    ).to(device)
    # Match the tokenizer's pad id — see tevatron/reranker/driver/train.py for
    # the full explanation. Without this, the seq-cls head reads position 0
    # instead of the last non-pad token under right-padding.
    model.hf_model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    shard_path = _shard_path(cfg.rerank_output, rank)
    written = 0
    with open(shard_path, "w") as out_f:
        with torch.no_grad():
            for batch in tqdm(loader, disable=(rank != 0), desc=f"rerank[{rank}]"):
                qids = batch.pop("qids")
                pids = batch.pop("pids")
                batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model(batch_on_device)
                scores = out.scores[:, 0].float().cpu().tolist()

                for qid, pid, s in zip(qids, pids, scores):
                    out_f.write(f"{qid}\t{pid}\t{s}\n")
                written += len(qids)

    logger.info("rank=%d wrote %d candidates to %s", rank, written, shard_path)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        _gather_shards(cfg.rerank_output, world_size)
        logger.info("Final TREC ranklist written to %s", cfg.rerank_output)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
