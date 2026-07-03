"""DDP rerank backend for Megatron-trained reranker checkpoints.

Loads a causal-LM checkpoint saved via the bridge from `tevatron.megatron.driver.train`,
tokenizes (query, passage) prompts with the RERANKER_PROMPT_SUFFIX yes/no
suffix, and scores each candidate as
    score = logit(' yes') - logit(' no')
at the last attended position. Same math as the Megatron training loss.

DDP: candidates are sharded across ranks via DistributedSampler. Each rank
writes a per-rank shard of (qid, pid, score). Rank 0 merges, sorts per
query desc, and writes the final TREC text.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tevatron.eval.backends.templates import get_template

logger = logging.getLogger(__name__)


@dataclass
class MegatronRerankConfig:
    model_name_or_path: str
    rerank_input: str
    rerank_output: str
    lora_name_or_path: str | None = None
    tokenizer_name: str | None = None
    rerank_max_len: int = 512
    per_device_eval_batch_size: int = 16
    query_prefix: str = "query:"
    passage_prefix: str = "passage:"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    prompt_template: str = "tevatron"


class _RerankCandidates(Dataset):
    """Reads rerank.jsonl into pre-tokenized (qid, pid, ids) tuples."""

    def __init__(
        self,
        path: str,
        tokenizer,
        template_name: str,
        max_len: int,
        query_prefix: str,
        passage_prefix: str,
    ):
        template = get_template(template_name)
        build_kwargs = {}
        if template.name == "tevatron":
            build_kwargs = dict(query_prefix=query_prefix, passage_prefix=passage_prefix)
        self.items: list[tuple[str, str, list[int]]] = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                ids = template.build_token_ids(
                    tokenizer,
                    ex["query"],
                    ex.get("text", ""),
                    ex.get("title", ""),
                    max_len,
                    **build_kwargs,
                )
                self.items.append((str(ex["query_id"]), str(ex["docid"]), ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str, list[int]]:
        return self.items[idx]


class _Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[tuple[str, str, list[int]]]) -> dict:
        qids = [b[0] for b in batch]
        pids = [b[1] for b in batch]
        encoded = [{"input_ids": b[2]} for b in batch]
        padded = self.tokenizer.pad(
            encoded,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {"qids": qids, "pids": pids, **padded}


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def _load_model(cfg: MegatronRerankConfig, device: torch.device):
    dtype = _dtype_from_str(cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.lora_name_or_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.lora_name_or_path)
    model.to(device).eval()
    return model


def _shard_path(output_path: str, rank: int) -> str:
    return f"{output_path}.rank{rank}.tmp"


def _gather_shards(output_path: str, world_size: int) -> None:
    """Rank-0: collect per-rank shards, sort per query desc, write TREC text."""
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


def run(cfg: MegatronRerankConfig) -> None:
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
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    template = get_template(cfg.prompt_template)
    yes_id, no_id = template.resolve_yes_no(tokenizer)
    if rank == 0:
        logger.info("template=%s yes_id=%d no_id=%d", template.name, yes_id, no_id)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.rerank_output)) or ".", exist_ok=True)

    dataset = _RerankCandidates(
        cfg.rerank_input,
        tokenizer,
        cfg.prompt_template,
        cfg.rerank_max_len,
        cfg.query_prefix,
        cfg.passage_prefix,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_device_eval_batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=_Collator(tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    model = _load_model(cfg, device)

    shard_path = _shard_path(cfg.rerank_output, rank)
    written = 0
    with open(shard_path, "w") as out_f:
        with torch.no_grad():
            for batch in tqdm(loader, disable=(rank != 0), desc=f"rerank[{rank}]"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # (B, S, V)

                last_pos = attention_mask.sum(dim=-1) - 1  # (B,)
                B = input_ids.size(0)
                last_logits = logits[torch.arange(B, device=device), last_pos]  # (B, V)
                scores = (last_logits[:, yes_id] - last_logits[:, no_id]).float().cpu().tolist()

                for qid, pid, s in zip(batch["qids"], batch["pids"], scores):
                    out_f.write(f"{qid}\t{pid}\t{s}\n")
                written += B

    logger.info("rank=%d wrote %d candidates to %s", rank, written, shard_path)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        _gather_shards(cfg.rerank_output, world_size)
        logger.info("Final TREC ranklist written to %s", cfg.rerank_output)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
