"""Distillation dataset + collator.

Reads a HF dataset produced by `tevatron.utils.annotate_with_teacher`:

    {
        "query_id": str,
        "query": str,
        "passages": [{"title": str, "text": str}, ...],   # length P
        "scores":   [float, ...],                          # length P
    }

`__getitem__` samples `train_group_size` passages per row (always keeping
`passages[0]` — the source positive — at index 0 to match the contrastive
trainer's invariant) and returns the formatted strings + matching teacher
scores. The collator extends `MegatronRerankerCollator` with one extra field
`teacher_scores: (B,) float32` that rides per-microbatch alongside
`attention_mask`, so every PP rank sees it (no P2P needed).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from tevatron.eval.backends.prompt import RERANKER_PROMPT_SUFFIX
from tevatron.reranker.dataset import format_pair

logger = logging.getLogger(__name__)


class MegatronRerankerDistilDataset(TorchDataset):
    """Sampled listwise distill dataset.

    Per row, returns `train_group_size` (formatted_pair, teacher_score) tuples.
    `train_group_size` defaults to len(passages) if smaller is available, else
    samples deterministically from the negative pool.
    """

    def __init__(
        self,
        dataset_path: str,
        train_group_size: int,
        query_prefix: str,
        passage_prefix: str,
        seed: int = 42,
        trainer=None,
    ):
        self.data: Dataset = load_from_disk(dataset_path)
        self.train_group_size = train_group_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.seed = seed
        self.trainer = trainer

        if "passages" not in self.data.column_names or "scores" not in self.data.column_names:
            raise ValueError(
                f"Distill dataset at {dataset_path} missing 'passages'/'scores' columns. "
                f"Got: {self.data.column_names}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        row = self.data[item]
        epoch = int(self.trainer.state.epoch) if self.trainer is not None else 0
        hashed_seed = hash((item, self.seed))

        query: str = row["query"]
        passages: list[dict] = row["passages"]
        scores: list[float] = row["scores"]
        if len(passages) != len(scores):
            raise ValueError(
                f"Row {item}: passages ({len(passages)}) and scores ({len(scores)}) "
                "must have identical length."
            )
        if len(passages) == 0:
            raise ValueError(f"Row {item}: empty passages.")

        # Always keep passages[0] (source positive) at the front. Negatives
        # are everything else; sample (train_group_size - 1) deterministically.
        pos_p, pos_s = passages[0], scores[0]
        neg_pairs = list(zip(passages[1:], scores[1:]))

        n_negs = self.train_group_size - 1
        if n_negs <= 0:
            chosen = [(pos_p, pos_s)]
        elif len(neg_pairs) >= n_negs:
            rng = random.Random(hashed_seed + epoch)
            offset = (epoch * n_negs) % len(neg_pairs)
            shuffled = list(neg_pairs)
            rng.shuffle(shuffled)
            shuffled = shuffled * 2  # cheap circular slicing
            chosen = [(pos_p, pos_s)] + shuffled[offset: offset + n_negs]
        else:
            # Pad with random repeats — same fallback as RerankerTrainDataset.
            rng = random.Random(hashed_seed + epoch)
            chosen = [(pos_p, pos_s)] + rng.choices(neg_pairs, k=n_negs)

        formatted = [
            format_pair(query, p["text"], p.get("title", ""),
                        self.query_prefix, self.passage_prefix)
            for p, _ in chosen
        ]
        teacher_scores = [float(s) for _, s in chosen]
        return formatted, teacher_scores


@dataclass
class MegatronRerankerDistilCollator:
    """Distill collator: same tokenization as MegatronRerankerCollator, plus teacher_scores.

    Mirrors the contrastive collator exactly so we can swap them in/out without
    touching the engine's data path. The only added field is `teacher_scores`,
    a flat `(B,)` float tensor where `B = num_queries * train_group_size` and
    the order matches the flattened pair order.
    """

    tokenizer: PreTrainedTokenizer
    max_seq_len: int = 512
    pad_to_multiple_of: int = 16

    def __call__(self, features: List[tuple]) -> dict:
        all_pairs: list[str] = []
        all_scores: list[float] = []
        for formatted, teacher_scores in features:
            assert len(formatted) == len(teacher_scores), \
                f"formatted/teacher_scores length mismatch: {len(formatted)} vs {len(teacher_scores)}"
            all_pairs.extend(formatted)
            all_scores.extend(teacher_scores)

        all_pairs = [text + RERANKER_PROMPT_SUFFIX for text in all_pairs]

        encoded = self.tokenizer(
            all_pairs,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        padded = self.tokenizer.pad(
            encoded,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch_size, seq_len = padded["input_ids"].shape
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
        teacher_scores = torch.tensor(all_scores, dtype=torch.float32)

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "position_ids": position_ids,
            "teacher_scores": teacher_scores,
        }
