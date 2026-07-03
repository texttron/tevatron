from dataclasses import dataclass
from typing import List

import torch
from transformers import PreTrainedTokenizer

from tevatron.eval.backends.prompt import RERANKER_PROMPT_SUFFIX


@dataclass
class MegatronRerankerCollator:
    """Collator that tokenizes reranker pairs into Megatron-compatible batch format.

    Each input pair is already formatted by `tevatron.reranker.dataset.format_pair`
    as "{query_prefix} {query} {passage_prefix} {title} {passage}" (default
    prefixes: "query:" / "passage:"). This collator appends RERANKER_PROMPT_SUFFIX
    so the full prompt becomes:

        "query: <q> passage: <title> <text>\nIs the passage relevant to the query?"

    The model is trained to predict " yes" / " no" as the next token at the
    final position.
    """

    tokenizer: PreTrainedTokenizer
    max_seq_len: int = 512
    pad_to_multiple_of: int = 16

    def __call__(self, features: List[List[str]]) -> dict:
        """
        Args:
            features: list of samples, each sample is a list of group_size formatted strings.

        Returns:
            dict with input_ids, attention_mask, position_ids as contiguous tensors.
        """
        all_pairs = []
        for pairs in features:
            all_pairs.extend(pairs)

        # Append the relevance prompt suffix to each pair
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

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "position_ids": position_ids,
        }
