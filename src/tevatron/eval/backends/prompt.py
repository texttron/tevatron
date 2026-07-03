"""Shared reranker prompt construction.

Single source of truth for the prompt template used by both training (Megatron
collator) and eval (HF / vLLM backends). Importing from one place prevents
train/eval drift on the suffix or the spacing.
"""

from tevatron.reranker.dataset import format_pair

RERANKER_PROMPT_SUFFIX = "\nIs the passage relevant to the query?"

__all__ = ["RERANKER_PROMPT_SUFFIX", "format_pair", "build_prompt"]


def build_prompt(
    query: str,
    passage: str,
    title: str = "",
    query_prefix: str = "",
    passage_prefix: str = "",
) -> str:
    """Construct the full reranker prompt ending in the yes/no suffix.

    Mirrors the train-time format exactly:
        "<qprefix> <query> <pprefix> <title> <passage>" + RERANKER_PROMPT_SUFFIX

    The model is expected to produce ' yes' / ' no' as the next BPE token.
    """
    pair = format_pair(query, passage, title, query_prefix, passage_prefix)
    return pair + RERANKER_PROMPT_SUFFIX
