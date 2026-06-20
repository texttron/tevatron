"""Prompt-template registry for causal-LM yes/no rerankers.

Different teacher / student rerankers use different prompt formats and
different yes/no token ids:

- "tevatron" (our own trained checkpoints): plain "query: ... passage: ..."
  format ending in `RERANKER_PROMPT_SUFFIX`, scored on ' yes' / ' no'
  (with a leading space — that's the BPE that follows '?').

- "qwen3_reranker" (Qwen/Qwen3-Reranker-{0.6B,4B,8B}): chat-template wrapped
  prompt ending in "<think>\\n\\n</think>\\n\\n", scored on 'yes' / 'no' (no
  leading space — the chat suffix ends with two newlines).

The registry lets eval / annotation backends select a template by name and
treat both teachers uniformly. Training keeps using the tevatron template
via `tevatron.megatron.data.MegatronRerankerCollator`; only eval-side and
teacher-annotation paths are parameterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from transformers import PreTrainedTokenizer

from tevatron.eval.backends.prompt import build_prompt
from tevatron.eval.backends.score import resolve_yes_no_token_ids
from tevatron.reranker.dataset import format_pair


@dataclass
class PromptTemplate:
    name: str
    build_token_ids: Callable[..., list[int]]
    # Batched form: takes parallel lists (queries, passages, titles) and
    # returns a list of token-id sequences. Uses the HF fast tokenizer's
    # batched mode so Rayon parallelizes inside the Rust impl. ~50x faster
    # than calling build_token_ids in a Python loop on millions of pairs.
    build_token_ids_batch: Callable[..., list[list[int]]]
    resolve_yes_no: Callable[[PreTrainedTokenizer], tuple[int, int]]


def _tevatron_build(
    tokenizer: PreTrainedTokenizer,
    query: str,
    passage: str,
    title: str = "",
    max_len: int = 512,
    *,
    query_prefix: str = "query:",
    passage_prefix: str = "passage:",
) -> list[int]:
    prompt = build_prompt(query, passage, title, query_prefix, passage_prefix)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def _tevatron_build_batch(
    tokenizer: PreTrainedTokenizer,
    queries: list[str],
    passages: list[str],
    titles: list[str],
    max_len: int = 512,
    *,
    query_prefix: str = "query:",
    passage_prefix: str = "passage:",
) -> list[list[int]]:
    prompts = [
        build_prompt(q, p, t, query_prefix, passage_prefix)
        for q, p, t in zip(queries, passages, titles)
    ]
    enc = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return enc["input_ids"]


def _tevatron_template() -> PromptTemplate:
    return PromptTemplate(
        name="tevatron",
        build_token_ids=_tevatron_build,
        build_token_ids_batch=_tevatron_build_batch,
        resolve_yes_no=resolve_yes_no_token_ids,
    )


# --------------------------------------------------------------------------- #
# Sequence-classification reranker (HF AutoModelForSequenceClassification,
# num_labels=1). NOT a yes/no causal-LM scorer: there is no prompt suffix and no
# yes/no token — the score is the single regression logit read at the LAST
# token. Training (tevatron.reranker) builds the pair via `format_pair` (no
# suffix) and appends EOS, then the seq-cls head pools the final position. The
# serve backend must reproduce that exactly: format_pair + EOS, score logits[:,0].
# --------------------------------------------------------------------------- #
def build_seqcls_token_ids_batch(
    tokenizer: PreTrainedTokenizer,
    queries: list[str],
    passages: list[str],
    titles: list[str],
    max_len: int = 512,
    *,
    query_prefix: str = "query:",
    passage_prefix: str = "passage:",
    append_eos: bool = True,
) -> list[list[int]]:
    """Mirror RerankerInferenceCollator: tokenize `format_pair(...)` with
    add_special_tokens=True, truncate to max_len-1 when appending EOS, then
    append eos_token_id. No yes/no suffix."""
    pairs = [
        format_pair(q, p, t, query_prefix, passage_prefix)
        for q, p, t in zip(queries, passages, titles)
    ]
    budget = max_len - 1 if append_eos else max_len
    enc = tokenizer(
        pairs,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=budget,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    if append_eos:
        ids = [seq + [tokenizer.eos_token_id] for seq in ids]
    return ids


# Qwen3-Reranker reference recipe. The instruct line is the default Qwen
# uses in their card; it's a hint, not a hard contract — but staying on the
# canonical wording matches their reported numbers most closely.
_QWEN3_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _qwen3_body(query: str, passage: str, title: str, instruction: str) -> str:
    title = (title or "").strip()
    doc = f"{title} {passage}".strip() if title else passage
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


def _qwen3_reranker_build(
    tokenizer: PreTrainedTokenizer,
    query: str,
    passage: str,
    title: str = "",
    max_len: int = 8192,
    *,
    instruction: str = _QWEN3_INSTRUCT,
    **_unused,
) -> list[int]:
    prefix_ids = tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)
    body = _qwen3_body(query, passage, title, instruction)
    body_budget = max_len - len(prefix_ids) - len(suffix_ids)
    if body_budget <= 0:
        raise ValueError(
            f"max_len={max_len} too small for qwen3_reranker template "
            f"(prefix={len(prefix_ids)} + suffix={len(suffix_ids)} alone exceeds it)"
        )
    body_ids = tokenizer.encode(body, add_special_tokens=False)
    if len(body_ids) > body_budget:
        body_ids = body_ids[:body_budget]
    return prefix_ids + body_ids + suffix_ids


def _qwen3_reranker_build_batch(
    tokenizer: PreTrainedTokenizer,
    queries: list[str],
    passages: list[str],
    titles: list[str],
    max_len: int = 8192,
    *,
    instruction: str = _QWEN3_INSTRUCT,
    **_unused,
) -> list[list[int]]:
    prefix_ids = tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False)
    body_budget = max_len - len(prefix_ids) - len(suffix_ids)
    if body_budget <= 0:
        raise ValueError(
            f"max_len={max_len} too small for qwen3_reranker template "
            f"(prefix={len(prefix_ids)} + suffix={len(suffix_ids)} alone exceeds it)"
        )
    bodies = [_qwen3_body(q, p, t, instruction) for q, p, t in zip(queries, passages, titles)]
    enc = tokenizer(
        bodies,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=body_budget,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [prefix_ids + body_ids + suffix_ids for body_ids in enc["input_ids"]]


def _qwen3_reranker_resolve(tokenizer: PreTrainedTokenizer) -> tuple[int, int]:
    # Qwen3-Reranker scores 'yes'/'no' WITHOUT leading space — the chat
    # suffix ends with "<think>\n\n</think>\n\n" so the next token has no
    # leading space, unlike our tevatron template which ends in '?'.
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if yes_id is None or no_id is None or yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
        raise ValueError(
            f"qwen3_reranker: tokenizer doesn't have single-token 'yes'/'no' "
            f"(got yes_id={yes_id}, no_id={no_id}). Use a Qwen tokenizer."
        )
    return yes_id, no_id


def _qwen3_reranker_template() -> PromptTemplate:
    return PromptTemplate(
        name="qwen3_reranker",
        build_token_ids=_qwen3_reranker_build,
        build_token_ids_batch=_qwen3_reranker_build_batch,
        resolve_yes_no=_qwen3_reranker_resolve,
    )


TEMPLATES: dict[str, PromptTemplate] = {
    "tevatron": _tevatron_template(),
    "qwen3_reranker": _qwen3_reranker_template(),
}


def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        raise KeyError(
            f"Unknown prompt template {name!r}. Available: {sorted(TEMPLATES)}"
        )
    return TEMPLATES[name]


__all__ = ["PromptTemplate", "TEMPLATES", "get_template"]
