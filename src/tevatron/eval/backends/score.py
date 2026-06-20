"""Yes/no token resolution for reranker scoring.

Same logic the Megatron training driver uses (see
tevatron.megatron.driver.train.resolve_yes_no_token_ids), lifted here so the
eval backends don't pull in Megatron.
"""

import logging

from transformers import PreTrainedTokenizer

from tevatron.eval.backends.prompt import RERANKER_PROMPT_SUFFIX

logger = logging.getLogger(__name__)


def resolve_yes_no_token_ids(tokenizer: PreTrainedTokenizer) -> tuple[int, int]:
    """Return (yes_id, no_id) for the contextually-correct continuations.

    The prompt ends with "?" (no trailing space) so the next BPE token is
    " yes" / " no" with a leading space, which has different ids than the
    bare "yes" / "no" tokens. Resolve by tokenizing the full string and
    taking the appended id.
    """
    base = tokenizer.encode(RERANKER_PROMPT_SUFFIX, add_special_tokens=False)
    yes_full = tokenizer.encode(RERANKER_PROMPT_SUFFIX + " yes", add_special_tokens=False)
    no_full = tokenizer.encode(RERANKER_PROMPT_SUFFIX + " no", add_special_tokens=False)
    assert yes_full[: len(base)] == base, f"yes prefix changed: {yes_full} vs {base}"
    assert no_full[: len(base)] == base, f"no prefix changed: {no_full} vs {base}"
    assert len(yes_full) == len(base) + 1, f"' yes' is multi-token: {yes_full[len(base):]}"
    assert len(no_full) == len(base) + 1, f"' no' is multi-token: {no_full[len(base):]}"
    yes_id = yes_full[-1]
    no_id = no_full[-1]
    logger.info("yes_id=%d (%r), no_id=%d (%r)",
                yes_id, tokenizer.decode([yes_id]),
                no_id, tokenizer.decode([no_id]))
    return yes_id, no_id
