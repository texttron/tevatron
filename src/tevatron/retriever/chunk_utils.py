import logging

logger = logging.getLogger(__name__)


def chunk_passage_text(text, tokenizer, data_args):
    """
    Utility to split text into overlapping chunks according to chunk+MaxSim settings.
    """
    if not getattr(data_args, "use_chunk_maxsim", False):
        return [text]

    if tokenizer is None:
        logger.warning("Tokenizer is not provided; falling back to unchunked passage.")
        return [text]

    text = text or ''
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )
    if not tokens:
        return [text]

    max_len = getattr(data_args, "chunk_max_length", 0) or 0
    overlap = getattr(data_args, "chunk_overlap", 0) or 0

    if max_len <= 0:
        return [text]

    if overlap < 0:
        overlap = 0
    if overlap >= max_len:
        overlap = max_len // 2

    chunks = []
    n = len(tokens)
    start = 0
    while start < n:
        end = min(start + max_len, n)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        chunks.append(chunk_text)
        if end >= n:
            break
        start = end - overlap

    return chunks or [text]

