"""Wire contract shared by the rerank scoring server and client.

The server owns the model + tokenizer + prompt template; the client is a thin
shuffler of candidate records. A request is a batch of (query_id, docid, query,
text, title) candidates; the response is the yes/no logit-difference score for
each, in the same order. Both vLLM and HF backends return the identical scalar
    score = logit(' yes') - logit(' no')
so client-side aggregation is backend-agnostic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Candidate(BaseModel):
    query_id: str
    docid: str
    query: str
    text: str = ""
    title: str = ""


class ScoreRequest(BaseModel):
    candidates: list[Candidate]
    # Tevatron-template prefixes (ignored by templates that hard-code their own,
    # e.g. qwen3_reranker). Defaults match the training collator.
    query_prefix: str = "query:"
    passage_prefix: str = "passage:"


class ScoreItem(BaseModel):
    query_id: str
    docid: str
    score: float
    # True when ' yes'/' no' fell outside the vLLM top-K and a sentinel was used
    # (always False for the HF backend, which reads raw logits). Lets the client
    # surface degenerate scoring (e.g. the all-0.0 failure mode).
    missing: bool = False


class ScoreResponse(BaseModel):
    scores: list[ScoreItem]


class InfoResponse(BaseModel):
    backend: str               # "vllm" | "hf"
    model_name_or_path: str
    prompt_template: str
    yes_id: int
    no_id: int
    tensor_parallel_size: Optional[int] = None
    world_size: Optional[int] = None
