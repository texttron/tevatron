"""Reranker scoring backends.

A *backend* turns (query, passage) pairs into relevance scores. Two ship here,
sharing the prompt/template/yes-no plumbing in this package:

- ``hf``   — HF ``AutoModelForSequenceClassification`` (single-logit score head),
             DDP-sharded; exact-match to ``tevatron.reranker`` training math.
- ``vllm`` — causal-LM yes/no log-odds scoring via vLLM, for Megatron-trained
             (or any yes/no) checkpoints.

Both expose the same shape: a ``*RerankConfig`` dataclass and a ``run(cfg)``
entrypoint. The thin CLIs ``tevatron.eval.rerank`` / ``tevatron.eval.rerank_vllm``
and the serving pool (``tevatron.eval.serve``) drive them.

Imports are lazy so that pulling one backend's config does not import the other's
heavy deps (torch DDP for ``hf``, vLLM for ``vllm``).
"""

__all__ = [
    "HFRerankConfig",
    "VLLMRerankConfig",
    "hf_run",
    "vllm_run",
]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "HFRerankConfig":
        from tevatron.eval.backends.hf import HFRerankConfig
        return HFRerankConfig
    if name == "hf_run":
        from tevatron.eval.backends.hf import run
        return run
    if name == "VLLMRerankConfig":
        from tevatron.eval.backends.vllm import VLLMRerankConfig
        return VLLMRerankConfig
    if name == "vllm_run":
        from tevatron.eval.backends.vllm import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
