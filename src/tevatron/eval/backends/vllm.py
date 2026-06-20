"""vLLM rerank backend for causal-LM reranker checkpoints (yes/no scoring).

Same prompt + scoring math as `tevatron.megatron.eval.rerank_backend`:
    score = logprob(' yes') - logprob(' no')
at the next-token position after the prompt. Subtracting two unconstrained
logprobs equals the raw-logit difference: the global softmax denominator is
shared and cancels in log(p_yes) - log(p_no) = logit_yes - logit_no.

We do NOT use vLLM's `allowed_token_ids` — empirically it makes vLLM return
only one of yes/no per request (whichever was sampled), so the other side is
None and the score collapses. Instead we ask for `logprobs=top_logprobs` and
read both ' yes' and ' no' from the returned top-K. A trained reranker keeps
both in the top-20 after the suffix '?'; missing entries fall back to a
sentinel logprob and get logged as warnings.

Single-process: vLLM manages its own workers via `tensor_parallel_size`.
Do NOT launch this under torchrun — vLLM spawns workers itself.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.eval.backends.templates import PromptTemplate, get_template

logger = logging.getLogger(__name__)


@dataclass
class VLLMRerankConfig:
    model_name_or_path: str
    rerank_input: str
    rerank_output: str
    lora_name_or_path: str | None = None
    tokenizer_name: str | None = None
    rerank_max_len: int = 512
    max_model_len: int = 1024
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    enforce_eager: bool = False
    trust_remote_code: bool = False
    chunk_size: int = 10000
    top_logprobs: int = 20
    missing_fallback_logprob: float = -20.0
    prompt_template: str = "tevatron"


def _load_candidates(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            items.append({
                "query_id": str(ex["query_id"]),
                "docid": str(ex["docid"]),
                "query": ex["query"],
                "text": ex.get("text", ""),
                "title": ex.get("title", ""),
            })
    return items


def run(cfg: VLLMRerankConfig, query_prefix: str = "query:", passage_prefix: str = "passage:") -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    os.makedirs(os.path.dirname(os.path.abspath(cfg.rerank_output)) or ".", exist_ok=True)

    tokenizer_name = cfg.tokenizer_name or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    template = get_template(cfg.prompt_template)
    yes_id, no_id = template.resolve_yes_no(tokenizer)
    logger.info("template=%s yes_id=%d no_id=%d", template.name, yes_id, no_id)

    items = _load_candidates(cfg.rerank_input)
    logger.info("Loaded %d candidates from %s", len(items), cfg.rerank_input)

    # Pre-tokenize + truncate per the chosen template. vLLM accepts strings
    # too, but token-id input avoids a redundant tokenization pass and makes
    # the per-template truncation budget (e.g. qwen3 has fixed prefix/suffix
    # ids) explicit.
    build_kwargs = {}
    if template.name == "tevatron":
        build_kwargs = dict(query_prefix=query_prefix, passage_prefix=passage_prefix)

    queries = [ex["query"] for ex in items]
    passages = [ex["text"] for ex in items]
    titles = [ex["title"] for ex in items]
    prompt_token_ids = template.build_token_ids_batch(
        tokenizer,
        queries,
        passages,
        titles,
        cfg.rerank_max_len,
        **build_kwargs,
    )

    # vLLM caps `SamplingParams.logprobs` at engine-init `max_logprobs`
    # (default 20). For distillation annotation we want top-200 to keep
    # `' yes'`/`' no'` inside the cutoff even when one is at p~1; raise
    # the engine cap to match the per-request top-K.
    llm_kwargs = dict(
        model=cfg.model_name_or_path,
        tokenizer=tokenizer_name,
        tensor_parallel_size=cfg.tensor_parallel_size,
        dtype=cfg.dtype,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        enforce_eager=cfg.enforce_eager,
        trust_remote_code=cfg.trust_remote_code,
        max_model_len=cfg.max_model_len,
        max_logprobs=cfg.top_logprobs,
    )
    if cfg.lora_name_or_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        logprobs=cfg.top_logprobs,
    )

    lora_req = (
        LoRARequest("rerank_lora", 1, cfg.lora_name_or_path)
        if cfg.lora_name_or_path
        else None
    )

    by_qid: dict[str, list[tuple[str, float]]] = {}
    missing = 0
    n = len(items)
    for start in range(0, n, cfg.chunk_size):
        end = min(start + cfg.chunk_size, n)
        chunk_ids = prompt_token_ids[start:end]
        chunk_items = items[start:end]
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in chunk_ids]
        logger.info("vLLM generate chunk %d:%d / %d", start, end, n)
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling,
            use_tqdm=True,
            lora_request=lora_req,
        )
        for ex, out in zip(chunk_items, outputs):
            qid, pid = ex["query_id"], ex["docid"]
            lp_map = out.outputs[0].logprobs[0]  # {token_id: Logprob(...)}
            lp_yes = lp_map.get(yes_id)
            lp_no = lp_map.get(no_id)
            if lp_yes is None or lp_no is None:
                missing += 1
            ly = lp_yes.logprob if lp_yes is not None else cfg.missing_fallback_logprob
            ln = lp_no.logprob if lp_no is not None else cfg.missing_fallback_logprob
            score = float(ly - ln)
            by_qid.setdefault(qid, []).append((pid, score))

    if missing:
        logger.warning(
            "%d/%d candidates had ' yes' or ' no' outside top-%d logprobs "
            "(filled with %.1f). Increase --top_logprobs if this is large.",
            missing, n, cfg.top_logprobs, cfg.missing_fallback_logprob,
        )

    with open(cfg.rerank_output, "w") as f:
        for qid, hits in by_qid.items():
            hits.sort(key=lambda x: x[1], reverse=True)
            for pid, score in hits:
                f.write(f"{qid}\t{pid}\t{score}\n")
    logger.info("Final TREC ranklist written to %s", cfg.rerank_output)
