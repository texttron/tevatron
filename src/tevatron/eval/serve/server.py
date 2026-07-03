"""Persistent rerank scoring server (vLLM or HF backend) behind one HTTP API.

Load the model ONCE, then score many batches over its lifetime — amortizing the
load + (for vLLM) graph-capture cost across an entire BEIR-15 sweep instead of
paying it per dataset. Both backends expose the identical `POST /score` contract
(see protocol.py): a batch of candidates in, yes/no logit-diff scores out.

Launch on a (possibly remote) GPU node:

    # vLLM backend, TP=2 on 2 GPUs, port 8100
    .venv-eval/bin/python -m tevatron.eval.serve.server \\
        --backend vllm --model /path/to/ckpt --tensor_parallel_size 2 --port 8100

    # HF backend, single GPU, port 8101
    .venv/bin/python -m tevatron.eval.serve.server \\
        --backend hf --model /path/to/ckpt --port 8101

To bring up several backends on one node, give each a disjoint GPU set via
CUDA_VISIBLE_DEVICES and a distinct --port; the client load-balances across them.

    CUDA_VISIBLE_DEVICES=0,1 ... --backend vllm --tensor_parallel_size 2 --port 8100 &
    CUDA_VISIBLE_DEVICES=2,3 ... --backend vllm --tensor_parallel_size 2 --port 8101 &
"""

from __future__ import annotations

import argparse
import logging

from tevatron.eval.serve.protocol import (
    InfoResponse,
    ScoreItem,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backends: each loads a model once and exposes score(candidates, prefixes).
# --------------------------------------------------------------------------- #
class _BaseBackend:
    name: str
    model_name_or_path: str
    prompt_template_name: str
    yes_id: int
    no_id: int

    def info(self) -> InfoResponse:  # pragma: no cover - trivial
        raise NotImplementedError

    def score(self, req: ScoreRequest) -> list[ScoreItem]:
        raise NotImplementedError


class VLLMBackend(_BaseBackend):
    name = "vllm"

    def __init__(self, args):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from tevatron.eval.backends.templates import get_template

        self.model_name_or_path = args.model
        self.prompt_template_name = args.prompt_template
        self.rerank_max_len = args.rerank_max_len
        self.tensor_parallel_size = args.tensor_parallel_size
        self.top_logprobs = args.top_logprobs
        self.missing_fallback_logprob = args.missing_fallback_logprob

        tok_name = args.tokenizer or args.model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self.template = get_template(args.prompt_template)
        self.yes_id, self.no_id = self.template.resolve_yes_no(self.tokenizer)
        logger.info("vLLM backend template=%s yes_id=%d no_id=%d",
                    self.template.name, self.yes_id, self.no_id)

        self.llm = LLM(
            model=args.model,
            tokenizer=tok_name,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            trust_remote_code=args.trust_remote_code,
            max_model_len=args.max_model_len,
            max_logprobs=args.top_logprobs,
        )
        self.sampling = SamplingParams(max_tokens=1, temperature=1.0, logprobs=args.top_logprobs)

    def info(self) -> InfoResponse:
        return InfoResponse(
            backend=self.name, model_name_or_path=self.model_name_or_path,
            prompt_template=self.prompt_template_name, yes_id=self.yes_id, no_id=self.no_id,
            tensor_parallel_size=self.tensor_parallel_size,
        )

    def score(self, req: ScoreRequest) -> list[ScoreItem]:
        from vllm.inputs import TokensPrompt

        cands = req.candidates
        build_kwargs = {}
        if self.template.name == "tevatron":
            build_kwargs = dict(query_prefix=req.query_prefix, passage_prefix=req.passage_prefix)
        ids_batch = self.template.build_token_ids_batch(
            self.tokenizer,
            [c.query for c in cands],
            [c.text for c in cands],
            [c.title for c in cands],
            self.rerank_max_len,
            **build_kwargs,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in ids_batch]
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling, use_tqdm=False)

        out: list[ScoreItem] = []
        for c, o in zip(cands, outputs):
            lp = o.outputs[0].logprobs[0]
            lp_yes = lp.get(self.yes_id)
            lp_no = lp.get(self.no_id)
            missing = lp_yes is None or lp_no is None
            ly = lp_yes.logprob if lp_yes is not None else self.missing_fallback_logprob
            ln = lp_no.logprob if lp_no is not None else self.missing_fallback_logprob
            out.append(ScoreItem(query_id=c.query_id, docid=c.docid, score=float(ly - ln), missing=missing))
        return out


class HFBackend(_BaseBackend):
    name = "hf"

    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.model_name_or_path = args.model
        self.prompt_template_name = args.prompt_template
        self.rerank_max_len = args.rerank_max_len
        self.batch_size = args.hf_batch_size
        # Two scoring contracts (see docs/bugs-and-fixes.md):
        #   yesno  — AutoModelForCausalLM; score = logit(' yes') - logit(' no')
        #            at the last token. Prompt ends in the yes/no suffix; no EOS.
        #            Matches the Megatron causal-LM reranker.
        #   seqcls — AutoModelForSequenceClassification(num_labels=1); score is
        #            the single regression logit at the last (EOS) token. Prompt
        #            is format_pair(...) with NO suffix + appended EOS. Matches
        #            the HF tevatron.reranker training path.
        self.score_mode = args.score_mode

        tok_name = args.tokenizer or args.model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        # Default: load straight onto the (single, CUDA_VISIBLE_DEVICES-pinned)
        # GPU with .to("cuda") — no accelerate / device_map, so no psutil dep.
        # --device_map auto is an opt-in escape hatch to shard one big model
        # across several visible GPUs (requires accelerate + psutil).
        from_kwargs = dict(torch_dtype=dtype, attn_implementation=args.attn_implementation)
        if args.device_map:
            from_kwargs["device_map"] = args.device_map

        if self.score_mode == "seqcls":
            from transformers import AutoModelForSequenceClassification
            self.template = None
            self.yes_id = self.no_id = -1  # unused
            self.model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=1, **from_kwargs)
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if args.lora_name_or_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, args.lora_name_or_path)
            logger.info("HF backend score_mode=seqcls (num_labels=1, score=logits[:,0] @ EOS)")
        else:
            from transformers import AutoModelForCausalLM
            from tevatron.eval.backends.templates import get_template
            self.template = get_template(args.prompt_template)
            self.yes_id, self.no_id = self.template.resolve_yes_no(self.tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(args.model, **from_kwargs)
            if args.lora_name_or_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, args.lora_name_or_path)
            logger.info("HF backend score_mode=yesno template=%s yes_id=%d no_id=%d",
                        self.template.name, self.yes_id, self.no_id)

        if not args.device_map:
            self.model = self.model.to("cuda")
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def info(self) -> InfoResponse:
        return InfoResponse(
            backend=self.name, model_name_or_path=self.model_name_or_path,
            prompt_template=(self.prompt_template_name if self.score_mode == "yesno" else "seqcls"),
            yes_id=self.yes_id, no_id=self.no_id,
        )

    def _build_ids(self, cands, req) -> list[list[int]]:
        if self.score_mode == "seqcls":
            from tevatron.eval.backends.templates import build_seqcls_token_ids_batch
            return build_seqcls_token_ids_batch(
                self.tokenizer,
                [c.query for c in cands],
                [c.text for c in cands],
                [c.title for c in cands],
                self.rerank_max_len,
                query_prefix=req.query_prefix,
                passage_prefix=req.passage_prefix,
                append_eos=True,
            )
        build_kwargs = {}
        if self.template.name == "tevatron":
            build_kwargs = dict(query_prefix=req.query_prefix, passage_prefix=req.passage_prefix)
        return self.template.build_token_ids_batch(
            self.tokenizer,
            [c.query for c in cands],
            [c.text for c in cands],
            [c.title for c in cands],
            self.rerank_max_len,
            **build_kwargs,
        )

    def score(self, req: ScoreRequest) -> list[ScoreItem]:
        torch = self.torch
        cands = req.candidates
        ids_batch = self._build_ids(cands, req)

        out: list[ScoreItem] = []
        with torch.no_grad():
            for s in range(0, len(ids_batch), self.batch_size):
                chunk = ids_batch[s:s + self.batch_size]
                padded = self.tokenizer.pad(
                    [{"input_ids": ids} for ids in chunk],
                    padding=True, return_attention_mask=True, return_tensors="pt",
                )
                input_ids = padded["input_ids"].to(self.device)
                attn = padded["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                if self.score_mode == "seqcls":
                    # Seq-cls head already pools to the last non-pad token and
                    # emits one logit per sequence: logits is (B, num_labels=1).
                    scores = logits[:, 0].float().cpu().tolist()
                else:
                    last_pos = attn.sum(dim=-1) - 1
                    b = input_ids.size(0)
                    last = logits[torch.arange(b, device=self.device), last_pos]
                    scores = (last[:, self.yes_id] - last[:, self.no_id]).float().cpu().tolist()
                for c, sc in zip(cands[s:s + self.batch_size], scores):
                    out.append(ScoreItem(query_id=c.query_id, docid=c.docid, score=float(sc), missing=False))
        return out


def build_app(backend: _BaseBackend):
    import threading

    from fastapi import FastAPI

    app = FastAPI(title="tevatron-rerank-server")

    # Starlette runs sync route handlers in a threadpool, so two concurrent
    # /score requests would otherwise call backend.score() — and thus the vLLM
    # LLM.generate() / HF forward — on the SAME model object from two threads.
    # vLLM's offline LLM engine is NOT thread-safe; concurrent generate() calls
    # corrupt engine state and hang. Serialize scoring per process: each backend
    # already batches a whole chunk in one call, so this costs no throughput
    # (the GPU was never going to run two batches at once anyway).
    score_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info():
        return backend.info()

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        with score_lock:
            return ScoreResponse(scores=backend.score(req))

    return app


def main():
    ap = argparse.ArgumentParser(description="Persistent rerank scoring server (vLLM or HF).")
    ap.add_argument("--backend", choices=["vllm", "hf"], required=True)
    ap.add_argument("--model", required=True, help="checkpoint dir / HF id")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--lora_name_or_path", default=None, help="HF backend: adapter to load on top of --model")
    ap.add_argument("--prompt_template", default="tevatron", choices=["tevatron", "qwen3_reranker"])
    ap.add_argument("--score_mode", default="yesno", choices=["yesno", "seqcls"],
                    help="HF backend scoring contract. 'yesno' (default): causal-LM, "
                         "score=logit(' yes')-logit(' no') at the last token (Megatron path). "
                         "'seqcls': AutoModelForSequenceClassification(num_labels=1), score is "
                         "the regression logit at the EOS token (HF tevatron.reranker path). "
                         "vLLM backend ignores this (always yesno).")
    ap.add_argument("--rerank_max_len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    # vLLM-specific
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--top_logprobs", type=int, default=20)
    ap.add_argument("--missing_fallback_logprob", type=float, default=-20.0)
    # HF-specific
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    ap.add_argument("--hf_batch_size", type=int, default=64)
    ap.add_argument("--device_map", default=None,
                    help="HF backend: opt-in device_map (e.g. 'auto') to shard one "
                         "model across visible GPUs. Default: single .to('cuda'), "
                         "no accelerate. Pin per-GPU via CUDA_VISIBLE_DEVICES instead.")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )

    backend: _BaseBackend = VLLMBackend(args) if args.backend == "vllm" else HFBackend(args)
    logger.info("%s backend ready: %s", backend.name, backend.model_name_or_path)

    import uvicorn
    uvicorn.run(build_app(backend), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
