"""vLLM reranker eval CLI for causal-LM yes/no reranker checkpoints.

Use for Megatron-trained checkpoints (or any causal-LM reranker that scores
via ' yes'/' no' logits at the prompt suffix). For HF SeqCls baselines, use
`tevatron.eval.rerank` instead.

vLLM manages workers via `tensor_parallel_size`; do NOT launch under torchrun.

Example (single 8xGPU node, TP=8):

    python -m tevatron.eval.rerank_vllm \\
        --model_name_or_path output/qwen3-0.6b-reranker/step_10136 \\
        --rerank_input  /path/to/eval_cache/e5_base/msmarco/rerank.jsonl \\
        --rerank_output /path/to/eval_cache/results/myckpt-vllm/msmarco.rerank.text \\
        --rerank_max_len 512 \\
        --tensor_parallel_size 8
"""

import argparse
import logging

from tevatron.eval.backends.vllm import VLLMRerankConfig, run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tevatron reranker eval (vLLM backend)")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--lora_name_or_path", default=None)
    p.add_argument("--tokenizer_name", default=None)
    p.add_argument("--rerank_input", required=True)
    p.add_argument("--rerank_output", required=True)
    p.add_argument("--rerank_max_len", type=int, default=512,
                   help="Max input length (prompt token ids truncated to this).")
    p.add_argument("--max_model_len", type=int, default=1024,
                   help="vLLM engine max_model_len. Should be >= rerank_max_len "
                        "with headroom for the +1 generated token and any BOS "
                        "the tokenizer adds.")
    p.add_argument("--query_prefix", default="query:")
    p.add_argument("--passage_prefix", default="passage:")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_num_seqs", type=int, default=256)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--chunk_size", type=int, default=10000,
                   help="Number of prompts per llm.generate() call. vLLM 0.19 "
                        "renders all prompts client-side before scheduling, so "
                        "very large requests inflate memory. Lower this if "
                        "EngineCore dies during 'Rendering prompts'.")
    p.add_argument("--top_logprobs", type=int, default=20,
                   help="Number of top logprobs to request. Both ' yes' and "
                        "' no' need to fall inside this top-K for a clean "
                        "score; 20 is plenty for trained rerankers.")
    p.add_argument("--missing_fallback_logprob", type=float, default=-20.0,
                   help="Fallback logprob value if ' yes' or ' no' isn't in "
                        "the returned top-K (rare for trained rerankers).")
    p.add_argument("--prompt_template", choices=["tevatron", "qwen3_reranker"],
                   default="tevatron",
                   help="Prompt template to use. 'tevatron' for our trained "
                        "checkpoints (' yes'/' no' after '?' suffix); "
                        "'qwen3_reranker' for Qwen/Qwen3-Reranker-* models "
                        "(chat template, 'yes'/'no' without leading space).")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        level=logging.INFO)
    cfg = VLLMRerankConfig(
        model_name_or_path=args.model_name_or_path,
        rerank_input=args.rerank_input,
        rerank_output=args.rerank_output,
        lora_name_or_path=args.lora_name_or_path,
        tokenizer_name=args.tokenizer_name,
        rerank_max_len=args.rerank_max_len,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        chunk_size=args.chunk_size,
        top_logprobs=args.top_logprobs,
        missing_fallback_logprob=args.missing_fallback_logprob,
        prompt_template=args.prompt_template,
    )
    run(cfg, query_prefix=args.query_prefix, passage_prefix=args.passage_prefix)


if __name__ == "__main__":
    main()
