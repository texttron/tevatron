"""CLI for reranking with a Megatron-trained checkpoint (yes/no log-odds).

Example:

    torchrun --nproc_per_node=8 -m tevatron.megatron.eval.rerank \\
        --model_name_or_path output/qwen3-0.6b-megatron-reranker/step_5000 \\
        --rerank_input  /path/to/eval_cache/e5_base/msmarco/rerank.jsonl \\
        --rerank_output /path/to/eval_cache/results/myckpt/msmarco.rerank.text \\
        --rerank_max_len 512 \\
        --per_device_eval_batch_size 16
"""

import argparse
import logging

from tevatron.megatron.eval.rerank_backend import MegatronRerankConfig, run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Megatron reranker eval (yes/no log-odds)")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--lora_name_or_path", default=None)
    p.add_argument("--tokenizer_name", default=None)
    p.add_argument("--rerank_input", required=True)
    p.add_argument("--rerank_output", required=True)
    p.add_argument("--rerank_max_len", type=int, default=512)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--query_prefix", default="query:")
    p.add_argument("--passage_prefix", default="passage:")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--prompt_template", choices=["tevatron", "qwen3_reranker"],
                   default="tevatron")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        level=logging.INFO)
    cfg = MegatronRerankConfig(
        model_name_or_path=args.model_name_or_path,
        lora_name_or_path=args.lora_name_or_path,
        tokenizer_name=args.tokenizer_name,
        rerank_input=args.rerank_input,
        rerank_output=args.rerank_output,
        rerank_max_len=args.rerank_max_len,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        prompt_template=args.prompt_template,
    )
    run(cfg)


if __name__ == "__main__":
    main()
