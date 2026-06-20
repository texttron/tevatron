# `tevatron.utils` — standalone helper CLIs

Off-to-the-side tooling that supports training and evaluation but isn't part of
either hot path. Everything here is a **standalone `python -m` driver** (kept out
of the trainers/drivers so those stay simple) or generic plumbing with no
`tevatron` deps. Grouped by purpose:

```
utils/
  prepare_queries.py        # eval data prep: materialize a BEIR queries.jsonl
  prepare_rerank_data.py     # eval data prep: ranklist + corpus -> rerank.jsonl
  annotate_with_teacher.py   # distillation: precompute teacher scores into a dataset
  merge_lora.py              # checkpoint post-proc: merge a LoRA adapter -> bf16 model dir
  bf16_save.py               # checkpoint post-proc: fp32 full-FT ckpt -> bf16-sharded copy
  format/                    # upstream Tevatron result-format converters (TREC / MS MARCO)
```

## Evaluation data prep

These two materialize the inputs the eval pipeline consumes; they have **no
`tevatron` imports** (`prepare_queries` reuses the BEIR loader from
`tevatron.eval.benchmarks`), which is why they live here rather than under
`tevatron.eval`. `prepare_rerank_data` is also imported by the retriever search
driver.

```bash
# BEIR queries.jsonl ({query_id, query}, test split, filtered by qrels)
python -m tevatron.utils.prepare_queries \
    --dataset nfcorpus \
    --output /path/to/eval_cache/e5_base/nfcorpus/queries.jsonl

# Join a first-stage ranklist with corpus text into a rerank.jsonl
python -m tevatron.utils.prepare_rerank_data \
    --dataset arguana \
    --rank_file   /path/to/eval_cache/e5_base/arguana/rank.text \
    --output_path /path/to/eval_cache/e5_base/arguana/rerank.jsonl \
    --depth 100
```

See [`../eval/README.md`](../eval/README.md) for how these feed `tevatron.eval.run`.

## Distillation: teacher annotation

`annotate_with_teacher` precomputes per-passage teacher scores **offline** into
the dataset, so distillation training adds no teacher forward pass at train time.
Runs the vLLM rerank backend (`tevatron.eval.backends.vllm`) over a materialized
rerank.jsonl. Standalone (no code imports it); consumed by
`tevatron.megatron.driver.distill_train`. Run in the vLLM env (`.venv-eval`).

```bash
python -m tevatron.utils.annotate_with_teacher \
    --teacher_model_name_or_path Qwen/Qwen3-Reranker-8B \
    --prompt_template qwen3_reranker \
    --rerank_max_len 2048 --max_model_len 2304 --tensor_parallel_size 8 \
    --source_dataset_name rlhn/rlhn-680K --max_negatives 15 \
    --output_dir /path/to/distill_cache/rlhn-680K-qwen3-reranker-8b
```

## Checkpoint post-processing

Run after training; both take a single checkpoint dir **or** a run dir (they
discover `checkpoint-*/step_*` subdirs). Kept out of the trainer so the training
loop stays simple.

```bash
# Merge a LoRA adapter into its base -> plain bf16 model dir (loads via normal
# HF/vLLM path, no --enable-lora plumbing). Megatron LoRA already saves merged.
python -m tevatron.utils.merge_lora /path/to/run_or_ckpt --dst /path/to/merged

# Re-save an fp32 full-FT checkpoint as a bf16-sharded copy (smaller, same weights)
python -m tevatron.utils.bf16_save /path/to/run_or_ckpt --dst /path/to/bf16
```

## `format/` — result-format converters (upstream)

The original Tevatron utilities for converting ranklists between formats. Plain
`argparse` scripts (run directly, not `python -m`):

- `convert_result_to_trec.py` — tevatron ranklist → 6-field TREC.
- `convert_result_to_marco.py` — tevatron ranklist → MS MARCO TSV.
- `prepare_rerank_input.py` — upstream rerank-input builder (the generic-HF
  counterpart to `prepare_rerank_data.py`, which is the BEIR-tuned version used
  by this project's eval pipeline).
