"""Annotate a public reranker dataset with raw teacher logits.

Produces a HF dataset whose schema is

    {
        "query_id": str,
        "query": str,
        "passages": [{"title": str, "text": str}, ...],   # length P
        "scores":   [float, ...],                          # length P, parallel to passages
    }

`scores[i]` is `logit_yes - logit_no` from the teacher reranker (raw log-odds,
NOT a probability — the trainer applies its own temperature-scaled softmax).
`passages[0]` is always the source dataset's positive (downstream invariant
for sanity-checks; the listwise loss does not depend on order).

Pipeline (single-node, .venv-eval):

    1. Load source HF dataset with `query`, `positive_passages`, `negative_passages`.
    2. For each row, materialize `[positive, neg_1, ..., neg_K]` (sample / cap).
    3. Emit a flat rerank.jsonl (one (qid, docid, query, title, text) per pair)
       to a tempdir.
    4. Call `tevatron.eval.backends.vllm.run` over it with the chosen template
       (defaults to qwen3_reranker for Qwen3-Reranker-* teachers).
    5. Read the resulting `qid \\t pid \\t score` text, group by qid, write
       the HF dataset to `--output_dir`.

Example (Qwen3-Reranker-8B teacher on rlhn-680K):

    python -m tevatron.utils.annotate_with_teacher \\
        --teacher_model_name_or_path Qwen/Qwen3-Reranker-8B \\
        --prompt_template qwen3_reranker \\
        --rerank_max_len 2048 --max_model_len 2304 \\
        --tensor_parallel_size 8 \\
        --source_dataset_name rlhn/rlhn-680K \\
        --max_negatives 15 \\
        --output_dir /path/to/distill_cache/rlhn-680K-qwen3-reranker-8b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections import defaultdict
from typing import Iterable

from datasets import Dataset, load_dataset

from tevatron.eval.backends.vllm import VLLMRerankConfig, run as vllm_run

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate a public reranker dataset with teacher logits."
    )
    # Source dataset
    p.add_argument("--source_dataset_name", required=True,
                   help="HF dataset id, e.g. rlhn/rlhn-680K")
    p.add_argument("--source_dataset_config", default=None)
    p.add_argument("--source_dataset_path", default=None,
                   help="Optional local data_files override (json/parquet).")
    p.add_argument("--source_dataset_split", default="train")
    p.add_argument("--source_dataset_cache_dir", default=None)
    p.add_argument("--max_negatives", type=int, default=15,
                   help="Cap negatives per query. Group size at training time "
                        "= 1 + max_negatives.")
    p.add_argument("--shuffle_negatives", action="store_true",
                   help="Shuffle negatives before truncating to --max_negatives "
                        "(deterministic via --seed). Default keeps source order.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on rows for smoke tests.")

    # Teacher (vLLM)
    p.add_argument("--teacher_model_name_or_path", required=True)
    p.add_argument("--teacher_lora_name_or_path", default=None)
    p.add_argument("--teacher_tokenizer_name", default=None)
    p.add_argument("--prompt_template", choices=["tevatron", "qwen3_reranker"],
                   default="qwen3_reranker")
    p.add_argument("--rerank_max_len", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=2304)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_num_seqs", type=int, default=256)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--chunk_size", type=int, default=10000)
    p.add_argument("--top_logprobs", type=int, default=20)
    p.add_argument("--missing_fallback_logprob", type=float, default=-20.0)
    p.add_argument("--query_prefix", default="query:",
                   help="Only used for the tevatron template.")
    p.add_argument("--passage_prefix", default="passage:",
                   help="Only used for the tevatron template.")

    # Output
    p.add_argument("--output_dir", required=True,
                   help="Where to save the annotated HF dataset (save_to_disk).")
    p.add_argument("--scratch_dir", default=None,
                   help="Scratch dir for the intermediate rerank.jsonl / "
                        "rerank.text. Defaults to a tempdir under TMPDIR.")
    p.add_argument("--keep_scratch", action="store_true",
                   help="Don't delete the intermediate files. Useful for debugging.")
    return p


def _iter_source_rows(
    dataset_name: str,
    dataset_config: str | None,
    dataset_path: str | None,
    dataset_split: str,
    cache_dir: str | None,
    limit: int | None,
) -> Iterable[dict]:
    ds = load_dataset(
        dataset_name,
        dataset_config,
        data_files=dataset_path,
        split=dataset_split,
        cache_dir=cache_dir,
    )
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for i, row in enumerate(ds):
        yield i, row


def _materialize_candidates(
    row: dict,
    row_index: int,
    max_negatives: int,
    shuffle_negatives: bool,
    seed: int,
) -> tuple[str, str, list[dict]]:
    """Return (qid, query, [positive, neg_1, ..., neg_K]).

    Synthesizes a query_id when the source dataset doesn't carry one
    (rlhn-680K's rows have a query string but no stable id).
    """
    qid = str(row.get("query_id", row_index))
    query = row["query"]
    pos_list = list(row.get("positive_passages") or [])
    neg_list = list(row.get("negative_passages") or [])
    if not pos_list:
        return qid, query, []

    pos = pos_list[0]
    if shuffle_negatives and neg_list:
        import random
        rng = random.Random(seed + row_index)
        rng.shuffle(neg_list)
    if max_negatives is not None:
        neg_list = neg_list[:max_negatives]

    candidates = [pos] + neg_list
    # Normalize to (title, text). Some sources store as `text`/`title`,
    # others as just `text` — keep it permissive but explicit.
    norm = []
    for c in candidates:
        norm.append({
            "title": str(c.get("title", "") or ""),
            "text": str(c.get("text", "") or ""),
            "docid": str(c.get("docid", c.get("passage_id", ""))) or "",
        })
    return qid, query, norm


def _write_pairs_jsonl(
    rows_iter: Iterable[tuple[int, dict]],
    out_path: str,
    *,
    max_negatives: int,
    shuffle_negatives: bool,
    seed: int,
) -> tuple[dict[str, dict], int]:
    """Flatten (query, candidates) rows into a per-pair rerank.jsonl.

    Each emitted line carries:
        query_id  : stable per source row
        docid     : `{qid}#{i}` so we can recover ordering after rerank
        title/text: candidate body
        query     : query string (needed by the template)

    Returns:
        (groups, n_pairs) where groups[qid] = {
            "query": str,
            "passages": [{"title": ..., "text": ...}, ...]   # positive at index 0
        }
        n_pairs is total pair count emitted.
    """
    groups: dict[str, dict] = {}
    n_pairs = 0
    with open(out_path, "w") as f:
        for i, row in rows_iter:
            qid, query, cands = _materialize_candidates(
                row, i, max_negatives, shuffle_negatives, seed
            )
            if not cands:
                continue
            groups[qid] = {
                "query": query,
                "passages": [{"title": c["title"], "text": c["text"]} for c in cands],
            }
            for j, c in enumerate(cands):
                f.write(json.dumps({
                    "query_id": qid,
                    "docid": f"{qid}#{j}",
                    "query": query,
                    "title": c["title"],
                    "text": c["text"],
                }) + "\n")
                n_pairs += 1
    return groups, n_pairs


def _read_rerank_scores(path: str) -> dict[str, dict[str, float]]:
    """Load the TREC-style `qid \\t pid \\t score` file into nested dict."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            qid, pid, score = line.rstrip().split("\t")
            out[qid][pid] = float(score)
    return out


def _build_dataset(groups: dict[str, dict], scores: dict[str, dict[str, float]]) -> Dataset:
    rows = []
    n_missing = 0
    for qid, g in groups.items():
        passages = g["passages"]
        sc = scores.get(qid, {})
        ordered_scores = []
        for j in range(len(passages)):
            pid = f"{qid}#{j}"
            if pid not in sc:
                n_missing += 1
                ordered_scores.append(float("nan"))
            else:
                ordered_scores.append(sc[pid])
        rows.append({
            "query_id": qid,
            "query": g["query"],
            "passages": passages,
            "scores": ordered_scores,
        })
    if n_missing:
        logger.warning("%d (qid, pid) pairs had no score from the rerank backend "
                       "(filled with NaN). Investigate before training.", n_missing)
    return Dataset.from_list(rows)


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO,
    )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError(
            f"--output_dir {args.output_dir} exists and is non-empty. "
            "Refusing to overwrite."
        )

    scratch_ctx = (
        tempfile.TemporaryDirectory(prefix="distill_annotate_")
        if args.scratch_dir is None
        else None
    )
    scratch_dir = scratch_ctx.name if scratch_ctx is not None else args.scratch_dir
    os.makedirs(scratch_dir, exist_ok=True)
    rerank_input = os.path.join(scratch_dir, "rerank.jsonl")
    rerank_output = os.path.join(scratch_dir, "rerank.text")

    try:
        logger.info("Loading source dataset %s ...", args.source_dataset_name)
        rows = _iter_source_rows(
            args.source_dataset_name,
            args.source_dataset_config,
            args.source_dataset_path,
            args.source_dataset_split,
            args.source_dataset_cache_dir,
            args.limit,
        )
        groups, n_pairs = _write_pairs_jsonl(
            rows,
            rerank_input,
            max_negatives=args.max_negatives,
            shuffle_negatives=args.shuffle_negatives,
            seed=args.seed,
        )
        logger.info(
            "Wrote %d pairs across %d queries to %s",
            n_pairs, len(groups), rerank_input,
        )

        cfg = VLLMRerankConfig(
            model_name_or_path=args.teacher_model_name_or_path,
            rerank_input=rerank_input,
            rerank_output=rerank_output,
            lora_name_or_path=args.teacher_lora_name_or_path,
            tokenizer_name=args.teacher_tokenizer_name,
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
        vllm_run(cfg, query_prefix=args.query_prefix, passage_prefix=args.passage_prefix)

        logger.info("Reading scores from %s ...", rerank_output)
        scores = _read_rerank_scores(rerank_output)

        logger.info("Building HF dataset (%d queries) ...", len(groups))
        ds = _build_dataset(groups, scores)
        logger.info("Saving to %s ...", args.output_dir)
        ds.save_to_disk(args.output_dir)
        logger.info("Done. %d rows.", len(ds))
    finally:
        if scratch_ctx is not None and not args.keep_scratch:
            scratch_ctx.cleanup()


if __name__ == "__main__":
    main()
