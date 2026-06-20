"""Score (and optionally rerank) a full benchmark sweep from ranklists in one call.

Benchmark-agnostic orchestrator: `--benchmark` selects a registry entry
(`tevatron.eval.benchmarks`, default `beir15`), which supplies the task list,
qrels resolution, and summary layout. Per dataset this script resolves a
ranklist (or reranks candidates via a scoring-server pool), scores it with the
invariant `metrics` core, caches the JSON, and renders the table with `summary`.

BEIR-15 = BEIR-13 + MS MARCO dev + CQADupstack (averaged over its 12
sub-forums), reported as the mean over 15 task scores — the unified headline
metric for both first-stage retrievers and rerankers.

The ranklist for each dataset is resolved from a pattern with a `{dataset}`
placeholder, so the same CLI serves:
  - first-stage baselines:  --ranklist_pattern /…/e5_base/{dataset}/rank.text
  - reranker outputs:       --ranklist_pattern /…/results/{ckpt}/{dataset}.rerank.text

Per-dataset metrics JSONs are written next to (or into) --results_dir so a
re-run skips already-scored datasets unless --overwrite is passed.

Two modes:

1. Score-only (default): --ranklist_pattern points at existing ranklists
   (first-stage rank.text or already-reranked .rerank.text) -> just score.

       python -m tevatron.eval.run \\
           --ranklist_pattern /path/to/eval_cache/e5_base/{dataset}/rank.text \\
           --results_dir /path/to/eval_cache/results/e5_base_only --name e5-base-v2

2. Rerank+score: --backends + --rerank_input_pattern reranks each dataset's
   rerank.jsonl through the persistent scoring server pool (one model load for
   the whole sweep), writes {dataset}.rerank.text into results_dir, then scores it.
   Backends may be local or remote (ip:port).

       python -m tevatron.eval.run \\
           --backends http://localhost:8101 ... http://localhost:8108 \\
           --rerank_input_pattern /path/to/eval_cache/e5_base/{dataset}/rerank.jsonl \\
           --results_dir /path/to/eval_cache/results/qwen3-8b-clora_e5 \\
           --name qwen3-8b-clora-e5
"""

import argparse
import json
import logging
import os

from tevatron.eval.benchmarks import get_benchmark
from tevatron.eval.metrics import DEFAULT_K_VALUES
from tevatron.eval.summary import DEFAULT_COLUMNS, _parse_column, render

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Score (and optionally rerank) a full benchmark sweep + aggregate.")
    ap.add_argument("--benchmark", default="beir15",
                    help="Benchmark registry entry (task list, qrels, summary layout). Default: beir15.")
    ap.add_argument("--ranklist_pattern", default=None,
                    help="Score-only mode: path with a literal {dataset} placeholder, e.g. "
                         "/…/e5_base/{dataset}/rank.text or /…/results/<ckpt>/{dataset}.rerank.text. "
                         "Omit when using --backends (the reranked .rerank.text is written "
                         "into --results_dir and scored from there).")
    # Rerank+score mode: drive a persistent scoring-server pool.
    ap.add_argument("--backends", nargs="+", default=None,
                    help="Rerank+score mode: scoring-server base URLs (local or ip:port). "
                         "Reranks each dataset's --rerank_input_pattern through the pool, "
                         "writes {dataset}.rerank.text into --results_dir, then scores it.")
    ap.add_argument("--rerank_input_pattern", default=None,
                    help="rerank.jsonl path with a {dataset} placeholder (rerank+score mode). "
                         "e.g. /path/to/eval_cache/e5_base/{dataset}/rerank.jsonl")
    ap.add_argument("--rerank_input_override", nargs="+", default=None, metavar="DATASET=PATH",
                    help="Per-dataset rerank.jsonl paths that don't fit --rerank_input_pattern.")
    ap.add_argument("--chunk_size", type=int, default=2000, help="Client chunk size (rerank+score mode).")
    ap.add_argument("--max_concurrency", type=int, default=None,
                    help="Max in-flight requests across the backend pool (default: #backends).")
    ap.add_argument("--query_prefix", default="query:")
    ap.add_argument("--passage_prefix", default="passage:")
    ap.add_argument("--results_dir", required=True,
                    help="Where per-dataset <dataset>.metrics.json are written/read.")
    ap.add_argument("--output", default=None,
                    help="Markdown summary path (default: <results_dir>/summary.md).")
    ap.add_argument("--name", default=None,
                    help="Display name for the summary header (default: results_dir basename).")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Subset of datasets to score (default: all of the benchmark's components).")
    ap.add_argument("--ranklist_override", nargs="+", default=None, metavar="DATASET=PATH",
                    help="Per-dataset ranklist paths that don't fit --ranklist_pattern, "
                         "e.g. 'msmarco=/…/bm25/run.msmarco-v1-passage.bm25-default.dev.txt'. "
                         "Repeatable.")
    ap.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS),
                    help="Summary metric columns as '<name>@<k>' (name in "
                         "ndcg/recall/map/mrr/precision). Default: ndcg@10 recall@200 ndcg@100.")
    ap.add_argument("--k_values", type=int, nargs="+", default=None,
                    help="Cutoff depths to compute per dataset. Default: union of the "
                         "k's implied by --columns and the standard 10/100/200/1000.")
    ap.add_argument("--beir_cache_dir", default=None,
                    help="Override the benchmark's qrels cache dir (BEIR: download root).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-score even if <dataset>.metrics.json already exists.")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    bench = get_benchmark(args.benchmark)

    rerank_mode = args.backends is not None
    if rerank_mode:
        if not args.rerank_input_pattern or "{dataset}" not in args.rerank_input_pattern:
            raise SystemExit("--backends requires --rerank_input_pattern with a {dataset} placeholder")
    else:
        if not args.ranklist_pattern or "{dataset}" not in args.ranklist_pattern:
            raise SystemExit("score-only mode requires --ranklist_pattern with a {dataset} placeholder")

    overrides: dict[str, str] = {}
    for spec in (args.ranklist_override or []):
        if "=" not in spec:
            raise SystemExit(f"--ranklist_override expects DATASET=PATH, got {spec!r}")
        d, path = spec.split("=", 1)
        overrides[d] = path

    rr_overrides: dict[str, str] = {}
    for spec in (args.rerank_input_override or []):
        if "=" not in spec:
            raise SystemExit(f"--rerank_input_override expects DATASET=PATH, got {spec!r}")
        d, path = spec.split("=", 1)
        rr_overrides[d] = path

    client = None
    if rerank_mode:
        from tevatron.eval.serve.client import RerankClient
        client = RerankClient(
            args.backends, chunk_size=args.chunk_size, max_concurrency=args.max_concurrency,
            query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
        )
        client.wait_ready()

    # Compute every cutoff the requested columns need, plus the standard set, so
    # the cached JSON always contains the table's metrics (and stays reusable).
    col_specs = [_parse_column(c) for c in args.columns]  # validates names early
    col_ks = {int(jm.split("@", 1)[1]) for _, jm, _ in col_specs}
    k_values = sorted(set(args.k_values) | col_ks) if args.k_values \
        else sorted(set(DEFAULT_K_VALUES) | col_ks)

    os.makedirs(args.results_dir, exist_ok=True)
    datasets = args.datasets or bench.datasets

    def _has_all_columns(m: dict) -> bool:
        return all(m.get(gk, {}).get(jm) is not None for gk, jm, _ in col_specs)

    metrics: dict[str, dict] = {}
    missing_ranklist: list[str] = []
    for d in datasets:
        out_json = os.path.join(args.results_dir, f"{d}.metrics.json")
        if os.path.exists(out_json) and not args.overwrite:
            with open(out_json) as f:
                cached = json.load(f)
            # Reuse only if it already carries every column the table needs;
            # otherwise fall through and re-score with the wider k_values.
            if _has_all_columns(cached):
                metrics[d] = cached
                logger.info("[cached] %s -> NDCG@10=%.4f", d, cached["ndcg"]["NDCG@10"])
                continue
            logger.info("[rescore] %s — cached JSON missing a requested column", d)

        if rerank_mode:
            # Rerank this dataset's candidates through the server pool, writing
            # the scored ranklist into results_dir. Skip the (expensive) rerank
            # if its .rerank.text already exists and we're not overwriting.
            rr_in = rr_overrides.get(d, args.rerank_input_pattern.format(dataset=d))
            ranklist = os.path.join(args.results_dir, f"{d}.rerank.text")
            if not os.path.exists(rr_in):
                logger.warning("[skip] %s — no rerank input at %s", d, rr_in)
                missing_ranklist.append(d)
                continue
            if os.path.exists(ranklist) and not args.overwrite:
                logger.info("[reuse] %s — rerank.text exists, scoring only", d)
            else:
                logger.info("[rerank] %s via %d backend(s)", d, len(args.backends))
                client.rerank_file(rr_in, ranklist)
        else:
            ranklist = overrides.get(d, args.ranklist_pattern.format(dataset=d))
            if not os.path.exists(ranklist):
                logger.warning("[skip] %s — no ranklist at %s", d, ranklist)
                missing_ranklist.append(d)
                continue

        m = bench.evaluate(d, ranklist, k_values=k_values, beir_cache_dir=args.beir_cache_dir)
        with open(out_json, "w") as f:
            json.dump(m, f, indent=2)
        metrics[d] = m
        logger.info("[scored] %s -> NDCG@10=%.4f", d, m["ndcg"]["NDCG@10"])

    if not metrics:
        raise SystemExit(f"No datasets scored (missing ranklists: {missing_ranklist})")

    name = args.name or os.path.basename(os.path.abspath(args.results_dir.rstrip("/")))
    summary = render(metrics, bench.summary_layout, name, columns=args.columns)

    output = args.output or os.path.join(args.results_dir, "summary.md")
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(output, "w") as f:
        f.write(summary)

    print(summary)
    if missing_ranklist:
        logger.warning("Missing ranklists for: %s", ", ".join(missing_ranklist))


if __name__ == "__main__":
    main()
