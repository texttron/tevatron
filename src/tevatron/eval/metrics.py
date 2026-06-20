"""Invariant retrieval metric computation.

Read a ranklist, score it against qrels with pytrec_eval (via BEIR's
`EvaluateRetrieval`): NDCG / Recall / MAP / MRR / Precision at arbitrary
cutoffs. This module has **no benchmark or model knowledge** — given a `qrels`
dict and a `results` dict it scores, nothing more.

*Which* datasets a suite contains, *where* its qrels come from, and the split
convention are benchmark facts and live in `tevatron.eval.benchmarks`. The
single-dataset CLI here resolves qrels by deferring to that registry, so the old
`python -m tevatron.eval.beir_eval --dataset scifact --ranklist ...` workflow
still works as `python -m tevatron.eval.metrics ...`.

Usage:
    python -m tevatron.eval.metrics \\
        --dataset scifact \\
        --ranklist /path/to/scifact.rerank.text \\
        --output /path/to/scifact.metrics.json
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (10, 100, 200, 1000)


def read_ranklist(path: str) -> Dict[str, Dict[str, float]]:
    """Parse a ranklist text file (3-field `qid pid score` or 6-field TREC)."""
    res: Dict[str, Dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) == 3:
                qid, pid, score = parts
            elif len(parts) == 6:
                qid, _, pid, _, score, _ = parts
            else:
                raise ValueError(f"unexpected ranklist line ({len(parts)} fields): {line!r}")
            res[qid][pid] = float(score)
    return res


def score(qrels: dict, results: dict, k_values=DEFAULT_K_VALUES) -> dict:
    """Score `results` against `qrels`, returning a metrics-block dict.

    Invariant over benchmark/model: the returned dict has keys
    ``ndcg``/``map``/``mrr``/``recall``/``precision``, each a
    ``{"<NAME>@<k>": value}`` mapping (BEIR's convention).
    """
    from beir.retrieval.evaluation import EvaluateRetrieval

    k_values = sorted({int(k) for k in k_values})
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels=qrels, results=results, k_values=k_values
    )
    # MRR is not in evaluate()'s default outputs; pytrec_eval's recip_rank
    # is exposed via BEIR's evaluate_custom hook.
    mrr = evaluator.evaluate_custom(
        qrels=qrels, results=results, k_values=k_values, metric="mrr"
    )
    return {
        "ndcg": ndcg,
        "map": _map,
        "mrr": mrr,
        "recall": recall,
        "precision": precision,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Score one ranklist against a benchmark's qrels (invariant metrics).")
    ap.add_argument("--dataset", required=True,
                    help="Dataset name within --benchmark (e.g. scifact, msmarco, cqadupstack-android)")
    ap.add_argument("--ranklist", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--benchmark", default="beir15",
                    help="Benchmark registry entry that resolves qrels (default: beir15).")
    ap.add_argument("--beir_cache_dir", default=None,
                    help="Override the benchmark's qrels cache dir (BEIR: download root).")
    ap.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES),
                    help="Cutoff depths for NDCG/Recall/MAP/MRR/Precision (default: 10 100 200 1000).")
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    # Defer qrels resolution to the benchmark registry (kept out of module import
    # to keep this core free of benchmark knowledge).
    from tevatron.eval.benchmarks import get_benchmark
    bench = get_benchmark(args.benchmark)
    metrics = bench.evaluate(args.dataset, args.ranklist, k_values=args.k_values,
                             beir_cache_dir=args.beir_cache_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("dataset", "ndcg", "recall")}, indent=2))


if __name__ == "__main__":
    main()
