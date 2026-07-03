"""Join a BEIR TREC ranklist with corpus content into a rerank JSONL.

Output schema (one line per (qid, pid) candidate):
    {"query_id": str, "query": str, "docid": str, "title": str, "text": str, "score": float}

Usage:
    python -m tevatron.utils.prepare_rerank_data \
        --dataset arguana \
        --rank_file /path/to/eval_cache/e5_base/arguana/rank.text \
        --output_path /path/to/eval_cache/e5_base/arguana/rerank.jsonl \
        --depth 100
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_trec(path: str) -> Dict[str, List[Tuple[str, float]]]:
    results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) == 3:
                qid, pid, score = parts
            elif len(parts) == 6:
                # TREC format: qid Q0 pid rank score run
                qid, _, pid, _, score, _ = parts
            else:
                raise ValueError(f"Unexpected ranklist line ({len(parts)} fields): {line!r}")
            results[qid].append((pid, float(score)))
    # ranklist files from tevatron search are already sorted by score desc per qid;
    # don't re-sort, but truncate later.
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="BEIR dataset config (e.g. arguana, msmarco, cqadupstack-android)")
    ap.add_argument("--rank_file", required=True, help="TREC ranklist from Phase 1")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--query_dataset", default="Tevatron/beir",
                    help="HF dataset name for queries (override for non-BEIR sets)")
    ap.add_argument("--query_jsonl", default=None,
                    help="Local JSONL with {query_id, query} (preferred for BEIR; "
                         "Tevatron/beir is a script-based HF dataset and breaks under datasets>=3).")
    ap.add_argument("--corpus_dataset", default="Tevatron/beir-corpus",
                    help="HF dataset name for corpus")
    ap.add_argument("--query_split", default="test")
    ap.add_argument("--corpus_split", default="train")
    ap.add_argument("--no_dataset_config", action="store_true",
                    help="Don't pass --dataset as the HF config name (use for flat datasets "
                         "like Tevatron/msmarco-passage-aug).")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    config = None if args.no_dataset_config else args.dataset

    qid_to_query: Dict[str, str] = {}
    if args.query_jsonl:
        logger.info("Loading queries from local JSONL %s", args.query_jsonl)
        with open(args.query_jsonl) as f:
            for line in f:
                ex = json.loads(line)
                qid_to_query[str(ex["query_id"])] = ex["query"]
    else:
        logger.info("Loading queries from %s config=%s split=%s",
                    args.query_dataset, config, args.query_split)
        queries = load_dataset(args.query_dataset, config, split=args.query_split, cache_dir=args.cache_dir)
        for ex in tqdm(queries, desc="queries"):
            qid_to_query[str(ex["query_id"])] = ex["query"]

    logger.info("Loading corpus from %s config=%s split=%s",
                args.corpus_dataset, config, args.corpus_split)
    corpus = load_dataset(args.corpus_dataset, config, split=args.corpus_split, cache_dir=args.cache_dir)
    docid_to_doc: Dict[str, dict] = {}
    for ex in tqdm(corpus, desc="corpus"):
        docid_to_doc[str(ex["docid"])] = ex

    logger.info("Reading ranklist %s", args.rank_file)
    rank = read_trec(args.rank_file)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    n_lines = 0
    n_missing_q = 0
    n_missing_d = 0
    with open(args.output_path, "w") as f:
        for qid, hits in tqdm(rank.items(), desc="join"):
            if qid not in qid_to_query:
                n_missing_q += 1
                continue
            query = qid_to_query[qid]
            for pid, score in hits[: args.depth]:
                doc = docid_to_doc.get(pid)
                if doc is None:
                    n_missing_d += 1
                    continue
                f.write(json.dumps({
                    "query_id": qid,
                    "query": query,
                    "docid": pid,
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": score,
                }) + "\n")
                n_lines += 1

    logger.info("Wrote %d lines to %s (missing q=%d, missing d=%d)",
                n_lines, args.output_path, n_missing_q, n_missing_d)


if __name__ == "__main__":
    main()
