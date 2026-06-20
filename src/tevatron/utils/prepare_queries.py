"""Materialize BEIR queries.jsonl (test split, filtered by qrels) into a cache.

The HF `Tevatron/beir` dataset is a script-based loader and is incompatible
with `datasets>=3`. Instead, reuse the BEIR archive that we already download
for qrels and write a flat JSONL with the schema the encoder expects:

    {"query_id": "<id>", "query": "<text>"}

Usage:
    python -m tevatron.utils.prepare_queries \\
        --dataset nfcorpus \\
        --output /path/to/eval_cache/e5_base/nfcorpus/queries.jsonl
"""

import argparse
import json
import logging
import os

# BEIR archive resolution (URL/subdir, split, data path) is owned by the
# benchmark registry; reuse it here so there is one source of truth.
from tevatron.eval.benchmarks import DEFAULT_BEIR_CACHE, beir_data_path, beir_split_for

logger = logging.getLogger(__name__)


def materialize(dataset: str, output: str, beir_cache_dir: str = DEFAULT_BEIR_CACHE) -> int:
    data_path = beir_data_path(dataset, beir_cache_dir)

    split = beir_split_for(dataset)
    qrels_path = os.path.join(data_path, "qrels", f"{split}.tsv")
    queries_path = os.path.join(data_path, "queries.jsonl")

    keep_qids: set[str] = set()
    with open(qrels_path) as f:
        header = f.readline()  # qid \t pid \t score
        for line in f:
            qid = line.split("\t", 1)[0].strip()
            if qid:
                keep_qids.add(qid)

    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with open(queries_path) as fin, open(output, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            qid = str(ex["_id"])
            if qid not in keep_qids:
                continue
            fout.write(json.dumps({"query_id": qid, "query": ex["text"]}) + "\n")
            n += 1
    logger.info("wrote %d queries (split=%s) for %s -> %s", n, split, dataset, output)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="BEIR dataset name (e.g. nfcorpus, msmarco, cqadupstack-android)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--beir_cache_dir", default=DEFAULT_BEIR_CACHE)
    args = ap.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    materialize(args.dataset, args.output, args.beir_cache_dir)


if __name__ == "__main__":
    main()
