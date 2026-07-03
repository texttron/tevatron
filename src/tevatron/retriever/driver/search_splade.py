"""SPLADE sparse retrieval for Tevatron-v2 — PySeismic (default) backend.

Consumes the sparse encodings produced by `encode_splade.py`:
  - corpus: one or more JSONL shards {"id", "content", "vector": {tok: w}}
  - query:  TSV, either JSON-dict (float) or repeated-token (int) form

and writes a score-bearing TREC ranklist (`qid\\tdocid\\tscore`, sorted desc
per query) — the same format `tevatron.retriever.driver.search` emits, so it
flows into `tevatron.utils.prepare_rerank_data` and `tevatron.eval.run`
unchanged.

Two retrieval backends are exposed via `--backend`:
  - seismic (default): PySeismic approximate sparse ANN (pure Python, no Java).
  - anserini: Lucene impact index (exact). Requires int-quantized encodings
    and a JVM; emitted as a stub that shells out to pyserini (TODO: wire up
    once the Anserini impact-index build path is needed).

Migrated from laconic-sparse-retrieval/src/retrieve_from_files_pyseismic.py.
Key fix vs the legacy script: when the index is built from in-memory vectors,
seismic returns the *positional* row index as corpus_id, so we map it back to
the real docid via the load order. The legacy `write_rank_file` wrote the raw
positional id and also discarded the score (rank only) — both corrected here.

Usage:
    python -m tevatron.retriever.driver.search_splade \\
        --corpus_files .../scifact/corpus/split*.jsonl \\
        --query_file   .../scifact/query/test.tsv \\
        --output_path  .../scifact/rank.text \\
        --depth 200
"""

import argparse
import json
import logging
import os
import time
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _load_corpus_shard(path: str):
    """Load one corpus JSONL shard -> (list[seismic_vector], list[docid])."""
    try:
        import orjson as _json
        loads = _json.loads
    except ImportError:
        loads = json.loads
    vectors, ids = [], []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = loads(line)
            except Exception:
                continue
            vectors.append([(k, float(v)) for k, v in doc["vector"].items()])
            ids.append(str(doc["id"]))
    return vectors, ids


def load_corpus(corpus_paths, num_workers=None):
    """Load (possibly sharded) corpus, preserving global doc order for id remap."""
    if num_workers is None:
        num_workers = min(len(corpus_paths), cpu_count() or 8)
    logger.info("Loading corpus from %d shard(s) with %d worker(s)...", len(corpus_paths), num_workers)
    t0 = time.time()
    all_vecs, all_ids = [], []
    if len(corpus_paths) == 1:
        all_vecs, all_ids = _load_corpus_shard(corpus_paths[0])
    else:
        # imap preserves input order -> global docid order is deterministic
        with Pool(processes=num_workers) as pool:
            for vecs, ids in pool.imap(_load_corpus_shard, corpus_paths):
                all_vecs.extend(vecs)
                all_ids.extend(ids)
    logger.info("Loaded %d docs in %.1fs", len(all_vecs), time.time() - t0)
    return all_vecs, all_ids


def load_queries(query_path):
    """Load query TSV (auto-detect JSON-dict float vs repeated-token int)."""
    if not os.path.exists(query_path):
        raise FileNotFoundError(query_path)
    vecs, qids = [], []
    with open(query_path) as f:
        first = f.readline().strip()
        fmt = "json" if (first.split("\t", 1)[1:] and first.split("\t", 1)[1].startswith("{")) else "repeated"
    logger.info("Query format: %s", fmt)
    with open(query_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            qid, payload = parts
            if fmt == "json":
                vec = [(k, float(v)) for k, v in json.loads(payload).items()]
            else:
                counts = {}
                for tok in payload.split():
                    counts[tok] = counts.get(tok, 0) + 1
                vec = [(k, float(v)) for k, v in counts.items()]
            vecs.append(vec)
            qids.append(qid)
    logger.info("Loaded %d queries", len(vecs))
    return vecs, qids


def search_seismic(corpus_vecs, corpus_ids, query_vecs, query_ids, depth,
                   index_cache=None, force_rebuild=False,
                   query_cut=10, heap_factor=0.7, merged_path=None):
    """Build a SeismicIndexLV (or load cache), batch-search, remap to docids.

    Two build modes (matching the legacy LACONIC retriever):
      - merged_path set: build natively from a single merged JSONL via
        ``SeismicIndexLV.build(path)`` — seismic parses the file itself and
        keys docs by their real ``id`` field, so no Python-side corpus load and
        no positional remap. Use this for large corpora (e.g. MS MARCO).
      - otherwise: load vectors in Python (``corpus_vecs``/``corpus_ids``) and
        ``build_from_dataset``; seismic returns positional ids, remapped here.
    """
    import numpy as np
    from seismic import SeismicDatasetLV, SeismicIndexLV, get_seismic_string

    st = get_seismic_string()
    index = None
    native_ids = merged_path is not None  # seismic keyed docs by real id, not pos
    # SeismicIndexLV is a Rust-backed object and is NOT picklable; use its native
    # save/load for caching. Caching is best-effort: a cache failure must never
    # kill the (expensive) search.
    if index_cache and os.path.exists(index_cache) and not force_rebuild:
        logger.info("Loading cached index from %s", index_cache)
        try:
            index = SeismicIndexLV.load(index_cache)
        except Exception as e:
            logger.warning("Failed to load cached index (%s); rebuilding.", e)
            index = None

    if index is None and merged_path is not None:
        logger.info("Building SeismicIndexLV natively from merged file %s...", merged_path)
        index = SeismicIndexLV.build(merged_path)
        if index_cache:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(index_cache)) or ".", exist_ok=True)
                index.save(index_cache)
                logger.info("Cached index to %s", index_cache)
            except Exception as e:
                logger.warning("Index caching failed (%s); continuing without cache.", e)

    if index is None:
        if not corpus_vecs:
            raise ValueError("No corpus vectors to index.")
        ds = SeismicDatasetLV()
        for i, vec in enumerate(corpus_vecs):
            d = dict(vec)
            ds.add_document(
                str(i),
                np.array(list(d.keys()), dtype=st),
                np.array(list(d.values()), dtype=np.float32),
            )
        logger.info("Building SeismicIndexLV over %d docs...", len(corpus_vecs))
        index = SeismicIndexLV.build_from_dataset(ds)
        if index_cache:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(index_cache)) or ".", exist_ok=True)
                index.save(index_cache)
                logger.info("Cached index to %s", index_cache)
            except Exception as e:
                logger.warning("Index caching failed (%s); continuing without cache.", e)

    n = len(query_vecs)
    q_components, q_values = [], []
    for vec in query_vecs:
        d = dict(vec)
        q_components.append(np.array(list(d.keys()), dtype=st))
        q_values.append(np.array(list(d.values()), dtype=np.float32))

    t0 = time.time()
    raw = index.batch_search(
        queries_ids=np.array(range(n), dtype=st),
        query_components=q_components,
        query_values=q_values,
        k=depth,
        query_cut=query_cut,
        heap_factor=heap_factor,
    )
    logger.info("Seismic search of %d queries in %.2fs", n, time.time() - t0)

    # batch_search may reorder; key by the integer query position we passed.
    raw = sorted(raw, key=lambda r: int(r[0][0]))
    # Each row: list of (query_pos, score, corpus_id). When built natively from
    # a merged file, corpus_id is already the real docid string; when built from
    # an in-memory dataset it's the positional index -> remap via corpus_ids.
    results = []
    for row in raw:
        if native_ids:
            hits = [(str(corpus_id), float(score)) for _, score, corpus_id in row]
        else:
            hits = [(corpus_ids[int(corpus_id)], float(score)) for _, score, corpus_id in row]
        results.append(hits)
    return results


def write_trec(results, query_ids, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    n = 0
    with open(output_path, "w") as f:
        for qid, hits in zip(query_ids, results):
            for docid, score in hits:
                f.write(f"{qid}\t{docid}\t{score}\n")
                n += 1
    logger.info("Wrote %d ranklist lines for %d queries to %s", n, len(query_ids), output_path)


def main():
    ap = argparse.ArgumentParser(description="SPLADE sparse retrieval (seismic default).")
    ap.add_argument("--corpus_files", nargs="+", required=True,
                    help="Corpus JSONL shard(s) from encode_splade.py.")
    ap.add_argument("--query_file", required=True, help="Query TSV from encode_splade.py.")
    ap.add_argument("--output_path", required=True, help="Output TREC ranklist (qid\\tdocid\\tscore).")
    ap.add_argument("--depth", type=int, default=200, help="Hits per query (default 200).")
    ap.add_argument("--backend", choices=["seismic", "anserini"], default="seismic")
    ap.add_argument("--index_cache", default=None, help="Path to save/load the seismic index (.pkl).")
    ap.add_argument("--force_rebuild_index", action="store_true")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--query_cut", type=int, default=10, help="Seismic query_cut.")
    ap.add_argument("--heap_factor", type=float, default=0.7, help="Seismic heap_factor.")
    args = ap.parse_args()

    if args.backend == "anserini":
        raise NotImplementedError(
            "Anserini impact-index backend not wired up yet. Encode with "
            "--splade_weight_format int and build a Lucene impact index via "
            "pyserini; see docs. Use --backend seismic for now."
        )

    # Native merged-file build: a single corpus file whose name contains
    # "merged" is handed straight to seismic (no Python-side corpus load), which
    # keys docs by their real `id` field. Big-corpus fast path (e.g. MS MARCO).
    merged_path = None
    corpus_vecs, corpus_ids = None, None
    if len(args.corpus_files) == 1 and "merged" in os.path.basename(args.corpus_files[0]):
        merged_path = args.corpus_files[0]
        logger.info("Detected merged corpus file; using native SeismicIndexLV.build path.")
    else:
        corpus_vecs, corpus_ids = load_corpus(args.corpus_files, num_workers=args.num_workers)

    query_vecs, query_ids = load_queries(args.query_file)
    results = search_seismic(
        corpus_vecs, corpus_ids, query_vecs, query_ids, args.depth,
        index_cache=args.index_cache, force_rebuild=args.force_rebuild_index,
        query_cut=args.query_cut, heap_factor=args.heap_factor, merged_path=merged_path,
    )
    write_trec(results, query_ids, args.output_path)


if __name__ == "__main__":
    main()
