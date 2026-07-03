"""Client for the rerank scoring server(s): chunk-dispatch + load-balance.

Reads a rerank.jsonl, splits it into fixed-size chunks of candidate records,
and dispatches chunks concurrently across one or more backend URLs (round-robin
+ a bounded worker pool). Results are reassembled per query and written as a
score-bearing TREC ranklist (qid\\tdocid\\tscore, sorted desc) — identical to
the output of the standalone rerank CLIs, so prepare/eval_beir15 consume it
unchanged.

The backends are stateless scorers (model loaded once, server-side), so adding
URLs scales throughput linearly with no client changes. The client is thin: it
holds no model, only shuffles records.

Usage (library):
    from tevatron.eval.serve.client import RerankClient
    client = RerankClient(["http://node-a:8100", "http://node-b:8100"])
    client.rerank_file("rerank.jsonl", "rank.text")

Usage (CLI):
    python -m tevatron.eval.serve.client \\
        --backends http://localhost:8100 http://localhost:8101 \\
        --rerank_input  .../scifact/rerank.jsonl \\
        --rerank_output .../results/myckpt/scifact.rerank.text
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx

logger = logging.getLogger(__name__)


def _load_candidates(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            items.append({
                "query_id": str(ex["query_id"]),
                "docid": str(ex["docid"]),
                "query": ex["query"],
                "text": ex.get("text", ""),
                "title": ex.get("title", ""),
            })
    return items


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield i, items[i:i + n]


class RerankClient:
    def __init__(
        self,
        backends: list[str],
        chunk_size: int = 2000,
        max_concurrency: int | None = None,
        timeout: float = 600.0,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        max_retries: int = 4,
        retry_backoff: float = 15.0,
    ):
        if not backends:
            raise ValueError("need at least one backend URL")
        self.backends = [b.rstrip("/") for b in backends]
        self.chunk_size = chunk_size
        # Default: one in-flight request per backend (servers batch internally).
        self.max_concurrency = max_concurrency or len(self.backends)
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def wait_ready(self, attempts: int = 60, interval: float = 10.0) -> None:
        """Block until every backend answers /health (so a sweep doesn't race
        a still-loading server)."""
        import time
        for b in self.backends:
            ok = False
            for _ in range(attempts):
                try:
                    r = httpx.get(f"{b}/health", timeout=5.0)
                    if r.status_code == 200:
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(interval)
            if not ok:
                raise RuntimeError(f"backend {b} not ready after {attempts*interval:.0f}s")
            info = httpx.get(f"{b}/info", timeout=10.0).json()
            logger.info("backend %s: %s %s (template=%s)", b, info["backend"],
                        info["model_name_or_path"], info["prompt_template"])

    def _score_chunk(self, url: str, chunk: list[dict]) -> list[dict]:
        payload = {
            "candidates": chunk,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
        }
        # Retry transient failures (cold-start graph capture, a momentarily
        # saturated backend) with linear backoff instead of letting one timeout
        # abort the whole sweep. wait_ready() already gated on /health, so a
        # failure here is transient, not a dead server.
        import time
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                r = httpx.post(f"{url}/score", json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["scores"]
            except Exception as exc:  # httpx.ReadTimeout, ConnectError, 5xx, …
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    backoff = self.retry_backoff * (attempt + 1)
                    logger.warning("%s/score failed (%s), retry %d/%d in %.0fs",
                                   url, type(exc).__name__, attempt + 1,
                                   self.max_retries - 1, backoff)
                    time.sleep(backoff)
        raise RuntimeError(f"{url}/score failed after {self.max_retries} attempts") from last_exc

    def rerank_file(self, rerank_input: str, rerank_output: str) -> dict:
        items = _load_candidates(rerank_input)
        chunks = list(_chunks(items, self.chunk_size))
        logger.info("%s: %d candidates -> %d chunks across %d backend(s)",
                    os.path.basename(rerank_input), len(items), len(chunks), len(self.backends))

        by_qid: dict[str, list[tuple[str, float]]] = {}
        n_missing = 0

        # One worker thread PER backend, each pulling chunks from a shared queue.
        # This guarantees at most one in-flight /score per backend (the vLLM
        # engine is not safe under concurrent generate()) AND load-balances
        # dynamically: a fast backend drains more chunks, a slow one fewer —
        # work is pulled, never pre-assigned. (The old round-robin pre-assigned
        # URLs at submit time, so a slow backend could get a second concurrent
        # request piled on it, hanging its non-thread-safe engine.)
        import threading
        from queue import Empty, Queue
        work: "Queue[list[dict]]" = Queue()
        for _, chunk in chunks:
            work.put(chunk)

        results: list[list[dict]] = []
        results_lock = threading.Lock()

        def _worker(url: str):
            while True:
                try:
                    chunk = work.get_nowait()
                except Empty:
                    return
                scores = self._score_chunk(url, chunk)
                with results_lock:
                    results.append(scores)

        n_workers = min(self.max_concurrency, len(self.backends))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker, self.backends[i]) for i in range(n_workers)]
            for fut in futs:
                fut.result()  # re-raise any worker exception

        for scores in results:
            for s in scores:
                if s.get("missing"):
                    n_missing += 1
                by_qid.setdefault(s["query_id"], []).append((s["docid"], float(s["score"])))

        os.makedirs(os.path.dirname(os.path.abspath(rerank_output)) or ".", exist_ok=True)
        n_lines = 0
        with open(rerank_output, "w") as f:
            for qid, hits in by_qid.items():
                hits.sort(key=lambda x: x[1], reverse=True)
                for pid, score in hits:
                    f.write(f"{qid}\t{pid}\t{score}\n")
                    n_lines += 1
        if n_missing:
            logger.warning("%d candidates had ' yes'/' no' outside top-K (sentinel-filled)", n_missing)
        logger.info("wrote %d lines (%d queries) -> %s", n_lines, len(by_qid), rerank_output)
        return {"n_lines": n_lines, "n_queries": len(by_qid), "n_missing": n_missing}


def main():
    ap = argparse.ArgumentParser(description="Rerank via persistent scoring server(s).")
    ap.add_argument("--backends", nargs="+", required=True, help="backend base URLs")
    ap.add_argument("--rerank_input", required=True)
    ap.add_argument("--rerank_output", required=True)
    ap.add_argument("--chunk_size", type=int, default=2000)
    ap.add_argument("--max_concurrency", type=int, default=None)
    ap.add_argument("--query_prefix", default="query:")
    ap.add_argument("--passage_prefix", default="passage:")
    ap.add_argument("--no_wait", action="store_true", help="skip /health readiness wait")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    client = RerankClient(
        args.backends, chunk_size=args.chunk_size, max_concurrency=args.max_concurrency,
        query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
    )
    if not args.no_wait:
        client.wait_ready()
    client.rerank_file(args.rerank_input, args.rerank_output)


if __name__ == "__main__":
    main()
