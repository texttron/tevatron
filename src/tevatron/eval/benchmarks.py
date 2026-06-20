"""Benchmark registry — *which* datasets, *where* their qrels come from, and
*how* their scores aggregate into a headline table.

A `Benchmark` bundles the three benchmark-specific facts that the invariant
`metrics` core and the generic `summary` renderer must not hardcode:

  - `datasets`     — the ordered task list,
  - `evaluate()`   — resolve qrels for one dataset and score a ranklist
                     (delegates the actual metric math to `metrics.score`),
  - `summary_layout` — the row/aggregate structure for `summary.render`.

Adding a new benchmark (MTEB, BRIGHT, a private set) = adding one entry to
`BENCHMARKS`; nothing in `metrics`/`run`/`summary`/`backends` changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List

from tevatron.eval.metrics import DEFAULT_K_VALUES, read_ranklist, score
from tevatron.eval.summary import Aggregate, SummaryLayout, SummaryRow

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# BEIR-15 task structure (was hardcoded across beir_eval + aggregate + eval_beir15)
# --------------------------------------------------------------------------- #
BEIR_13 = [
    "arguana", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa",
    "nfcorpus", "nq", "quora", "scidocs", "scifact", "trec-covid",
    "webis-touche2020",
]
CQADUP_SUBS = [
    "cqadupstack-android", "cqadupstack-english", "cqadupstack-gaming",
    "cqadupstack-gis", "cqadupstack-mathematica", "cqadupstack-physics",
    "cqadupstack-programmers", "cqadupstack-stats", "cqadupstack-tex",
    "cqadupstack-unix", "cqadupstack-webmasters", "cqadupstack-wordpress",
]
# msmarco sits between BEIR-13 and cqadupstack in the 15-task set.
BEIR_15 = BEIR_13 + ["msmarco"] + CQADUP_SUBS

DEFAULT_BEIR_CACHE = os.environ.get(
    "BEIR_CACHE_DIR", os.path.expanduser("~/.cache/tevatron/beir_qrels"))


@dataclass
class Benchmark:
    name: str                                  # display name, e.g. "BEIR-15"
    datasets: List[str]                        # ordered task list
    load_qrels: Callable[[str, str], dict]     # (dataset, cache_dir) -> qrels
    split_for: Callable[[str], str]            # dataset -> split name
    summary_layout: SummaryLayout              # row/aggregate structure
    default_cache_dir: str | None = None       # qrels cache/download root

    def evaluate(self, dataset: str, ranklist_path: str,
                 k_values=DEFAULT_K_VALUES, beir_cache_dir: str | None = None) -> dict:
        """Resolve qrels for `dataset`, score `ranklist_path`, return a metrics dict.

        The returned dict is `metrics.score(...)` plus a small provenance header
        (dataset/split/counts), matching the legacy `beir_eval.evaluate` schema so
        cached `*.metrics.json` files stay compatible.
        """
        cache_dir = beir_cache_dir or self.default_cache_dir
        qrels = self.load_qrels(dataset, cache_dir)
        results = read_ranklist(ranklist_path)
        m = score(qrels, results, k_values=k_values)
        return {
            "dataset": dataset,
            "split": self.split_for(dataset),
            "n_queries_qrels": len(qrels),
            "n_queries_ranked": len(results),
            **m,
        }


# --------------------------------------------------------------------------- #
# BEIR qrels loader (was beir_eval._beir_url_and_subdir + the GenericDataLoader
# block inside beir_eval.evaluate). Shared with utils/prepare_queries.
# --------------------------------------------------------------------------- #
def beir_url_and_subdir(dataset: str) -> tuple[str, str]:
    """Map a dataset name to (BEIR archive URL, on-disk data folder)."""
    base = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
    if dataset.startswith("cqadupstack-"):
        sub = dataset[len("cqadupstack-"):]
        return f"{base}/cqadupstack.zip", os.path.join("cqadupstack", sub)
    return f"{base}/{dataset}.zip", dataset


def beir_split_for(dataset: str) -> str:
    return "dev" if dataset == "msmarco" else "test"


def beir_data_path(dataset: str, beir_cache_dir: str) -> str:
    """Download/unzip the BEIR archive and return the dataset's data folder."""
    from beir import util

    os.makedirs(beir_cache_dir, exist_ok=True)
    url, subdir = beir_url_and_subdir(dataset)
    archive_root = util.download_and_unzip(url, beir_cache_dir)
    return os.path.join(os.path.dirname(archive_root), subdir) \
        if dataset.startswith("cqadupstack-") else archive_root


def _beir_load_qrels(dataset: str, beir_cache_dir: str | None) -> dict:
    from beir.datasets.data_loader import GenericDataLoader

    data_path = beir_data_path(dataset, beir_cache_dir or DEFAULT_BEIR_CACHE)
    split = beir_split_for(dataset)
    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return qrels


def _beir15_summary_layout() -> SummaryLayout:
    """BEIR-15 table: 13 plain BEIR rows, a collapsed cqadupstack row, an
    msmarco row, then BEIR-13 and BEIR-15 aggregates.

    BEIR-15 = BEIR-13 + msmarco-dev + cqadupstack (avg of 12) = 15 task units;
    cqadupstack contributes a single averaged unit (not 12), matching the
    MTEB-Retrieval English convention.
    """
    rows = [SummaryRow(label=d, members=[d]) for d in BEIR_13]
    rows.append(SummaryRow(label="cqadupstack (avg of {n})", members=CQADUP_SUBS,
                           show_if_absent=False))
    rows.append(SummaryRow(label="msmarco", members=["msmarco"], show_if_absent=False))
    aggregates = [
        Aggregate(label="BEIR-13 avg", units=[[d] for d in BEIR_13]),
        Aggregate(label="BEIR-15 avg (of {n})",
                  units=[[d] for d in BEIR_13] + [["msmarco"], list(CQADUP_SUBS)]),
    ]
    return SummaryLayout(rows=rows, aggregates=aggregates)


BENCHMARKS: dict[str, Benchmark] = {
    "beir15": Benchmark(
        name="BEIR-15",
        datasets=list(BEIR_15),
        load_qrels=_beir_load_qrels,
        split_for=beir_split_for,
        summary_layout=_beir15_summary_layout(),
        default_cache_dir=DEFAULT_BEIR_CACHE,
    ),
}


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]


__all__ = [
    "Benchmark", "BENCHMARKS", "get_benchmark",
    "BEIR_13", "CQADUP_SUBS", "BEIR_15", "DEFAULT_BEIR_CACHE",
    "beir_url_and_subdir", "beir_split_for", "beir_data_path",
]
