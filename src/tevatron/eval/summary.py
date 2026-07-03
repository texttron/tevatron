"""Generic markdown rendering of per-dataset metrics into a summary table.

This module is **benchmark-agnostic**: it knows how to parse `<name>@<k>`
columns, format a markdown table, average present values, and emit aggregate
rows — but *which* datasets are listed, which collapse into a single averaged
row (e.g. cqadupstack), and how the headline averages are composed are
benchmark facts described by a `SummaryLayout` (see `tevatron.eval.benchmarks`).

Usage (aggregate already-scored *.metrics.json into a table):
    python -m tevatron.eval.summary \\
        --results_dir /path/to/eval_cache/results/myckpt \\
        --output /path/to/eval_cache/results/myckpt/summary.md
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

# (metrics-block key, display prefix) — how a "<prefix>@<k>" column maps to
# the per-dataset metrics JSON written by metrics.score.
_METRIC_GROUPS = {
    "ndcg": "NDCG",
    "recall": "Recall",
    "map": "MAP",
    "mrr": "MRR",
    "precision": "P",
}
# Default columns: NDCG@10 (ranking quality) + Recall@200 (rerank recall
# ceiling at depth 200) + NDCG@100 (deeper ranking quality).
DEFAULT_COLUMNS = ("ndcg@10", "recall@200", "ndcg@100")


@dataclass
class SummaryRow:
    """One row above the aggregate rule.

    members of length 1 -> a plain per-dataset row; longer -> a *collapsed*
    row whose cells are the per-column average over present members (label may
    contain ``{n}``, filled with the count of present members).
    """
    label: str
    members: List[str]
    show_if_absent: bool = True  # render a dashed row when no member is present


@dataclass
class Aggregate:
    """One aggregate row below the rule.

    `units` is a list of task units; each unit is a list of datasets averaged
    into a single task value (length-1 unit = a plain dataset). The aggregate
    is the mean over units that have at least one present member. The label may
    contain ``{n}`` (filled with the number of contributing units).
    """
    label: str
    units: List[List[str]]


@dataclass
class SummaryLayout:
    rows: List[SummaryRow] = field(default_factory=list)
    aggregates: List[Aggregate] = field(default_factory=list)


def _parse_column(col: str) -> tuple[str, str, str]:
    """'ndcg@10' -> (group_key, json_metric_name, display_label)."""
    if "@" not in col:
        raise ValueError(f"metric column must be '<name>@<k>', got {col!r}")
    name, k = col.split("@", 1)
    name = name.strip().lower()
    if name not in _METRIC_GROUPS:
        raise ValueError(f"unknown metric {name!r}; choose from {sorted(_METRIC_GROUPS)}")
    prefix = _METRIC_GROUPS[name]
    return name, f"{prefix}@{k}", f"{prefix}@{k}"


def _get(metrics: dict, group_key: str, json_metric: str) -> float | None:
    return metrics.get(group_key, {}).get(json_metric)


def _avg(values: List[float]) -> float | None:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "-"


def _load_metrics(results_dir: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.metrics.json"))):
        name = os.path.basename(path).rsplit(".metrics.json", 1)[0]
        with open(path) as f:
            out[name] = json.load(f)
    return out


def render(results: Dict[str, dict], layout: SummaryLayout, name: str,
           columns=DEFAULT_COLUMNS) -> str:
    """Render `results` (dataset -> metrics dict) into a markdown summary."""
    cols = [_parse_column(c) for c in columns]  # [(group_key, json_metric, label), ...]
    header = "| Dataset | " + " | ".join(label for _, _, label in cols) + " |"
    rule = "|" + "---|" * (len(cols) + 1)
    lines = [f"# Eval — {name}", "", header, rule]

    def col_vals_for(member: str) -> list:
        m = results.get(member)
        return [None] * len(cols) if m is None else [_get(m, gk, jm) for gk, jm, _ in cols]

    def unit_value(unit: List[str], ci: int) -> float | None:
        # one task value = avg over present members of this unit, for column ci
        return _avg([col_vals_for(member)[ci] for member in unit])

    for row in layout.rows:
        present = [d for d in row.members if d in results]
        if not present and not row.show_if_absent:
            continue
        if len(row.members) == 1:
            cells = col_vals_for(row.members[0])
        else:
            cells = [_avg([col_vals_for(m)[ci] for m in present]) for ci in range(len(cols))]
        label = row.label.format(n=len(present)) if "{n}" in row.label else row.label
        lines.append(f"| {label} | " + " | ".join(_fmt(v) for v in cells) + " |")

    if layout.aggregates:
        lines.append(rule)
    for agg in layout.aggregates:
        contributing = [u for u in agg.units if any(d in results for d in u)]
        cells = []
        for ci in range(len(cols)):
            unit_vals = [unit_value(u, ci) for u in contributing]
            cells.append(_avg(unit_vals))
        label = agg.label.format(n=len(contributing)) if "{n}" in agg.label else agg.label
        lines.append(f"| **{label}** | " + " | ".join(_fmt(v) for v in cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Aggregate *.metrics.json into a markdown summary.")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--benchmark", default="beir15",
                    help="Benchmark whose summary layout to use (default: beir15).")
    ap.add_argument("--name", default=None,
                    help="display name; default = basename of results_dir")
    ap.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS),
                    help="Metric columns as '<name>@<k>' (name in "
                         "ndcg/recall/map/mrr/precision). Default: ndcg@10 recall@200 ndcg@100.")
    args = ap.parse_args()

    from tevatron.eval.benchmarks import get_benchmark
    layout = get_benchmark(args.benchmark).summary_layout

    name = args.name or os.path.basename(os.path.abspath(args.results_dir.rstrip("/")))
    metrics = _load_metrics(args.results_dir)
    if not metrics:
        raise SystemExit(f"No *.metrics.json found in {args.results_dir}")
    summary = render(metrics, layout, name, columns=args.columns)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
