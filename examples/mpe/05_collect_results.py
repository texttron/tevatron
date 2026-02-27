#!/usr/bin/env python3
"""
Collect all evaluation results into a single paper-ready table.

Scans .trec files produced by 02/03/04 eval scripts, re-runs
pyserini trec_eval, and prints the unified table matching the paper format.

Usage:
    python 05_collect_results.py
    python 05_collect_results.py --latex
"""

import argparse
import os
import subprocess
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.environ.get("EXP_ROOT", os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..")))
RESULTS_ROOT = os.path.join(EXP_ROOT, "results")

# Qrels
MLDR_QRELS = os.path.join(EXP_ROOT, "data", "qrels.tsv")
BC_QRELS_EVIDENCE = os.path.join(
    EXP_ROOT, "examples", "BrowseComp-Plus", "topics-qrels", "qrel_evidence.txt"
)
LONGEMBED_QRELS = lambda ds: os.path.join(EXP_ROOT, "data", "longembed", ds, "qrels.tsv")

# Trec file paths
MLDR_TREC = lambda cfg: os.path.join(RESULTS_ROOT, "mldr-en", cfg, "ranking.trec")
BC_TREC = lambda cfg: os.path.join(RESULTS_ROOT, "browsecomp-plus", cfg, "ranking.trec")
LE_TREC = lambda cfg, ds: os.path.join(RESULTS_ROOT, "longembed", ds, cfg, "ranking.trec")

# 5 computed configs: (dir_key, display_name, trained_on_mldr)
CONFIGS = [
    ("single-vector",       "Single-Vector",       "✓"),
    ("maxp",                "MaxP",                "✓"),
    ("maxp-train",          "MaxP-Train",          "✓"),
    ("mpe-fixed64",         "MPE Fixed-64",        "✓"),
    ("mpe-rand-32to1024",   "MPE-Rand-32to1024",   "✓"),
]

LONGEMBED_DATASETS = ["narrativeqa", "2wikimqa", "summ_screen_fd", "qmsum"]
LONGEMBED_DISPLAY  = ["NarrativeQA", "2WikiMQA", "SummScreen", "QMSum"]

COLUMNS = ["Method", "MLDR-en", "BrowseComp-Plus",
           "NarrativeQA", "2WikiMQA", "SummScreen", "QMSum"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_trec_eval(qrels: str, trec_file: str, flags: list[str]) -> dict[str, float]:
    if not os.path.isfile(qrels) or not os.path.isfile(trec_file):
        return {}
    cmd = [sys.executable, "-m", "pyserini.eval.trec_eval", *flags, qrels, trec_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {}
    except Exception:
        return {}
    metrics = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "all":
            try:
                metrics[parts[0].strip()] = float(parts[2])
            except ValueError:
                pass
    return metrics


def get_ndcg10(qrels, trec):
    m = run_trec_eval(qrels, trec, ["-c", "-m", "ndcg_cut.10"])
    return m.get("ndcg_cut_10")


def fmt(val):
    if val is None:
        return "–"
    return f"{val:.3f}"


# ── Collect ───────────────────────────────────────────────────────────────────
def collect():
    rows = []
    for cfg_key, display, _ in CONFIGS:
        mldr = get_ndcg10(MLDR_QRELS, MLDR_TREC(cfg_key))
        bc = get_ndcg10(BC_QRELS_EVIDENCE, BC_TREC(cfg_key))
        le = {}
        for ds in LONGEMBED_DATASETS:
            le[ds] = get_ndcg10(LONGEMBED_QRELS(ds), LE_TREC(cfg_key, ds))
        rows.append((
            display, mldr, bc,
            le["narrativeqa"], le["2wikimqa"], le["summ_screen_fd"], le["qmsum"],
        ))
    return rows


# ── Print ─────────────────────────────────────────────────────────────────────
def print_ascii_table(rows):
    # Build string rows
    str_rows = []
    for name, mldr, bc, narr, wiki, summ, qm in rows:
        str_rows.append([name, fmt(mldr), fmt(bc),
                         fmt(narr), fmt(wiki), fmt(summ), fmt(qm)])

    # Column widths
    widths = [len(c) for c in COLUMNS]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    widths = [w + 2 for w in widths]

    def row_str(cells):
        return "  ".join(cells[i].ljust(widths[i]) for i in range(len(COLUMNS)))

    # Header
    print()
    print(row_str(COLUMNS))
    print("─" * sum(widths + [2 * (len(COLUMNS) - 1)]))

    for r in str_rows:
        print(row_str(r))

    print()


def print_latex_table(rows):
    print()
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Retrieval results (NDCG@10) across benchmarks.}")
    print("\\label{tab:main-results}")
    print("\\begin{tabular}{l" + "c" * 6 + "}")
    print("\\toprule")
    print(" & ".join(COLUMNS) + " \\\\")
    print("\\midrule")

    for name, mldr, bc, narr, wiki, summ, qm in rows:
        cells = [name, fmt(mldr), fmt(bc),
                 fmt(narr), fmt(wiki), fmt(summ), fmt(qm)]
        print(" & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    args = parser.parse_args()

    rows = collect()

    print("=" * 80)
    print("  Paper Results — NDCG@10 across all benchmarks")
    print("=" * 80)
    print_ascii_table(rows)

    if args.latex:
        print_latex_table(rows)

    print("Results dir:", RESULTS_ROOT)


if __name__ == "__main__":
    main()
