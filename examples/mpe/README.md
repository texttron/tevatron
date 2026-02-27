# Multi-Prefix Embedding — Reproduction Scripts

Scripts to reproduce all results from the paper. Five methods are compared:

| Config | Training | Eval chunking |
|--------|----------|---------------|
| Single-Vector | `--passage_chunk_size 0` | None |
| MaxP | `--passage_chunk_size 0` | Independent chunk 64 |
| MaxP-Train | `--passage_chunk_size 64 --passage_chunk_independent` | Independent chunk 64 |
| MPE Fixed-64 | `--passage_chunk_size 64` | Chunk 64 |
| MPE-Rand | `--passage_chunk_size_range 32,1024` | Chunk 64 |

Benchmarks: MLDR-EN, BrowseComp-Plus, and 4 LongEmbed datasets (NarrativeQA, 2WikiMQA, SummScreen, QMSum).

## Requirements

```bash
pip install transformers datasets peft faiss-cpu pyserini
pip install -e .  # install tevatron from repo root
```

Hardware: 8 GPUs recommended. Training uses `torchrun`; corpus encoding is sharded across GPUs in parallel.

## Quick Start

Run all steps end-to-end from the repo root:
```bash
cd examples/mpe
python 00_prepare_data.py
bash 01_train.sh
bash 02_eval_mldr_en.sh
bash 03_eval_browsecomp_plus.sh
bash 04_eval_longembed.sh
python 05_collect_results.py
```

Or with a custom output directory:
```bash
export EXP_ROOT=/path/to/experiment/dir
cd examples/mpe
python 00_prepare_data.py
bash 01_train.sh
# ...
```

## Step 0 — Prepare Data

Downloads and formats evaluation data from HuggingFace. Files are MD5-verified and skipped if already present.

```bash
python 00_prepare_data.py                           # prepare all benchmarks
python 00_prepare_data.py --benchmark mldr-en       # only MLDR-EN
python 00_prepare_data.py --benchmark browsecomp    # only BrowseComp-Plus
python 00_prepare_data.py --exp_root /path/to/root  # custom experiment root
```

LongEmbed data is prepared automatically by `04_eval_longembed.sh` via `prepare_longembed.py`.

**Outputs:**
- MLDR-EN: `data/queries.jsonl`, `data/corpus.jsonl`, `data/qrels.tsv`
- BrowseComp-Plus: `examples/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`

## Step 1 — Train

Trains 4 LoRA adapters on MLDR-EN (1 epoch each) using Qwen3-Embedding-0.6B as the base model:

```bash
bash 01_train.sh
```

| Model | Args |
|-------|------|
| `nochunk-epoch1` | `--passage_chunk_size 0` |
| `maxp-train-epoch1` | `--passage_chunk_size 64 --passage_chunk_independent` |
| `fixed-64-epoch1` | `--passage_chunk_size 64` |
| `prand-32to1024-epoch1` | `--passage_chunk_size_range 32,1024` |

Training is skipped if `adapter_config.json` already exists in the model directory. Checkpoints are saved to `models/`.

## Step 2 — Evaluate on MLDR-EN

Encodes queries and corpus, runs FAISS search, and evaluates with pyserini (nDCG@10, Recall@100):

```bash
bash 02_eval_mldr_en.sh
```

Corpus encoding is sharded across 8 GPUs in parallel. Each of the 5 configs is evaluated sequentially.

## Step 3 — Evaluate on BrowseComp-Plus

```bash
bash 03_eval_browsecomp_plus.sh
```

Evaluates against both evidence and gold qrels (nDCG@10, Recall@5,100,1000). Corpus is loaded from `Tevatron/browsecomp-plus-corpus` on HuggingFace.

## Step 4 — Evaluate on LongEmbed

```bash
bash 04_eval_longembed.sh
```

Runs all 5 configs across 4 LongEmbed datasets (20 evaluations total). Data is auto-downloaded and prepared on first run.

## Step 5 — Collect Results

Aggregates all TREC eval scores into a single table:

```bash
python 05_collect_results.py           # ASCII table
python 05_collect_results.py --latex   # also print LaTeX
```

## Configuration

All scripts default `EXP_ROOT` to the repo root (auto-detected from script location). Override via environment variable:
```bash
export EXP_ROOT=/my/experiment/dir
```

`NUM_GPUS` defaults to 8. Edit the variable at the top of each script to change it.

## Output Structure

```
{EXP_ROOT}/
├── models/                         # LoRA checkpoints
│   ├── nochunk-epoch1/
│   ├── maxp-train-epoch1/
│   ├── fixed-64-epoch1/
│   └── prand-32to1024-epoch1/
├── encode/                         # Pickled embeddings
│   ├── mldr-en/{config}/
│   ├── browsecomp-plus/{config}/
│   └── longembed/{dataset}/{config}/
├── results/                        # Rankings and TREC files
│   ├── mldr-en/{config}/
│   ├── browsecomp-plus/{config}/
│   └── longembed/{dataset}/{config}/
├── data/                           # Evaluation data
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   ├── qrels.tsv
│   └── longembed/{dataset}/
└── logs/repro/                     # Training and eval logs
```
