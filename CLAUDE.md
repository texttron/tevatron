# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tevatron is a dense neural retrieval toolkit for training and running billion-scale LLM retrievers. The primary code lives in `src/tevatron/retriever/`. The workflow has three stages: train a dual-encoder model, encode queries/passages to embeddings, then search with FAISS.

## Research Goal (current branch: `0127-debug-random-size`)

Comparing **plain last/eos pooling** (one embedding per passage) vs **chunked passage pooling** (multiple embeddings per passage, scored via MaxSim). The key question is how different chunking strategies at training time affect retrieval quality.

**Chunking strategies being compared:**

1. **No chunking** (`passage_chunk_size=0`): Standard single-embedding-per-passage with last/eos pooling.
2. **Fixed chunking** (`passage_chunk_size=N`): Every passage is split into chunks of exactly N tokens (plus EOS).
3. **Passage-level random chunking** (`passage_chunk_size_range="min,max"`, `passage_chunk_size_variable=False`): Each passage gets one random chunk size drawn from [min, max]; all chunks within that passage use the same size.
4. **Fully random chunking** (`passage_chunk_size_range="min,max"`, `passage_chunk_size_variable=True`): Each individual chunk within a passage gets its own random size from [min, max].

**How chunked training works:** Training passages are split into token chunks → chunks are joined with EOS tokens as separators → the full sequence is encoded by the model → `_pooling_chunked` extracts the hidden state at each EOS position to get one embedding per chunk → MaxSim computes query-passage similarity as the max dot product across chunks → contrastive loss is computed on the MaxSim scores.

## Experiment Plan

### Naming convention

Training configs use these names (used as directory names everywhere):

| Name | Training args |
|------|--------------|
| `nochunk` | `--passage_chunk_size 0` |
| `fixed-32` | `--passage_chunk_size 32` |
| `fixed-64` | `--passage_chunk_size 64` |
| `fixed-128` | `--passage_chunk_size 128` |
| `fixed-256` | `--passage_chunk_size 256` |
| `fixed-512` | `--passage_chunk_size 512` |
| `fixed-1024` | `--passage_chunk_size 1024` |
| `fixed-2048` | `--passage_chunk_size 2048` |
| `fixed-4096` | `--passage_chunk_size 4096` |
| `prand-32to64` | `--passage_chunk_size_range "32,64"` |
| `prand-32to128` | `--passage_chunk_size_range "32,128"` |
| `prand-32to256` | `--passage_chunk_size_range "32,256"` |
| `prand-32to512` | `--passage_chunk_size_range "32,512"` |
| `prand-32to1024` | `--passage_chunk_size_range "32,1024"` |
| `prand-32to2048` | `--passage_chunk_size_range "32,2048"` |
| `prand-32to4096` | `--passage_chunk_size_range "32,4096"` |

Retrieval configs (applied at encode time, independent of training config):

| Name | Encode args |
|------|------------|
| `ret-nochunk` | `--passage_chunk_size 0` |
| `ret-fixed-32` | `--passage_chunk_size 32` |
| `ret-fixed-64` | `--passage_chunk_size 64` |
| `ret-fixed-128` | `--passage_chunk_size 128` |
| `ret-fixed-256` | `--passage_chunk_size 256` |
| `ret-fixed-512` | `--passage_chunk_size 512` |
| `ret-fixed-1024` | `--passage_chunk_size 1024` |
| `ret-fixed-2048` | `--passage_chunk_size 2048` |
| `ret-fixed-4096` | `--passage_chunk_size 4096` |

Total: 16 training runs × 9 retrieval configs = 144 encode+search+eval runs.

### Directory structure

Use a single `exp/` root. This structure is the same on every machine — copy whichever subtrees you need.

```
exp/
├── models/                              # LoRA checkpoints from training
│   ├── nochunk/
│   ├── fixed-32/
│   ├── fixed-64/
│   ├── ...
│   ├── prand-32to64/
│   └── prand-32to4096/
├── encode/                              # Pickled embeddings
│   └── {train_name}/                    # e.g. fixed-256/
│       ├── queries.pkl                  # one per trained model (always no-chunk)
│       ├── corpus-ret-nochunk.pkl       # corpus encoded without chunking
│       ├── corpus-ret-fixed-32.pkl      # corpus encoded with chunk_size=32
│       ├── corpus-ret-fixed-64.pkl
│       └── ...
├── rank/                                # Raw ranking files
│   └── {train_name}/
│       ├── ret-nochunk.txt
│       ├── ret-fixed-32.txt
│       └── ...
├── trec/                                # TREC-formatted rankings
│   └── {train_name}/
│       ├── ret-nochunk.trec
│       ├── ret-fixed-32.trec
│       └── ...
└── results.tsv                          # Aggregated eval metrics
```

### passage_max_len consideration

`passage_max_len` caps the total token length of the chunked passage (all chunks + EOS tokens combined). With `passage_max_len=512` and `chunk_size=1024`, the entire passage fits in one chunk, which is effectively no chunking. To get meaningful multi-chunk behavior with large chunk sizes, `passage_max_len` must be larger than the chunk size. Choose a `passage_max_len` large enough to hold at least 2 chunks of your largest chunk size (e.g., `passage_max_len=8192` or higher for `chunk_size=4096`). This affects VRAM usage — larger passage_max_len requires more memory.

### Script generation

Two separate generators: one for training, one for retrieval. Edit the config block at the top of each, then run.

**Training** — `scripts/generate_training.sh`:
```bash
bash scripts/generate_training.sh
```
Produces 16 scripts in `scripts/training/` (e.g. `train_fixed-256.sh`). Each runs torchrun multi-GPU training with the correct chunk args. Skips if `adapter_config.json` already exists.

**Retrieval** — `scripts/generate_retrieval.sh`:
```bash
bash scripts/generate_retrieval.sh
```
Produces 16 scripts in `scripts/retrieval/` (e.g. `eval_mldr-en_fixed-256.sh`). Each encodes queries once, then for each retrieval chunk size (0, 32, 64, ..., 4096) plus pre-chunked: encodes corpus in parallel across GPUs, searches (with `--chunked` when chunk size > 0), converts to TREC, and runs pyserini eval.

Retrieval scripts accept an optional checkpoint path argument:
```bash
bash scripts/retrieval/eval_mldr-en_fixed-256.sh                    # uses default MLDR checkpoint
bash scripts/retrieval/eval_mldr-en_fixed-256.sh /path/to/checkpoint # uses custom checkpoint
```

To evaluate a different corpus, edit `CORPUS_PATH`, `PRECHUNKED_CORPUS_PATH`, `EVAL_NAME`, `QUERY_PREFIX`, and `QRELS` in `generate_retrieval.sh`, then re-run. Results are separated by `EVAL_NAME` in the output paths (`encode/{eval_name}/{train_name}/`, `results/{eval_name}/{train_name}/`).

Pre-chunked corpus evaluation is included automatically if `prechunked-corpus.jsonl` exists in the data directory. Expected format: `{"docid": "...", "chunks": ["chunk1 text", "chunk2 text", ...]}` per line.

To add/remove training or retrieval configs, edit the arrays in the respective generator:
```bash
# In generate_training.sh:
TRAIN_CONFIGS=(
  "nochunk|--passage_chunk_size 0"
  "fixed-256|--passage_chunk_size 256"
  "prand-32to256|--passage_chunk_size_range 32,256"
)

# In generate_retrieval.sh:
RET_CHUNKS=(0 32 64 128 256 512 1024 2048 4096)
```

### Cross-machine workflow

1. **Training machine**: `bash scripts/experiments/train_fixed-256.sh` — produces `models/fixed-256/` (LoRA adapter, small).
2. **Eval machine**: Clone repo, `pip install -e .`, copy `models/fixed-256/` from training machine. Run `bash scripts/experiments/eval_fixed-256.sh`.
3. **Only `models/` needs transfer** — LoRA adapters are small. The base model downloads from HuggingFace on each machine.
4. **Results**: Eval metrics print to stdout. Rankings and TREC files in `results/{train_name}/`.
5. **Data dir**: Must contain `queries.jsonl`, `corpus.jsonl`, `qrels.tsv` on the eval machine. Path set in `generate_experiments.sh`.

## Commands

### Install
```bash
pip install transformers datasets peft faiss-cpu
pip install -e .
```

### Train (single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.train \
  --output_dir <output> --model_name_or_path <model> --lora --do_train \
  --dataset_name <dataset> --bf16 --pooling last --normalize --temperature 0.01 \
  --per_device_train_batch_size 4 --train_group_size 16 \
  --query_max_len 32 --passage_max_len 512
```

### Train (multi-GPU with torchrun)
```bash
torchrun --nproc_per_node=4 -m tevatron.retriever.driver.train \
  --output_dir <output> --model_name_or_path <model> --lora --do_train ...
```

### Encode queries and corpus
```bash
python -m tevatron.retriever.driver.encode \
  --model_name_or_path <model> --lora_name_or_path <lora> \
  --dataset_name <dataset> --dataset_split test --encode_is_query \
  --encode_output_path queries.pkl --bf16 --pooling last --normalize

python -m tevatron.retriever.driver.encode \
  --model_name_or_path <model> --lora_name_or_path <lora> \
  --dataset_name <corpus> --encode_output_path corpus.pkl \
  --passage_max_len 512 --bf16 --pooling last --normalize
```
Encoding is single-GPU only (multi-GPU raises `NotImplementedError`). For large corpora, use `--dataset_number_of_shards` / `--dataset_shard_index` to shard across jobs.

### Search
```bash
python -m tevatron.retriever.driver.search \
  --query_reps queries.pkl --passage_reps 'corpus*.pkl' \
  --depth 100 --batch_size 64 --save_text --save_ranking_to rank.txt
```

### Evaluate
```bash
python -m tevatron.utils.format.convert_result_to_trec --input rank.txt --output rank.trec --remove_query
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 <qrels> rank.trec
```

### Tests
```bash
pytest tests/test_chunking.py -m unit   # fast unit tests (no downloads)
pytest tests/                           # all tests
```

## Architecture (`src/tevatron/retriever/`)

### Model hierarchy (`modeling/`)

`EncoderModel` (abstract base, `encoder.py`) defines the dual-encoder contract:
- `encode_query(qry)` → `[B, H]` tensor
- `encode_passage(psg, eos_positions=None)` → `[B, H]` or `([B, C, H], [B, C])` for chunked
- `forward(query, passage)` → `EncoderOutput(q_reps, p_reps, loss, scores, chunk_mask)`
- `compute_similarity(q, p)` → dot product `[Q, P]`
- `compute_maxsim_similarity(q, p, mask)` → max chunk similarity `[Q, P]`
- `build()` class method for training (loads base model, optionally wraps with LoRA via peft)
- `load()` class method for inference (loads base + merges LoRA weights)

`DenseModel` (`dense.py`) is the concrete implementation. It adds:
- `_pooling(hidden_states, attention_mask)` — supports `cls`, `mean`/`avg`, `last`/`eos`. For `last`/`eos`, auto-detects left-padding (takes `[:, -1]`) vs right-padding (takes position at `attention_mask.sum()-1`).
- `_pooling_chunked(hidden_states, eos_positions)` — extracts the hidden state at each EOS position as the chunk embedding. Returns `(chunk_reps [B, C, H], chunk_mask [B, C])`. Synchronizes `max_chunks` across DDP ranks via `all_reduce(MAX)` so tensor shapes match for gathering.
- `passage_chunk_size` attribute: `0` = no chunking, `>0` = chunked mode (the actual per-chunk token count is determined by the collator, this is just a boolean signal).

### Collator (`collator.py`)

Handles tokenization, chunking, and padding. Key components:

**`_chunk_tokens(tokens, chunk_size, eos_token_id, max_length, chunk_size_range, passage_seed)`**:
Splits a token list into chunks, appending EOS after each chunk. Each chunk is `chunk_size - 1` content tokens + 1 EOS token. Returns `(chunked_ids, eos_positions)`. When `chunk_size_range=(min, max)` is provided, each chunk gets a random size seeded by `passage_seed * 31 + chunk_index` for DDP-safe reproducibility.

**`_deterministic_hash(s)`**: MD5-based hash (truncated to 32 bits) used instead of Python's `hash()` which is randomized across processes. This ensures the same passage gets the same chunk boundaries on every DDP rank.

**`_pad_and_adjust_eos_positions()`**: After padding, shifts EOS positions by the padding amount when using left-padding.

**Collator classes**:
- `TrainCollator` — returns `(q_collated, d_collated)` or `(q_collated, d_collated, eos_positions)` when chunking is enabled
- `EncodeCollator` — standard encoding, returns `(content_ids, collated_inputs)`
- `ChunkedEncodeCollator` — chunked encoding for inference, returns `(doc_ids, collated_inputs, eos_positions)`. Supports fixed, per-passage random, and per-chunk variable random sizes.
- `PreChunkedEncodeCollator` — expects dataset with `chunks` field (list of strings), tokenizes each chunk separately and inserts EOS between them

Chunking mode is selected by these `DataArguments`:
- `passage_chunk_size > 0` → fixed chunking
- `passage_chunk_size_range = "min,max"` → random chunking
- `passage_chunk_size_variable = True` → each chunk within a passage gets a different random size (vs all chunks in a passage sharing one random size)
- `encode_use_pre_chunked = True` → pre-chunked from dataset

### Trainer (`trainer.py`)

`TevatronTrainer` extends HuggingFace `Trainer`. Key behaviors:

**`compute_loss(model, inputs)`**: Unpacks `(query, passage, [eos_positions])` from the collator. Attaches `eos_positions` to the unwrapped model as an attribute (read by `forward()` via `getattr`). After forward pass, gathers `q_reps`, `p_reps`, and `chunk_mask` across DDP ranks. Computes contrastive loss with in-batch negatives: target for query `i` is passage `i * train_group_size`.

**`_dist_gather_tensor(t, name)`**: Gathers tensors across DDP ranks with safety checks — verifies all ranks have the same tensor presence (None vs non-None) and same shape (except dim 0) to prevent NCCL deadlocks. Keeps local tensor in the gathered result for gradient flow.

**Loss scaling**: Loss is multiplied by `world_size` in `compute_loss` then divided by `world_size` in `training_step` — this is because HF Trainer averages gradients across ranks, but the contrastive loss should use the full gathered batch.

**`GradCacheTrainer`** (`gc_trainer.py`): Memory-efficient alternative using the GradCache library. Splits queries/passages into sub-batches, computes gradients incrementally. Uses `SimpleContrastiveLoss` or `DistributedContrastiveLoss`.

### Dataset (`dataset.py`)

**`TrainDataset`**: Each item returns `(formatted_query, formatted_documents)` where documents is a list of `(text, image, video, audio)` tuples. The first document is a positive, the rest are negatives (count = `train_group_size - 1`). Supports two formats:
- Legacy: inline `positive_passages`/`negative_passages` with text
- New: `positive_document_ids`/`negative_document_ids` referencing a separate corpus

Negative selection uses deterministic shuffling seeded by `hash(item + seed)` with epoch-based offset rotation.

**`EncodeDataset`**: Returns `(content_id, text, image, video, audio)`. For pre-chunked mode, `text` is a list of chunk strings instead of a single string.

### Drivers (`driver/`)

- `train.py` — parses `ModelArguments`, `DataArguments`, `TrainingArguments` via `HfArgumentParser`. Notable: sets `tokenizer.eos_token_id = tokenizer.pad_token_id` (they share the same token). Sets `model.passage_chunk_size = 1` when random chunking is used (as a signal to enable chunked codepath).
- `encode.py` — single-GPU only. Selects collator based on chunking mode. For chunked output, stores each chunk as a separate embedding with `(doc_id, chunk_idx)` lookup tuple. Non-chunked stores flat `doc_id` lookup.
- `search.py` — loads pickled embeddings into FAISS `IndexFlatIP`. Auto-detects chunked format by checking if lookup entries are tuples. Chunked search uses `search_queries_chunked()`: searches at `depth * chunk_multiplier` chunks, then aggregates to document-level scores via MaxSim (max score per doc across its chunks).

### Data flow summary

```
Train:  HF dataset → TrainDataset.__getitem__ → TrainCollator.__call__ → TevatronTrainer.compute_loss
        → model.forward(query, passage) → gather across ranks → contrastive loss

Encode: HF dataset → EncodeDataset → [Chunked]EncodeCollator → model.encode_query/passage
        → pickle.dump((embeddings, lookup_ids))

Search: pickle.load → FAISS index → top-k → [MaxSim aggregation] → ranking file
```

### Key argument groups (`arguments.py`)

**ModelArguments**: `model_name_or_path`, `pooling` (cls/mean/last), `normalize`, `temperature`, LoRA config (`lora`, `lora_r=16`, `lora_alpha=64`, `lora_target_modules`), `attn_implementation` (default: flash_attention_2)

**DataArguments**: `dataset_name`, `query_max_len` (32), `passage_max_len` (128), `train_group_size` (8), `query_prefix`/`passage_prefix`, `padding_side` (right), `passage_chunk_size`, `passage_chunk_size_range`, `passage_chunk_size_variable`, `encode_use_pre_chunked`, `append_eos_token`, `pad_to_multiple_of` (16)

**TevatronTrainingArguments** (extends HF TrainingArguments): `warmup_ratio` (0.1), `grad_cache`, `gc_q_chunk_size` (4), `gc_p_chunk_size` (32)

## Code Review Notes

### Chunked training + DDP + MaxSim correctness (verified)

The full path — collator → trainer.compute_loss → model.forward → encode_passage → _pooling_chunked → DDP gather → compute_maxsim_similarity → loss — has been traced end-to-end and is correct:

- **DDP shape sync**: `_pooling_chunked` uses `all_reduce(MAX)` on `max_chunks` before creating output tensors, so all ranks produce `[B, max_chunks, H]` and `[B, max_chunks]`. The trainer's `_dist_gather_tensor` then concatenates along dim 0 safely.
- **MaxSim**: `einsum('qh,pch->qpc')` computes per-chunk scores, padding chunks are masked to `-inf`, `max(dim=-1)` selects the best chunk. Gradients flow through `max` correctly.
- **Normalization of zero-padded chunks**: `F.normalize` uses `eps=1e-12`, so zero vectors stay zero (no NaN). Masked to `-inf` in MaxSim, so they never affect ranking.
- **Loss scaling**: `loss * world_size` in `compute_loss` then `/ world_size` in `training_step` correctly compensates for HF Trainer's gradient averaging.
- **eos_positions attribute passing**: Trainer sets on `model.module` (unwrapped), `forward()` reads via `getattr(self, ...)` on the same DenseModel instance. Correct with DDP.
- **eos_token_id == pad_token_id**: Both `train.py` and `encode.py` set `tokenizer.eos_token_id = tokenizer.pad_token_id`. Not a problem because `_pooling_chunked` uses explicit position indices, not token scanning. Attention mask correctly masks out padding.

### Random chunking consistency across GPUs/runs (verified)

- **Training (DDP)**: Each GPU sees different passages (disjoint batches), so cross-rank chunk consistency doesn't matter. Random chunking acts as data augmentation.
- **Encoding (sharded)**: Each shard is a disjoint partition of the corpus (`--dataset_shard_index`). No passage is encoded by more than one GPU, so cross-GPU consistency is irrelevant.
- **Encoding re-runs**: `_deterministic_hash` (MD5-based) ensures the same passage always gets the same chunk boundaries, regardless of which machine or process runs it. This was the fix for the earlier bug where Python's `hash()` randomization caused different chunks across processes.
- **Train vs eval chunks differ by design**: Training uses random chunk sizes (augmentation). Evaluation uses fixed `--passage_chunk_size`. These are intentionally different — the experiment measures how models trained with random chunks perform when evaluated with each fixed chunk size.

### Known minor issues (non-critical)

- `encode.py:142` prints `eos_positions` on every batch — will spam stdout on large corpora. Should be `logger.debug` or removed.
- `collator.py:14` sets `torch.set_printoptions(threshold=float('inf'))` at module level — globally changes torch print behavior for the entire process. Should be removed or guarded.
- `encode.py:88-95` has six `print()` calls logging data_args on every run. Should be `logger.info` or removed.
- `encode.py` uses inconsistent encoding paths: chunked calls `model.encode_passage()` directly (line 143), non-chunked goes through `model(passage=...)` → `forward()` (line 161). Both work but the chunked path skips `forward()`'s safety checks.
- `EncoderModel._dist_gather_tensor` (`encoder.py:186`) is a simple version without None-consistency or shape checks. Not used in the main training path (the trainer's safe version is used), but could cause NCCL deadlocks if called directly in other contexts.
