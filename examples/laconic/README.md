# LACONIC: decoder-LM SPLADE in Tevatron

Reproduces [**LACONIC**](https://arxiv.org/abs/2601.01684) — *Dense-Level
Effectiveness for Scalable Sparse Retrieval via a Two-Phase Training
Curriculum* — using Tevatron's decoder-LM SPLADE (`SpladeModelForCausalLM`).

LACONIC turns a decoder LM (Llama 3 / Qwen2.5) into a learned sparse retriever:
convert it to **bidirectional** attention, project hidden states through the
(tied) `lm_head` to vocabulary logits, **max-pool** over the sequence into one
sparse vocab-space vector, and train contrastively with **FLOPS** sparsity
regularization. Retrieval is over a [Seismic](https://github.com/TusKANNy/seismic)
sparse index.

The core logic lives in the main package (see "What was migrated" below); this
folder is just the runnable recipe plus the Seismic-dependent evaluation, which
we keep out of the core so the training path has no Seismic dependency.

---

## ⚠️ Two encoding details that make or break it

The public checkpoint (`utahnlp/laconic-1b`) silently produces **garbage**
(scifact NDCG@10 ≈ 0.02, every document mapping to the same generic tokens) if
either is wrong. Both are now controllable from the CLI:

1. **No BOS token — `--add_special_tokens False` (REQUIRED).**
   `<|begin_of_text|>` is an attention sink: its hidden state projects very large
   logits onto generic tokens (`Question`, `#`, `def`, …). Under SPLADE max-pool
   that single position dominates the pooled vector for *every* input, collapsing
   all representations (pairwise cosine ≈ 0.99). LACONIC trained and encoded
   without special tokens, so the sink never enters the representation.

   > This is **not** a transformers-version issue. transformers 4.51.3 and 5.9
   > behave identically here; the earlier "downgrade to 4.51.3 fixes it" folklore
   > was incidental. The real lever is the BOS token.

2. **E5-style prefixes — `--query_prefix "query: "` / `--passage_prefix "passage: "` (REQUIRED).**
   LACONIC trains with these prefixes; dropping them costs ~10 nDCG points.

### Measured effect (scifact, `utahnlp/laconic-1b`, transformers 5.9)

| Encoding | scifact NDCG@10 |
|---|---|
| BOS on (default), no prefix | 0.022 |
| **no BOS**, no prefix | 0.648 |
| **no BOS + `query:`/`passage:` prefixes** | **0.752** |
| Paper | 0.756 |

The 0.752 vs 0.756 gap is Seismic approximate-ANN nondeterminism.

---

## Environment

Decoder-LM SPLADE runs on the **main `.venv`** (transformers 5.x). The
bidirectional conversion uses transformers 5.x's `create_bidirectional_mask`
(see `tevatron.retriever.modeling.bidirectional`); no `llm2vec` dependency.

A separate **`.venv-laconic`** (transformers 4.51.3 stack) exists only for
cross-checking against the legacy repo; you do **not** need it to reproduce.

Search + eval use Seismic (`.venv`, has `seismic`) and BEIR/pytrec_eval
(`.venv-eval`).

```bash
# .venv         : training + encoding (transformers 5.x) + seismic search
# .venv-eval    : beir + pytrec_eval scoring
```

---

## Reproduce (single dataset)

`run_beir.sh` does encode → seismic search → BEIR score for one dataset. The
tokenizer comes from the Llama-3 base (the checkpoint ships no tokenizer files);
any ungated Llama-3 tokenizer mirror works.

```bash
cd examples/laconic
MODEL=utahnlp/laconic-1b TOKENIZER=unsloth/Llama-3.2-1B DATASET=scifact ./run_beir.sh
# => scifact NDCG@10 ≈ 0.752
```

Key flags it passes to `encode_splade` (the non-obvious ones):

```bash
--splade_model_type causal      # decoder-LM SPLADE (vs the default MLM SPLADE)
--is_bidirectional              # full attention (Llama3/Qwen2.5)
--pooling_strategy max          # SPLADE max-pool
--add_special_tokens False      # NO BOS  <-- critical
--query_prefix "query: "        # E5-style prefixes  <-- critical
--passage_prefix "passage: "
--splade_topk 512               # keep top-512 terms/vector (decoder logits are dense)
```

## Train

`SpladeModelForCausalLM` + FLOPS regularization (`SpladeTrainer`). LoRA on the
decoder, bidirectional attention, contrastive listwise CE:

```bash
torchrun --nproc_per_node=8 -m tevatron.retriever.driver.train_splade \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --dataset_name rlhn/rlhn-680K \
    --splade_model_type causal --is_bidirectional --pooling_strategy max \
    --add_special_tokens False --query_prefix "query: " --passage_prefix "passage: " \
    --lora --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --q_flops_loss_factor 1e-3 --p_flops_loss_factor 1e-3 --flops_warmup 100 \
    --bf16 --train_group_size 16 --query_max_len 192 --passage_max_len 192 \
    --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.05 \
    --num_train_epochs 1 --output_dir model_laconic
```

---

## What was migrated (and the design principle)

LACONIC was a fork of Tevatron-v1 plus the full `llm2vec` package (~1.1k LoC).
Migration to v3 kept **only the genuinely new logic**, with minimum modification
to the existing framework:

| LACONIC piece | v3 home |
|---|---|
| Decoder-LM SPLADE (max/mean/last pooling) | `SpladeModelForCausalLM` in `retriever/modeling/splade.py` (the MLM `SpladeModel` is untouched) |
| Bidirectional attention (whole `llm2vec` pkg) | `retriever/modeling/bidirectional.py` — a ~30-line in-place mask swap on transformers 5.x; no `llm2vec` |
| FLOPS regularizer + warmup schedule | `retriever/splade_trainer.py` (`SpladeTrainer`); base `TevatronTrainer` untouched |
| No-BOS / prefix encoding | `--add_special_tokens` arg + existing `--query_prefix`/`--passage_prefix`, threaded through the existing collators |
| Seismic encode/search/eval | already in v3 (`encode_splade`, `search_splade`, `eval.metrics`); only `--splade_topk` was added |

Echo embedding and hard-top-k-in-model were intentionally **not** ported (rarely
used; not needed to reproduce the headline results).
