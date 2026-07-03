# Megatron reranker training (Tevatron 3.0)

Train LLM rerankers with Tevatron's **Megatron-Core backend**: tensor / pipeline
/ **expert** parallelism and a ZeRO-1 distributed optimizer, for dense and MoE
models that the HuggingFace-Trainer (FSDP1) path can't reach at this budget.
Same data path, same eval pipeline, same HF-loadable checkpoint format — only
the trainer differs.

A reranker is scored as a plain causal LM: `logit(' yes') − logit(' no')` at the
final prompt token, with listwise groups of one positive + `G−1` hard negatives.

> **Install first.** The Megatron stack (TransformerEngine, Megatron-Core,
> mbridge, the venvs) is set up per [`docs/megatron_installation.md`](../../docs/megatron_installation.md).
> Full flag reference and parallelism recipes:
> [`docs/megatron_reranker_usage.md`](../../docs/megatron_reranker_usage.md).
> On this cluster always export `NVTE_FUSED_ATTN=0` (dual CUDA-runtime conflict).

Two venvs: **`.venv`** for full-parameter (mbridge save path), **`.venv-lora`**
for LoRA (megatron-bridge, saves pre-merged HF weights).

---

## 1. Contrastive training (listwise CE)

### Dense 8B — one 8×H200 node, TP=2 / DP=4

```bash
NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=8 -m tevatron.megatron.driver.train \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --dataset_name rlhn/rlhn-680K \
    --tensor_model_parallel_size 2 \
    --train_group_size 8 --micro_batch_size 4 --num_micro_batches 4 \
    --max_seq_len 512 --learning_rate 1e-5 --min_lr 0 \
    --num_epochs 1 --warmup_ratio 0.2 --grad_clip 1.0 \
    --save_dir output/qwen3-8b-reranker --save_interval 1000
```

`TP=2` is light model sharding (8B fits easily; no need for TP=8's comm tax);
`DP=4` is derived as `world_size/(TP·PP)` and lets the distributed optimizer
shard the Adam states. Effective batch = `DP · micro_batch_size ·
num_micro_batches` = `4·4·4` = 64 queries/step.

### MoE 30B-A3B — two 8×H200 nodes, EP=16 / DP=16

Expert parallelism shards the 128 experts across ranks — the reason the backend
exists. EP is orthogonal to DP (every GPU is still its own DP rank). Multi-node
needs a rendezvous; use the SLURM wrapper:

```bash
sbatch scripts/megatron_support/train_qwen3_30b_a3b_reranker_2node.slurm
# inner command (per node), for reference:
#   torchrun --nnodes=2 --nproc_per_node=8 ... -m tevatron.megatron.driver.train \
#       --model_name_or_path Qwen/Qwen3-30B-A3B-Base --dataset_name rlhn/rlhn-680K \
#       --expert_model_parallel_size 16 \
#       --train_group_size 8 --micro_batch_size 1 --num_micro_batches 4 ...
```

MoE micro-batch is capped at 1 (MBS≥4 OOMs); reach the batch with
`--num_micro_batches`.

### LoRA (run from `.venv-lora`)

Add `--use_lora` and target groups. **On MoE, use the expert-aware spec** — a
dense MLP pattern silently matches all 128 expert FFNs (see usage doc):

```bash
# dense 8B LoRA: attention + MLP
--use_lora --lora_rank 16 --lora_alpha 32 --lora_target_groups attn mlp

# MoE 30B-A3B LoRA: attention + experts + shared expert (router frozen)
--use_lora --lora_rank 16 --lora_alpha 32 --lora_target_groups attn moe_experts moe_shared
```

Checkpoints are written in **HF format** (LoRA pre-merged), so they load through
the normal HF / vLLM eval path with no conversion.

---

## 2. Distillation training (listwise KL vs a teacher)

Two steps: (a) annotate the training set with teacher scores **once, offline**,
then (b) train on the annotated set. Because teacher scores are precomputed,
distillation training has **no teacher forward pass** — it costs the same as
contrastive and uses identical parallelism flags.

### Step (a) — create the distillation dataset from the teacher

`tevatron.utils.annotate_with_teacher` scores every (query, candidate) pair with
a teacher reranker (via vLLM) and writes a HF dataset with per-passage `scores`
(raw `logit_yes − logit_no`). Run from **`.venv-eval`** (has vLLM):

```bash
NVTE_FUSED_ATTN=0 python -m tevatron.utils.annotate_with_teacher \
    --teacher_model_name_or_path Qwen/Qwen3-Reranker-8B \
    --prompt_template qwen3_reranker \
    --source_dataset_name rlhn/rlhn-680K \
    --max_negatives 15 \
    --rerank_max_len 2048 --max_model_len 2304 \
    --tensor_parallel_size 8 \
    --top_logprobs 200 \
    --output_dir /path/to/rlhn-680K-qwen3-reranker-8b
```

> **`--top_logprobs 200`, not the default 20.** For confident teachers the
> losing token (`' no'`) often falls outside the top-20 logprobs and gets
> clipped to a sentinel, which flattens the teacher's confidence and acts like
> an implicit higher temperature. Top-200 captures it; the cost is negligible
> (logprobs are read off the full softmax). See
> [`docs/bugs-and-fixes.md`](../../docs/bugs-and-fixes.md).

Output schema (one row per query):

```
{ "query_id": str, "query": str,
  "passages": [{"title": str, "text": str}, ...],   # passages[0] is the positive
  "scores":   [float, ...] }                         # teacher logit_yes - logit_no, parallel to passages
```

The released `rlhn-680K-qwen3-reranker-8b-top200` dataset is exactly this output.

### Step (b) — train on the annotated dataset

Same driver family, `distill_train` instead of `train`; identical parallelism
flags plus three distillation knobs (`--distill_dataset_path`,
`--teacher_temp`, `--student_temp`). We use `teacher_temp=2.0`.

```bash
# Dense 8B, TP=2 / DP=4:
NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=8 -m tevatron.megatron.driver.distill_train \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --distill_dataset_path /path/to/rlhn-680K-qwen3-reranker-8b \
    --tensor_model_parallel_size 2 \
    --train_group_size 8 --micro_batch_size 4 --num_micro_batches 4 \
    --teacher_temp 2.0 --student_temp 1.0 \
    --max_seq_len 512 --learning_rate 1e-5 --num_epochs 1 --warmup_ratio 0.2 \
    --save_dir output/qwen3-8b-reranker-distill --save_interval 1000
```

MoE and LoRA work exactly as in contrastive training — same EP / `--use_lora`
/ target-group flags, only the driver and the three distillation flags differ.

> Note: teacher and student here are the **same size** (8B teacher → 8B
> student), which leaves no capacity headroom; in our study distillation behaves
> as a recipe-dependent regularizer rather than a uniform win.

---

## 3. Evaluate the trained checkpoint

Megatron checkpoints are ordinary HF causal-LM dirs, so they evaluate through
the standard reranker eval pipeline (BEIR-15, HF or vLLM backend). See
[`src/tevatron/eval/README.md`](../../src/tevatron/eval/README.md) and `scripts/eval/`.

---

## Ready-to-run scripts

Full SLURM / single-node scripts for every config (8B & 30B-A3B × contrastive &
distill × full-FT & LoRA, plus the annotate job) live in
[`scripts/megatron_support/`](../../scripts/megatron_support/). Set your own
`WANDB_*` and model paths before launching.
