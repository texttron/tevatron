# Megatron Reranker — Usage Guide

How to train rerankers with Tevatron's Megatron backend: parallelism strategies
and the flags that set them, contrastive vs distillation training, LoRA target
selection, and the checkpoint format. For installing the stack (TransformerEngine,
Megatron-Core, mbridge, the venvs), see
[megatron_installation.md](megatron_installation.md) first.

The Megatron backend exists for what HF Trainer can't reach at this budget:
expert-parallel MoE training, tensor/pipeline parallelism, a ZeRO-1 distributed
optimizer, and activation recompute — enough to fit large dense and MoE models on
one node. Same data path, same eval pipeline, same HF-loadable checkpoint format
as the rest of Tevatron — only the trainer
under the hood differs.

---

## TL;DR

```bash
# Dense 8B, contrastive full-FT, TP=2 + DP=4 on one 8-GPU node (.venv):
torchrun --nproc_per_node=8 -m tevatron.megatron.driver.train \
    --model_name_or_path /fsx/.../Qwen3-8B-Base --dataset_name rlhn/rlhn-680K \
    --tensor_model_parallel_size 2 \
    --train_group_size 8 --micro_batch_size 4 --num_micro_batches 4 \
    --max_seq_len 512 --learning_rate 1e-5 --num_epochs 1 --warmup_ratio 0.2 \
    --save_dir output/qwen3-8b-reranker

# MoE 30B-A3B, contrastive full-FT, EP=16 + DP=16 on two 8-GPU nodes (.venv):
#   (launch via the SLURM wrapper, which sets the rendezvous; see "Multi-node")
sbatch scripts/megatron_support/train_qwen3_30b_a3b_reranker_2node.slurm
```

Two entry points:
- `tevatron.megatron.driver.train` — contrastive (listwise CE)
- `tevatron.megatron.driver.distill_train` — listwise-KL distillation

Two venvs (see installation guide):
- `.venv` — full-FT (mbridge save path)
- `.venv-lora` — LoRA (megatron-bridge, which saves pre-merged HF weights)

---

## Parallelism strategies

Four parallel dimensions compose. The backend reads three from flags; **data
parallel (DP) is derived**, not set directly:

```
DP = world_size / (TP × PP)          # EP is orthogonal — see below
world_size = nnodes × 8
```

| Dimension | Flag | What it shards | When to use |
|---|---|---|---|
| **Tensor (TP)** | `--tensor_model_parallel_size` | each layer's matmuls across ranks | dense models too big / to relieve memory; light TP=2 for 8B |
| **Pipeline (PP)** | `--pipeline_model_parallel_size` | layer *stages* across ranks | very deep models / many nodes; usually 1 here |
| **Expert (EP)** | `--expert_model_parallel_size` | the MoE experts across ranks | **MoE only** — the reason the backend exists |
| **Data (DP)** | *(derived)* | the global batch across replicas | always; comes for free from the leftover ranks |

### Data-parallel sharding: ZeRO-1

DP uses Megatron's **distributed optimizer** (ZeRO-1): the optimizer states
(Adam m/v + fp32 master) are sharded across DP ranks, while **parameters and
gradients are replicated**. This is the lightest-communication scheme and is the
only DP sharding mode the backend currently supports.

When you need to relieve more per-rank memory than ZeRO-1 gives, use **tensor or
expert parallelism**, not a higher ZeRO stage:
- dense models too large for ZeRO-1 alone → raise TP (e.g. dense 8B with TP=2);
- MoE models → EP shards the experts, which is the real memory wall.

In practice this covers everything in the reranker recipes here: dense 8B fits
with TP=2/DP=4 (ZeRO-1), and 30B-A3B MoE fits with EP=16/DP=16 (ZeRO-1). FSDP-style
ZeRO-2/3 (gradient/parameter sharding) is **not** currently supported — see the
[Roadmap](#roadmap) and the difficulty writeup in
[bugs-and-fixes.md](bugs-and-fixes.md).

### EP is orthogonal to DP

A common confusion: EP does **not** consume the DP budget. Every GPU is still its
own DP rank; EP just changes *which expert weights* live on each rank. So a
single 8-GPU node with `--expert_model_parallel_size 8` gives **DP=8, EP=8**
simultaneously (8-way data parallel *and* the 128 experts sharded 8 ways), not
DP=1. EP must divide `num_experts` and divide `world_size`.

### Worked layouts (all hold a 64-query global batch)

Global batch in sequences = `DP × micro_batch_size × num_micro_batches ×
train_group_size`. With `train_group_size=8`, 64 queries/step = 512 seqs.

| Model | Node(s) | Flags | Derived | MBS × n_micro | = seqs |
|---|---|---|---|---|---|
| Dense 8B | 1 (8 GPU) | `--tensor_model_parallel_size 2` | TP=2, **DP=4** | 4 × 4 | 512 ✓ |
| MoE 30B-A3B | 1 (8 GPU) | `--expert_model_parallel_size 8` | EP=8, **DP=8** | 1 × 8 | 512 ✓ |
| MoE 30B-A3B | 2 (16 GPU) | `--expert_model_parallel_size 16` | EP=16, **DP=16** | 1 × 4 | 512 ✓ |

Notes:
- **Dense 8B → TP=2** is a *memory* choice: full-FT 8B + AdamW doesn't fit a
  141 GB H200 under DP-only; TP=2 halves the per-rank param/activation footprint.
  TP=2 (not TP=8) keeps the per-layer all-reduce tax low since 8B fits easily
  once lightly sharded.
- **MoE → EP, never TP** for memory. The MoE's memory pressure is the 128
  experts, which EP shards; the attention/dense trunk (hidden=2048) is small.
  Adding TP would only pay comm tax to relieve memory EP already relieves.
- **MoE MBS is capped at 1** on these models (MBS≥4 OOMs); reach the batch with
  `--num_micro_batches` instead.

### Multi-node

Single-node runs launch with plain `torchrun --nproc_per_node=8`. Multi-node
needs a rendezvous; use the SLURM wrappers, which set `MASTER_ADDR`/`MASTER_PORT`
from the allocation and `srun` the inner script on every node (it reads
`SLURM_NNODES`/`SLURM_NODEID`):

```bash
sbatch scripts/megatron_support/train_qwen3_30b_a3b_reranker_2node.slurm
```

---

## Contrastive vs distillation training

Two losses, two drivers, same parallelism flags.

### Contrastive — `tevatron.megatron.driver.train`

Listwise cross-entropy on the log-odds of the `group_size` candidates, target =
index 0 (the positive). One positive vs `G−1` hard negatives, no teacher. The
score per pair is `logit(' yes') − logit(' no')` at the prompt's final `?`.

```bash
torchrun --nproc_per_node=8 -m tevatron.megatron.driver.train \
    --model_name_or_path /fsx/.../Qwen3-8B-Base \
    --dataset_name rlhn/rlhn-680K \
    --tensor_model_parallel_size 2 \
    --train_group_size 8 --micro_batch_size 4 --num_micro_batches 4 \
    --max_seq_len 512 --learning_rate 1e-5 --warmup_ratio 0.2 --num_epochs 1 \
    --save_dir output/qwen3-8b-reranker
```

### Distillation — `tevatron.megatron.driver.distill_train`

Listwise KL (top-1 ListNet) against a teacher's logits over the same `G`
candidates. The teacher scores are **precomputed offline** into the dataset
(no teacher forward at train time — distillation costs ~the same as contrastive).
Adds three flags; everything else is identical:

| Flag | Meaning |
|---|---|
| `--distill_dataset_path` | dataset with per-pair `teacher_scores` (required) |
| `--teacher_temp` | softmax temperature on teacher logits (default 1.0; we use 2.0) |
| `--student_temp` | softmax temperature on student logits (default 1.0) |

```bash
torchrun --nproc_per_node=8 -m tevatron.megatron.driver.distill_train \
    --model_name_or_path /fsx/.../Qwen3-8B-Base \
    --distill_dataset_path /fsx/.../rlhn-680K-qwen3-reranker-8b-top200 \
    --tensor_model_parallel_size 2 \
    --train_group_size 8 --micro_batch_size 4 --num_micro_batches 4 \
    --teacher_temp 2.0 --student_temp 1.0 \
    --max_seq_len 512 --learning_rate 1e-5 --warmup_ratio 0.2 --num_epochs 1 \
    --save_dir output/qwen3-8b-reranker-distill
```

`temperature` is a real lever (sharper teacher targets vs label-smoothing); we
fix `teacher_temp=2.0` and do not sweep it. See paper notes.

---

## LoRA

Add `--use_lora` (run from `.venv-lora`). Rank/alpha/dropout:
`--lora_rank 16 --lora_alpha 32 --lora_dropout 0.0`. LR is typically bumped ~10×
over full-FT.

### Target selection — use named groups, not raw patterns

`--lora_target_groups` picks adapters by *role*, expanding to the right
module patterns. This avoids the **MoE foot-gun**: the dense default
`--lora_target_modules linear_qkv linear_proj linear_fc1 linear_fc2` matches
`linear_fc1/fc2` on *every one of the 128 expert FFNs* on an MoE model.

| Group | Modules | Notes |
|---|---|---|
| `attn` | `linear_qkv`, `linear_proj` | safe for any architecture |
| `mlp` | `linear_fc1`, `linear_fc2` | dense MLP — **on MoE this hits every expert**; use the moe_* groups instead |
| `moe_experts` | `*.experts.*.linear_fc1/fc2` | the 128 routed expert FFNs (where MoE reranking signal lives) |
| `moe_shared` | `*.shared_experts.linear_fc1/fc2` | the shared-expert branch (skipped silently if absent) |
| `moe_router` | `*.router.weight` | the top-k gate — **leave frozen**: rank-16 on a `[128,hidden]` gate is near-full-rank and risks routing/load-balance instability for negligible gain |

```bash
# Dense 8B LoRA — attention + MLP:
--use_lora --lora_rank 16 --lora_alpha 32 --lora_target_groups attn mlp

# MoE 30B-A3B LoRA — expert-aware spec (attn-only fails on MoE; router frozen):
--use_lora --lora_rank 16 --lora_alpha 32 --lora_target_groups attn moe_experts moe_shared
```

`--lora_target_modules` remains as a power-user escape hatch for raw bridge
patterns the registry doesn't expose; prefer the groups.

---

## Checkpoints

The backend writes **HF-format** checkpoints (`config.json` + sharded
`model-*.safetensors` + tokenizer) to `--save_dir/step_<N>` every
`--save_interval` steps, via mbridge. They load directly through the normal
HF / vLLM path — no conversion step.

- **Full-FT** (`.venv`): mbridge `save_weights` path → plain HF causal-LM dir.
- **LoRA** (`.venv-lora`): the megatron-bridge save path writes **pre-merged**
  full weights (adapters folded into the base), so there is no separate adapter
  to merge — unlike the HF-PEFT path, which needs `tevatron.utils.merge_lora`.

All checkpoints are causal-LM yes/no rerankers (`lm_head`, scored on
`' yes'`/`' no'` at the final `?`) — distinct from the HF-Trainer reranker, which
is a sequence-classification head scored at EOS. Match the serving backend's
scoring mode to the checkpoint (see the serve docs / bugs-and-fixes.md).

---

## Common flags reference

| Flag | Default | Notes |
|---|---|---|
| `--model_name_or_path` | (required) | HF model dir; bridge reads its config |
| `--dataset_name` / `--dataset_path` | `json` / None | HF dataset id or local path |
| `--tensor_model_parallel_size` | 1 | TP |
| `--pipeline_model_parallel_size` | 1 | PP |
| `--expert_model_parallel_size` | 1 | EP (MoE) |
| `--recompute_enabled` | off | activation/gradient checkpointing (matches HF FSDP `activation_checkpointing`) |
| `--train_group_size` | 8 | candidates per query (1 pos + G−1 neg) |
| `--micro_batch_size` | 1 | queries per micro-batch (MoE: keep 1) |
| `--num_micro_batches` | 1 | grad-accum steps; raise to hit target batch |
| `--max_seq_len` | 512 | per query–passage pair |
| `--learning_rate` / `--min_lr` | 1e-5 / 1e-6 | LoRA: ~10× full-FT |
| `--warmup_ratio` | 0.0 | we use 0.2 |
| `--num_epochs` | 1 | |
| `--grad_clip` | 1.0 | |
| `--append_eos_token` | off | **leave OFF** for this causal-LM path (the yes/no probe is at `?`, not EOS) |
| `--use_lora` + `--lora_*` | off | see LoRA section |
| `--moe_router_load_balancing_type` | `aux_loss` | MoE |
| `--moe_aux_loss_coeff` | 1e-6 | MoE load-balancing weight |
| `--save_dir` / `--save_interval` | `output` / 1000 | HF-format checkpoints |
| `--wandb_project` / `--wandb_run_name` | tevatron-megatron / None | |

Ready-to-run scripts for every config live in
`scripts/megatron_support/` (e.g. `train_qwen3_8b_reranker.sh`,
`train_qwen3_30b_a3b_reranker_2node.sh`,
`distill_qwen3_30b_a3b_reranker_2node.sh`, and their `_lora` variants).

## Roadmap

Known gaps and planned work, in rough priority order:

- **FSDP-style ZeRO-2/3 data-parallel sharding.** Today DP is ZeRO-1 only
  (distributed optimizer); param/gradient sharding comes from TP/EP. Direct
  ZeRO-2/3 (Megatron-FSDP / Torch FSDP2) would help fit very large *dense* models
  where raising TP is undesirable. We attempted it and backed out due to
  tevatron↔mbridge↔mcore version coupling (the `custom_fsdp` module was removed;
  mbridge's wrapper selection has no path to the current `MCoreFSDP`); see the
  difficulty writeup in [bugs-and-fixes.md](bugs-and-fixes.md). The viable routes
  if revisited are Torch FSDP2 via mbridge's `use_torch_fsdp2` (accepting
  `reshard_after_forward` semantics rather than the ZeRO-2/3 stage distinction),
  or wrapping `MCoreFSDP` directly and bypassing mbridge's wrapper selection.
  Gated on a stable mbridge/mcore pairing.
- **Context parallelism (CP)** for long-context reranking (currently TP/PP/EP/DP
  only).
- **Pipeline parallelism (PP)** is wired but unexercised in the reranker recipes
  here; validate on deeper / multi-node configurations.
