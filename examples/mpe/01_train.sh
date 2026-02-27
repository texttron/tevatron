#!/bin/bash
set -euo pipefail
# ══════════════════════════════════════════════════════════════════════════════
# Click 1 of 2 — Train All Models
#
# Trains 4 models on MLDR-EN (Shitao/MLDR, en split) with 1 epoch:
#
#   Model               │ Chunk args
#   ────────────────────┼──────────────────────────────────────
#   nochunk-epoch1      │ --passage_chunk_size 0
#   maxp-train-epoch1   │ --passage_chunk_size 64 --passage_chunk_independent
#   fixed-64-epoch1     │ --passage_chunk_size 64
#   prand-32to1024-epoch1│ --passage_chunk_size_range 32,1024
#
# These 4 models support the 5 evaluation configs in 02_eval.sh:
#   Single-Vector  → nochunk-epoch1      (eval: no chunking)
#   MaxP           → nochunk-epoch1      (eval: independent chunk 64)
#   MaxP-Train     → maxp-train-epoch1   (eval: independent chunk 64)
#   MPE-fixed64    → fixed-64-epoch1     (eval: chunk 64)
#   MPE-rand       → prand-32to1024-epoch1 (eval: chunk 64)
#
# Usage:
#   bash 01_train.sh
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}}"
MODEL_ROOT="${EXP_ROOT}/models"
LOG_DIR="${EXP_ROOT}/logs/repro"
NUM_GPUS=8
export OMP_NUM_THREADS=1

mkdir -p "${LOG_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_cmd() {
  echo ""
  echo "[CMD] $*"
  echo ""
  if [[ "$1" == *=* ]]; then
    env "$@"
  else
    "$@"
  fi
}

train_model() {
  local train_name="$1"
  shift
  local extra_args="$@"
  local model_dir="${MODEL_ROOT}/${train_name}"
  local log_file="${LOG_DIR}/train_${train_name}.log"

  echo ""
  echo "================================================================"
  echo "  Training: ${train_name}"
  echo "  Model dir: ${model_dir}"
  echo "  Extra args: ${extra_args}"
  echo "  Started: $(date)"
  echo "================================================================"

  if [ -f "${model_dir}/adapter_config.json" ]; then
    echo "=== Skipping training (checkpoint exists at ${model_dir}) ==="
    return
  fi

  mkdir -p "${model_dir}"

  run_cmd \
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    torchrun --nproc_per_node ${NUM_GPUS} --master_port 60001 \
      -m tevatron.retriever.driver.train \
      --output_dir "${model_dir}" \
      --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
      --bf16 --pooling last --padding_side right --normalize \
      --attn_implementation sdpa \
      --do_train --lora \
      --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
      --save_steps 5000 \
      --dataset_name Shitao/MLDR --dataset_config en --dataset_split train \
      --query_prefix "Instruct: Given a question, retrieve documents that answer the question.\nQuery:" \
      --passage_prefix "" \
      --temperature 0.03 \
      --per_device_train_batch_size 2 --train_group_size 4 \
      --learning_rate 1e-4 \
      --query_max_len 512 --passage_max_len 8192 \
      ${extra_args} \
      --num_train_epochs 1 \
      --logging_steps 10 \
      --overwrite_output_dir \
      --gradient_checkpointing \
      --gradient_accumulation_steps 1 \
      --run_name "${train_name}" \
    2>&1 | tee -a "${log_file}"

  echo ""
  echo "=== Training done: ${train_name} at $(date) ==="
}

# ══════════════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Paper Reproduction — Step 1/2: Train All Models               ║"
echo "║  Started: $(date)                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# 1. nochunk  (for Single-Vector and MaxP eval)
train_model "nochunk-epoch1" \
  --passage_chunk_size 0

# 2. maxp-train  (for MaxP-Train eval: independent chunking during training)
train_model "maxp-train-epoch1" \
  --passage_chunk_size 64 --passage_chunk_independent

# 3. fixed-64  (for MPE-fixed64 eval)
train_model "fixed-64-epoch1" \
  --passage_chunk_size 64

# 4. prand-32to1024  (for MPE-rand eval: random chunk range, eval at 64)
train_model "prand-32to1024-epoch1" \
  --passage_chunk_size_range 32,1024

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Step 1/2 complete — all 4 models trained.                     ║"
echo "║  Finished: $(date)                                   ║"
echo "║                                                                ║"
echo "║  Next: bash 02_eval.sh                                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
