#!/bin/bash
set -euo pipefail

# Log a command and execute it (supports VAR=val cmd ... syntax)
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
TRAIN_NAME="fixed-64-independent"
MODEL_DIR="/root/autodl-tmp/tevatron/models/${TRAIN_NAME}"
LOG_DIR="/root/autodl-tmp/tevatron/logs/training"
LOG_FILE="${LOG_DIR}/train_${TRAIN_NAME}.log"

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"

# Tee all stdout and stderr to log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "Training: ${TRAIN_NAME}"
echo "Started:  $(date)"
echo "Log file: ${LOG_FILE}"
echo "================================================================"

if [ ! -f "${MODEL_DIR}/adapter_config.json" ]; then
  run_cmd \
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 7) \
    torchrun --nproc_per_node 8 --master_port 60001 \
      -m tevatron.retriever.driver.train \
      --output_dir "${MODEL_DIR}" \
      --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
      --bf16 --pooling last --padding_side right --normalize \
      --attn_implementation sdpa \
      --do_train --lora \
      --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
      --save_steps 5000 \
      --dataset_name Shitao/MLDR --dataset_config en --dataset_split train \
      --query_prefix "Instruct: Given a question, retrieve documents that answer the question.\nQuery:" --passage_prefix "" \
      --temperature 0.03 \
      --per_device_train_batch_size 2 --train_group_size 4 \
      --learning_rate 1e-4 \
      --query_max_len 512 --passage_max_len 8192 \
      --passage_chunk_size 64 \
      --passage_chunk_independent \
      --num_train_epochs 2 \
      --logging_steps 10 \
      --overwrite_output_dir \
      --gradient_checkpointing \
      --gradient_accumulation_steps 1 \
      --run_name "${TRAIN_NAME}" \
      --report_to wandb
else
  echo "=== Skipping training (checkpoint exists at ${MODEL_DIR}) ==="
fi

echo ""
echo "=== Training done (${TRAIN_NAME}) at $(date) ==="
