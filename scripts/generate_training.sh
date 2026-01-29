#!/bin/bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Edit these before running the generator.

BASE_MODEL="Qwen/Qwen3-Embedding-0.6B"
NUM_GPUS=8
MASTER_PORT=60001
PASSAGE_MAX_LEN=8192

EXP_ROOT="/root/autodl-tmp/tevatron"
MODELS_DIR="${EXP_ROOT}/models"
LOGS_DIR="${EXP_ROOT}/logs/training"

DATASET_NAME="Shitao/MLDR"
DATASET_CONFIG="en"
DATASET_SPLIT="train"

QUERY_PREFIX='Instruct: Given a question, retrieve documents that answer the question.\nQuery:'
PASSAGE_PREFIX=""
TEMPERATURE=0.03

PER_DEVICE_TRAIN_BATCH=2
TRAIN_GROUP_SIZE=4
LEARNING_RATE="1e-4"
QUERY_MAX_LEN=512
NUM_TRAIN_EPOCHS=2
SAVE_STEPS=5000

# ── Training configs ────────────────────────────────────────────────────────
# Format: "name|chunk_args"
TRAIN_CONFIGS=(
  "nochunk|--passage_chunk_size 0"
  "fixed-32|--passage_chunk_size 32"
  "fixed-64|--passage_chunk_size 64"
  "fixed-128|--passage_chunk_size 128"
  "fixed-256|--passage_chunk_size 256"
  "fixed-512|--passage_chunk_size 512"
  "fixed-1024|--passage_chunk_size 1024"
  "fixed-2048|--passage_chunk_size 2048"
  "fixed-4096|--passage_chunk_size 4096"
  "prand-32to64|--passage_chunk_size_range 32,64"
  "prand-32to128|--passage_chunk_size_range 32,128"
  "prand-32to256|--passage_chunk_size_range 32,256"
  "prand-32to512|--passage_chunk_size_range 32,512"
  "prand-32to1024|--passage_chunk_size_range 32,1024"
  "prand-32to2048|--passage_chunk_size_range 32,2048"
  "prand-32to4096|--passage_chunk_size_range 32,4096"
)

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR="scripts/training"
mkdir -p "${OUT_DIR}"

echo "Generating training scripts in ${OUT_DIR}/ ..."

# ── Generate one train script per training config ────────────────────────────
for entry in "${TRAIN_CONFIGS[@]}"; do
  TRAIN_NAME="${entry%%|*}"
  TRAIN_CHUNK_ARGS="${entry#*|}"

  cat > "${OUT_DIR}/train_${TRAIN_NAME}.sh" <<SCRIPT
#!/bin/bash
set -euo pipefail

TRAIN_NAME="${TRAIN_NAME}"
MODEL_DIR="${MODELS_DIR}/\${TRAIN_NAME}"
LOG_DIR="${LOGS_DIR}"
LOG_FILE="\${LOG_DIR}/train_\${TRAIN_NAME}.log"

mkdir -p "\${MODEL_DIR}" "\${LOG_DIR}"

# Tee all stdout and stderr to log file
exec > >(tee -a "\${LOG_FILE}") 2>&1

echo "================================================================"
echo "Training: \${TRAIN_NAME}"
echo "Started:  \$(date)"
echo "Log file: \${LOG_FILE}"
echo "================================================================"

if [ ! -f "\${MODEL_DIR}/adapter_config.json" ]; then
  TRAIN_CMD="CUDA_VISIBLE_DEVICES=\$(seq -s, 0 $((NUM_GPUS-1))) \\\\
  torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \\\\
    -m tevatron.retriever.driver.train \\\\
    --output_dir \${MODEL_DIR} \\\\
    --model_name_or_path ${BASE_MODEL} \\\\
    --bf16 --pooling last --padding_side right --normalize \\\\
    --attn_implementation sdpa \\\\
    --do_train --lora \\\\
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \\\\
    --save_steps ${SAVE_STEPS} \\\\
    --dataset_name ${DATASET_NAME} --dataset_config ${DATASET_CONFIG} --dataset_split ${DATASET_SPLIT} \\\\
    --query_prefix '${QUERY_PREFIX}' --passage_prefix '${PASSAGE_PREFIX}' \\\\
    --temperature ${TEMPERATURE} \\\\
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH} --train_group_size ${TRAIN_GROUP_SIZE} \\\\
    --learning_rate ${LEARNING_RATE} \\\\
    --query_max_len ${QUERY_MAX_LEN} --passage_max_len ${PASSAGE_MAX_LEN} \\\\
    ${TRAIN_CHUNK_ARGS} \\\\
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \\\\
    --logging_steps 10 \\\\
    --overwrite_output_dir \\\\
    --gradient_checkpointing \\\\
    --gradient_accumulation_steps 1"

  echo ""
  echo "[CMD] \${TRAIN_CMD}"
  echo ""

  CUDA_VISIBLE_DEVICES=\$(seq -s, 0 $((NUM_GPUS-1))) \\
  torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \\
    -m tevatron.retriever.driver.train \\
    --output_dir "\${MODEL_DIR}" \\
    --model_name_or_path ${BASE_MODEL} \\
    --bf16 --pooling last --padding_side right --normalize \\
    --attn_implementation sdpa \\
    --do_train --lora \\
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \\
    --save_steps ${SAVE_STEPS} \\
    --dataset_name ${DATASET_NAME} --dataset_config ${DATASET_CONFIG} --dataset_split ${DATASET_SPLIT} \\
    --query_prefix "${QUERY_PREFIX}" --passage_prefix "${PASSAGE_PREFIX}" \\
    --temperature ${TEMPERATURE} \\
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH} --train_group_size ${TRAIN_GROUP_SIZE} \\
    --learning_rate ${LEARNING_RATE} \\
    --query_max_len ${QUERY_MAX_LEN} --passage_max_len ${PASSAGE_MAX_LEN} \\
    ${TRAIN_CHUNK_ARGS} \\
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \\
    --logging_steps 10 \\
    --overwrite_output_dir \\
    --gradient_checkpointing \\
    --gradient_accumulation_steps 1
else
  echo "=== Skipping training (checkpoint exists at \${MODEL_DIR}) ==="
fi

echo ""
echo "=== Training done (\${TRAIN_NAME}) at \$(date) ==="
SCRIPT
done

# ── Make all scripts executable ──────────────────────────────────────────────
chmod +x "${OUT_DIR}"/*.sh

# ── Summary ──────────────────────────────────────────────────────────────────
NUM_TRAIN=$(ls "${OUT_DIR}"/train_*.sh 2>/dev/null | wc -l)
echo ""
echo "Generated ${NUM_TRAIN} training scripts in ${OUT_DIR}/:"
echo ""
ls -1 "${OUT_DIR}"/train_*.sh | sed 's/^/  /'
echo ""
echo "Usage:"
echo "  bash ${OUT_DIR}/train_fixed-256.sh"
echo ""
echo "Checkpoints saved to: ${MODELS_DIR}/{name}/"
echo "Logs saved to:        ${LOGS_DIR}/train_{name}.log"
