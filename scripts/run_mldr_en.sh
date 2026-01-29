#!/bin/bash
set -euo pipefail

# ── Experiment config ────────────────────────────────────────────────────────
TRAIN_NAME="prand-32to512"          # descriptive name for this training run
RET_CHUNK_SIZE=128                  # retrieval-time chunk size (0 = no chunking)

BASE_MODEL=Qwen/Qwen3-Embedding-0.6B
NUM_GPUS=8
MASTER_PORT=60001

# Paths
EXP_ROOT=/root/autodl-tmp/tevatron
DATA_DIR="${EXP_ROOT}/data"
MODEL_DIR="${EXP_ROOT}/models/${TRAIN_NAME}"
ENCODE_DIR="${EXP_ROOT}/encode/${TRAIN_NAME}"
RESULTS_DIR="${EXP_ROOT}/results/${TRAIN_NAME}"

# Retrieval naming
if [ "${RET_CHUNK_SIZE}" -eq 0 ]; then
  RET_NAME="ret-nochunk"
else
  RET_NAME="ret-fixed-${RET_CHUNK_SIZE}"
fi

mkdir -p "${MODEL_DIR}" "${ENCODE_DIR}" "${RESULTS_DIR}"

# ── Shared args ──────────────────────────────────────────────────────────────
QUERY_PREFIX="Instruct: Given a question, retrieve documents that answer the question.\nQuery:"
PASSAGE_MAX_LEN=8192

MODEL_ARGS="--model_name_or_path ${BASE_MODEL} \
  --bf16 --pooling last --padding_side right --normalize \
  --attn_implementation sdpa"

# ── Step 1: Train ────────────────────────────────────────────────────────────
if [ ! -f "${MODEL_DIR}/adapter_config.json" ]; then
  echo "=== Training (${TRAIN_NAME}) ==="
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
  torchrun --nproc_per_node ${NUM_GPUS} --master_port ${MASTER_PORT} \
    -m tevatron.retriever.driver.train \
    --output_dir "${MODEL_DIR}" \
    ${MODEL_ARGS} \
    --do_train --lora \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps 5000 \
    --dataset_name Shitao/MLDR --dataset_config en --dataset_split train \
    --query_prefix "${QUERY_PREFIX}" --passage_prefix "" \
    --temperature 0.03 \
    --per_device_train_batch_size 2 --train_group_size 4 \
    --learning_rate 1e-4 \
    --query_max_len 512 --passage_max_len ${PASSAGE_MAX_LEN} \
    --passage_chunk_size_range 32,512 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1
else
  echo "=== Skipping training (checkpoint exists) ==="
fi

# ── Step 2: Encode queries ───────────────────────────────────────────────────
if [ ! -f "${ENCODE_DIR}/queries.pkl" ]; then
  echo "=== Encoding queries ==="
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    ${MODEL_ARGS} \
    --lora_name_or_path "${MODEL_DIR}" \
    --per_device_eval_batch_size 16 \
    --query_prefix "${QUERY_PREFIX}" --query_max_len 512 \
    --dataset_name json \
    --dataset_path "${DATA_DIR}/queries.jsonl" \
    --encode_is_query \
    --encode_output_path "${ENCODE_DIR}/queries.pkl"
else
  echo "=== Skipping query encoding (exists) ==="
fi

# ── Step 3: Encode corpus (sharded across GPUs) ─────────────────────────────
CORPUS_PREFIX="${ENCODE_DIR}/corpus-${RET_NAME}"

# Check if ALL shards exist
all_shards_exist=true
for s in $(seq 0 $((NUM_GPUS-1))); do
  if [ ! -f "${CORPUS_PREFIX}.${s}.pkl" ]; then
    all_shards_exist=false
    break
  fi
done

if [ "${all_shards_exist}" = false ]; then
  echo "=== Encoding corpus (${NUM_GPUS} shards, ${RET_NAME}) ==="

  CHUNK_ARGS=""
  if [ "${RET_CHUNK_SIZE}" -gt 0 ]; then
    CHUNK_ARGS="--passage_chunk_size ${RET_CHUNK_SIZE}"
  fi

  pids=()
  for s in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=${s} python -m tevatron.retriever.driver.encode \
      --output_dir temp \
      ${MODEL_ARGS} \
      --lora_name_or_path "${MODEL_DIR}" \
      --per_device_eval_batch_size 4 \
      --passage_prefix "" --passage_max_len ${PASSAGE_MAX_LEN} \
      ${CHUNK_ARGS} \
      --dataset_name json \
      --dataset_path "${DATA_DIR}/corpus.jsonl" \
      --dataset_number_of_shards ${NUM_GPUS} \
      --dataset_shard_index ${s} \
      --encode_output_path "${CORPUS_PREFIX}.${s}.pkl" &
    pids+=($!)
  done

  # Wait for each shard individually so failures are caught by set -e
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "All ${NUM_GPUS} shards encoded."
else
  echo "=== Skipping corpus encoding (all shards exist) ==="
fi

# ── Step 4: Search ───────────────────────────────────────────────────────────
RANK_FILE="${RESULTS_DIR}/${RET_NAME}.txt"

if [ ! -f "${RANK_FILE}" ]; then
  echo "=== Searching (${RET_NAME}) ==="

  SEARCH_ARGS=""
  if [ "${RET_CHUNK_SIZE}" -gt 0 ]; then
    SEARCH_ARGS="--chunked --chunk_multiplier 10"
  fi

  python -m tevatron.retriever.driver.search \
    --query_reps "${ENCODE_DIR}/queries.pkl" \
    --passage_reps "${CORPUS_PREFIX}.*.pkl" \
    --depth 100 --batch_size 64 --save_text \
    ${SEARCH_ARGS} \
    --save_ranking_to "${RANK_FILE}"
else
  echo "=== Skipping search (exists) ==="
fi

# ── Step 5: Evaluate ─────────────────────────────────────────────────────────
TREC_FILE="${RESULTS_DIR}/${RET_NAME}.trec"

echo "=== Converting to TREC format ==="
python -m tevatron.utils.format.convert_result_to_trec \
  --input "${RANK_FILE}" \
  --output "${TREC_FILE}" \
  --remove_query

echo ""
echo "=== Evaluation (${TRAIN_NAME} / ${RET_NAME}) ==="
python -m pyserini.eval.trec_eval \
  -m ndcg_cut.10 -m recall.100 \
  "${DATA_DIR}/qrels.tsv" \
  "${TREC_FILE}"

echo ""
echo "Done. Ranking: ${RANK_FILE}  TREC: ${TREC_FILE}"
