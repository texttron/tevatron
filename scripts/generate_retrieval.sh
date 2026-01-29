#!/bin/bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Edit these before running the generator.

BASE_MODEL="Qwen/Qwen3-Embedding-0.6B"
DEFAULT_NUM_GPUS=8
PASSAGE_MAX_LEN=8192

# Paths
EXP_ROOT="/root/autodl-tmp/tevatron"
MODELS_DIR="${EXP_ROOT}/models"        # default checkpoint dir (from generate_training.sh)
DATA_DIR="${EXP_ROOT}/data"            # must contain queries.jsonl, qrels.tsv, and corpus file(s)
LOGS_DIR="${EXP_ROOT}/logs/retrieval"

# Corpus to encode — change this to point to a different corpus
CORPUS_PATH="${DATA_DIR}/corpus.jsonl"
PRECHUNKED_CORPUS_PATH="${DATA_DIR}/prechunked-corpus.jsonl"

# Eval dataset name — used in output paths to separate results from different corpora
# e.g. "mldr-en", "scifact", "nq" — change when evaluating a different corpus
EVAL_NAME="mldr-en"

QUERY_PREFIX='Instruct: Given a question, retrieve documents that answer the question.\nQuery:'
PASSAGE_PREFIX=""
QUERY_MAX_LEN=512

PER_DEVICE_EVAL_BATCH_QUERY=16
PER_DEVICE_EVAL_BATCH_CORPUS=4

SEARCH_DEPTH=100
SEARCH_BATCH=64
CHUNK_MULTIPLIER=10

QRELS="${DATA_DIR}/qrels.tsv"

# ── Training configs (names only — must match checkpoint dirs under MODELS_DIR) ──
TRAIN_NAMES=(
  nochunk
  fixed-32
  fixed-64
  fixed-128
  fixed-256
  fixed-512
  fixed-1024
  fixed-2048
  fixed-4096
  prand-32to64
  prand-32to128
  prand-32to256
  prand-32to512
  prand-32to1024
  prand-32to2048
  prand-32to4096
)

# ── Retrieval chunk sizes (0 = no chunking) ─────────────────────────────────
RET_CHUNKS=(0 32 64 128 256 512 1024 2048 4096)

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR="scripts/retrieval"
mkdir -p "${OUT_DIR}"

echo "Generating retrieval scripts in ${OUT_DIR}/ ..."

# ── Generate one eval script per training config ─────────────────────────────
RET_CHUNKS_STR=$(printf '%s ' "${RET_CHUNKS[@]}")

for TRAIN_NAME in "${TRAIN_NAMES[@]}"; do

  cat > "${OUT_DIR}/eval_${EVAL_NAME}_${TRAIN_NAME}.sh" <<'SCRIPT_HEAD'
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

# Log a command template for sharded parallel jobs
log_shard_cmd() {
  echo ""
  echo "[CMD] (x${NUM_GPUS} shards) CUDA_VISIBLE_DEVICES={gpu} $*"
  echo ""
}
SCRIPT_HEAD

  cat >> "${OUT_DIR}/eval_${EVAL_NAME}_${TRAIN_NAME}.sh" <<SCRIPT_BODY
# ── Usage ────────────────────────────────────────────────────────────────────
# bash eval_${EVAL_NAME}_${TRAIN_NAME}.sh [checkpoint_path] [num_gpus]
#
# Args:
#   checkpoint_path  Path to LoRA checkpoint dir (default: ${MODELS_DIR}/${TRAIN_NAME})
#   num_gpus         Number of GPUs for sharded corpus encoding (default: ${DEFAULT_NUM_GPUS})

TRAIN_NAME="${TRAIN_NAME}"
EVAL_NAME="${EVAL_NAME}"

BASE_MODEL="${BASE_MODEL}"
DEFAULT_NUM_GPUS=${DEFAULT_NUM_GPUS}
NUM_GPUS="\${2:-\${DEFAULT_NUM_GPUS}}"
PASSAGE_MAX_LEN=${PASSAGE_MAX_LEN}

DEFAULT_MODEL_DIR="${MODELS_DIR}/\${TRAIN_NAME}"
MODEL_DIR="\${1:-\${DEFAULT_MODEL_DIR}}"

EXP_ROOT="${EXP_ROOT}"
DATA_DIR="${DATA_DIR}"
ENCODE_DIR="\${EXP_ROOT}/encode/\${EVAL_NAME}/\${TRAIN_NAME}"
RESULTS_DIR="\${EXP_ROOT}/results/\${EVAL_NAME}/\${TRAIN_NAME}"
LOG_DIR="${LOGS_DIR}/\${EVAL_NAME}/\${TRAIN_NAME}"
QRELS="${QRELS}"

CORPUS_PATH="${CORPUS_PATH}"
PRECHUNKED_CORPUS_PATH="${PRECHUNKED_CORPUS_PATH}"

MODEL_ARGS="--model_name_or_path \${BASE_MODEL} \\
  --bf16 --pooling last --padding_side right --normalize \\
  --attn_implementation sdpa"

QUERY_PREFIX="${QUERY_PREFIX}"

RET_CHUNKS=(${RET_CHUNKS_STR})

mkdir -p "\${ENCODE_DIR}" "\${RESULTS_DIR}" "\${LOG_DIR}"

# Helper: switch tee to a new per-chunk log file
start_log() {
  local log_file="\$1"
  exec > >(tee -a "\${log_file}") 2>&1
}

if [ ! -d "\${MODEL_DIR}" ]; then
  echo "ERROR: Checkpoint not found at \${MODEL_DIR}"
  echo "  Usage: bash \$0 [checkpoint_path] [num_gpus]"
  exit 1
fi

echo "================================================================"
echo "Evaluating: \${TRAIN_NAME} on \${EVAL_NAME}"
echo "Started:    \$(date)"
echo "Checkpoint: \${MODEL_DIR}"
echo "Corpus:     \${CORPUS_PATH}"
echo "Num GPUs:   \${NUM_GPUS}"
echo "Log dir:    \${LOG_DIR}"
echo "================================================================"

# ── Encode queries (once) ────────────────────────────────────────────────────
start_log "\${LOG_DIR}/queries.log"
if [ ! -f "\${ENCODE_DIR}/queries.pkl" ]; then
  run_cmd CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \\
    --output_dir temp \\
    \${MODEL_ARGS} \\
    --lora_name_or_path "\${MODEL_DIR}" \\
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_QUERY} \\
    --query_prefix "\${QUERY_PREFIX}" --query_max_len ${QUERY_MAX_LEN} \\
    --dataset_name json \\
    --dataset_path "\${DATA_DIR}/queries.jsonl" \\
    --encode_is_query \\
    --encode_output_path "\${ENCODE_DIR}/queries.pkl"
else
  echo "=== [\${TRAIN_NAME}] Skipping query encoding (exists) ==="
fi

# ── Loop over retrieval chunk sizes ──────────────────────────────────────────
for RET_CHUNK in "\${RET_CHUNKS[@]}"; do
  if [ "\${RET_CHUNK}" -eq 0 ]; then
    RET_NAME="ret-nochunk"
    CHUNK_ARGS=""
    SEARCH_ARGS=""
  else
    RET_NAME="ret-fixed-\${RET_CHUNK}"
    CHUNK_ARGS="--passage_chunk_size \${RET_CHUNK}"
    SEARCH_ARGS="--chunked --chunk_multiplier ${CHUNK_MULTIPLIER}"
  fi

  start_log "\${LOG_DIR}/\${RET_NAME}.log"

  CORPUS_PREFIX="\${ENCODE_DIR}/corpus-\${RET_NAME}"
  RANK_FILE="\${RESULTS_DIR}/\${RET_NAME}.txt"
  TREC_FILE="\${RESULTS_DIR}/\${RET_NAME}.trec"

  # ── Encode corpus (sharded) ────────────────────────────────────────────────
  all_shards_exist=true
  for s in \$(seq 0 \$((NUM_GPUS-1))); do
    if [ ! -f "\${CORPUS_PREFIX}.\${s}.pkl" ]; then
      all_shards_exist=false
      break
    fi
  done

  if [ "\${all_shards_exist}" = false ]; then
    log_shard_cmd python -m tevatron.retriever.driver.encode \\
      --output_dir temp \\
      \${MODEL_ARGS} \\
      --lora_name_or_path "\${MODEL_DIR}" \\
      --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_CORPUS} \\
      --passage_prefix "'${PASSAGE_PREFIX}'" --passage_max_len \${PASSAGE_MAX_LEN} \\
      \${CHUNK_ARGS} \\
      --dataset_name json \\
      --dataset_path "\${CORPUS_PATH}" \\
      --dataset_number_of_shards \${NUM_GPUS} \\
      --dataset_shard_index '{s}' \\
      --encode_output_path "\${CORPUS_PREFIX}.{s}.pkl"
    pids=()
    for s in \$(seq 0 \$((NUM_GPUS-1))); do
      CUDA_VISIBLE_DEVICES=\${s} python -m tevatron.retriever.driver.encode \\
        --output_dir temp \\
        \${MODEL_ARGS} \\
        --lora_name_or_path "\${MODEL_DIR}" \\
        --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_CORPUS} \\
        --passage_prefix "${PASSAGE_PREFIX}" --passage_max_len \${PASSAGE_MAX_LEN} \\
        \${CHUNK_ARGS} \\
        --dataset_name json \\
        --dataset_path "\${CORPUS_PATH}" \\
        --dataset_number_of_shards \${NUM_GPUS} \\
        --dataset_shard_index \${s} \\
        --encode_output_path "\${CORPUS_PREFIX}.\${s}.pkl" &
      pids+=(\$!)
    done
    for pid in "\${pids[@]}"; do
      wait "\${pid}"
    done
    echo "    All \${NUM_GPUS} shards encoded."
  else
    echo "=== [\${TRAIN_NAME}] Skipping corpus encoding (\${RET_NAME}, all shards exist) ==="
  fi

  # ── Search ─────────────────────────────────────────────────────────────────
  if [ ! -f "\${RANK_FILE}" ]; then
    run_cmd python -m tevatron.retriever.driver.search \\
      --query_reps "\${ENCODE_DIR}/queries.pkl" \\
      --passage_reps "\${CORPUS_PREFIX}.*.pkl" \\
      --depth ${SEARCH_DEPTH} --batch_size ${SEARCH_BATCH} --save_text \\
      \${SEARCH_ARGS} \\
      --save_ranking_to "\${RANK_FILE}"
  else
    echo "=== [\${TRAIN_NAME}] Skipping search (\${RET_NAME}, exists) ==="
  fi

  # ── Evaluate ───────────────────────────────────────────────────────────────
  run_cmd python -m tevatron.utils.format.convert_result_to_trec \\
    --input "\${RANK_FILE}" \\
    --output "\${TREC_FILE}" \\
    --remove_query

  run_cmd python -m pyserini.eval.trec_eval \\
    -m ndcg_cut.10 -m recall.100 \\
    "\${QRELS}" "\${TREC_FILE}"

  echo ""
done

# ── Pre-chunked corpus evaluation ────────────────────────────────────────────
RET_NAME="ret-prechunked"
start_log "\${LOG_DIR}/\${RET_NAME}.log"
CORPUS_PREFIX="\${ENCODE_DIR}/corpus-\${RET_NAME}"
RANK_FILE="\${RESULTS_DIR}/\${RET_NAME}.txt"
TREC_FILE="\${RESULTS_DIR}/\${RET_NAME}.trec"

if [ -f "\${PRECHUNKED_CORPUS_PATH}" ]; then
  all_shards_exist=true
  for s in \$(seq 0 \$((NUM_GPUS-1))); do
    if [ ! -f "\${CORPUS_PREFIX}.\${s}.pkl" ]; then
      all_shards_exist=false
      break
    fi
  done

  if [ "\${all_shards_exist}" = false ]; then
    log_shard_cmd python -m tevatron.retriever.driver.encode \\
      --output_dir temp \\
      \${MODEL_ARGS} \\
      --lora_name_or_path "\${MODEL_DIR}" \\
      --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_CORPUS} \\
      --passage_prefix "'${PASSAGE_PREFIX}'" --passage_max_len \${PASSAGE_MAX_LEN} \\
      --encode_use_pre_chunked \\
      --dataset_name json \\
      --dataset_path "\${PRECHUNKED_CORPUS_PATH}" \\
      --dataset_number_of_shards \${NUM_GPUS} \\
      --dataset_shard_index '{s}' \\
      --encode_output_path "\${CORPUS_PREFIX}.{s}.pkl"
    pids=()
    for s in \$(seq 0 \$((NUM_GPUS-1))); do
      CUDA_VISIBLE_DEVICES=\${s} python -m tevatron.retriever.driver.encode \\
        --output_dir temp \\
        \${MODEL_ARGS} \\
        --lora_name_or_path "\${MODEL_DIR}" \\
        --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_CORPUS} \\
        --passage_prefix "${PASSAGE_PREFIX}" --passage_max_len \${PASSAGE_MAX_LEN} \\
        --encode_use_pre_chunked \\
        --dataset_name json \\
        --dataset_path "\${PRECHUNKED_CORPUS_PATH}" \\
        --dataset_number_of_shards \${NUM_GPUS} \\
        --dataset_shard_index \${s} \\
        --encode_output_path "\${CORPUS_PREFIX}.\${s}.pkl" &
      pids+=(\$!)
    done
    for pid in "\${pids[@]}"; do
      wait "\${pid}"
    done
    echo "    All \${NUM_GPUS} shards encoded."
  else
    echo "=== [\${TRAIN_NAME}] Skipping pre-chunked corpus encoding (all shards exist) ==="
  fi

  if [ ! -f "\${RANK_FILE}" ]; then
    run_cmd python -m tevatron.retriever.driver.search \\
      --query_reps "\${ENCODE_DIR}/queries.pkl" \\
      --passage_reps "\${CORPUS_PREFIX}.*.pkl" \\
      --depth ${SEARCH_DEPTH} --batch_size ${SEARCH_BATCH} --save_text \\
      --chunked --chunk_multiplier ${CHUNK_MULTIPLIER} \\
      --save_ranking_to "\${RANK_FILE}"
  else
    echo "=== [\${TRAIN_NAME}] Skipping search (\${RET_NAME}, exists) ==="
  fi

  run_cmd python -m tevatron.utils.format.convert_result_to_trec \\
    --input "\${RANK_FILE}" \\
    --output "\${TREC_FILE}" \\
    --remove_query

  run_cmd python -m pyserini.eval.trec_eval \\
    -m ndcg_cut.10 -m recall.100 \\
    "\${QRELS}" "\${TREC_FILE}"

  echo ""
else
  echo "=== [\${TRAIN_NAME}] Skipping pre-chunked eval (\${PRECHUNKED_CORPUS_PATH} not found) ==="
fi

echo ""
echo "=== All retrieval configs evaluated for \${TRAIN_NAME} on \${EVAL_NAME} at \$(date) ==="
SCRIPT_BODY
done

# ── Make all scripts executable ──────────────────────────────────────────────
chmod +x "${OUT_DIR}"/*.sh

# ── Summary ──────────────────────────────────────────────────────────────────
NUM_EVAL=$(ls "${OUT_DIR}"/eval_*.sh 2>/dev/null | wc -l)
echo ""
echo "Generated ${NUM_EVAL} retrieval scripts in ${OUT_DIR}/:"
echo ""
ls -1 "${OUT_DIR}"/eval_*.sh | sed 's/^/  /'
echo ""
echo "Usage:"
echo "  # With default MLDR checkpoints and ${DEFAULT_NUM_GPUS} GPUs:"
echo "  bash ${OUT_DIR}/eval_${EVAL_NAME}_fixed-256.sh"
echo ""
echo "  # With custom checkpoint path:"
echo "  bash ${OUT_DIR}/eval_${EVAL_NAME}_fixed-256.sh /path/to/checkpoint"
echo ""
echo "  # With custom checkpoint and 4 GPUs:"
echo "  bash ${OUT_DIR}/eval_${EVAL_NAME}_fixed-256.sh /path/to/checkpoint 4"
echo ""
echo "Logs saved to: ${LOGS_DIR}/eval_{eval_name}_{train_name}.log"
echo ""
echo "To evaluate a different corpus, edit CORPUS_PATH, PRECHUNKED_CORPUS_PATH,"
echo "EVAL_NAME, QUERY_PREFIX, and QRELS in this generator, then re-run."
