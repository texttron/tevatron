#!/bin/bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
# Edit these before running the generator.

BASE_MODEL="Qwen/Qwen3-Embedding-0.6B"
NUM_GPUS=8
MASTER_PORT=60001
PASSAGE_MAX_LEN=8192

EXP_ROOT="/root/autodl-tmp/tevatron"
DATA_DIR="${EXP_ROOT}/data"            # must contain queries.jsonl, corpus.jsonl, prechunked-corpus.jsonl, qrels.tsv

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

PER_DEVICE_EVAL_BATCH_QUERY=16
PER_DEVICE_EVAL_BATCH_CORPUS=4

SEARCH_DEPTH=100
SEARCH_BATCH=64
CHUNK_MULTIPLIER=10

QRELS="${DATA_DIR}/qrels.tsv"

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

# ── Retrieval chunk sizes (0 = no chunking) ─────────────────────────────────
RET_CHUNKS=(0 32 64 128 256 512 1024 2048 4096)

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR="scripts/experiments"
mkdir -p "${OUT_DIR}"

echo "Generating experiment scripts in ${OUT_DIR}/ ..."

# ── Generate one train script per training config ────────────────────────────
for entry in "${TRAIN_CONFIGS[@]}"; do
  TRAIN_NAME="${entry%%|*}"
  TRAIN_CHUNK_ARGS="${entry#*|}"

  cat > "${OUT_DIR}/train_${TRAIN_NAME}.sh" <<SCRIPT
#!/bin/bash
set -euo pipefail

TRAIN_NAME="${TRAIN_NAME}"
MODEL_DIR="${EXP_ROOT}/models/\${TRAIN_NAME}"
mkdir -p "\${MODEL_DIR}"

if [ ! -f "\${MODEL_DIR}/adapter_config.json" ]; then
  echo "=== Training (\${TRAIN_NAME}) ==="
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

echo "=== Training done (\${TRAIN_NAME}) ==="
SCRIPT
done

# ── Generate one eval script per training config ─────────────────────────────
# Each eval script encodes queries once, then loops over all retrieval chunk sizes.
for entry in "${TRAIN_CONFIGS[@]}"; do
  TRAIN_NAME="${entry%%|*}"

  # Build the RET_CHUNKS array literal for embedding in the script
  RET_CHUNKS_STR=$(printf '%s ' "${RET_CHUNKS[@]}")

  cat > "${OUT_DIR}/eval_${TRAIN_NAME}.sh" <<'SCRIPT_HEAD'
#!/bin/bash
set -euo pipefail
SCRIPT_HEAD

  cat >> "${OUT_DIR}/eval_${TRAIN_NAME}.sh" <<SCRIPT_BODY
TRAIN_NAME="${TRAIN_NAME}"

BASE_MODEL="${BASE_MODEL}"
NUM_GPUS=${NUM_GPUS}
PASSAGE_MAX_LEN=${PASSAGE_MAX_LEN}

EXP_ROOT="${EXP_ROOT}"
DATA_DIR="${DATA_DIR}"
MODEL_DIR="\${EXP_ROOT}/models/\${TRAIN_NAME}"
ENCODE_DIR="\${EXP_ROOT}/encode/\${TRAIN_NAME}"
RESULTS_DIR="\${EXP_ROOT}/results/\${TRAIN_NAME}"
QRELS="${QRELS}"

MODEL_ARGS="--model_name_or_path \${BASE_MODEL} \\
  --bf16 --pooling last --padding_side right --normalize \\
  --attn_implementation sdpa"

QUERY_PREFIX="${QUERY_PREFIX}"

RET_CHUNKS=(${RET_CHUNKS_STR})

mkdir -p "\${ENCODE_DIR}" "\${RESULTS_DIR}"

# ── Encode queries (once) ────────────────────────────────────────────────────
if [ ! -f "\${ENCODE_DIR}/queries.pkl" ]; then
  echo "=== [\${TRAIN_NAME}] Encoding queries ==="
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \\
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
    echo "=== [\${TRAIN_NAME}] Encoding corpus (\${NUM_GPUS} shards, \${RET_NAME}) ==="
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
        --dataset_path "\${DATA_DIR}/corpus.jsonl" \\
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
    echo "=== [\${TRAIN_NAME}] Searching (\${RET_NAME}) ==="
    python -m tevatron.retriever.driver.search \\
      --query_reps "\${ENCODE_DIR}/queries.pkl" \\
      --passage_reps "\${CORPUS_PREFIX}.*.pkl" \\
      --depth ${SEARCH_DEPTH} --batch_size ${SEARCH_BATCH} --save_text \\
      \${SEARCH_ARGS} \\
      --save_ranking_to "\${RANK_FILE}"
  else
    echo "=== [\${TRAIN_NAME}] Skipping search (\${RET_NAME}, exists) ==="
  fi

  # ── Evaluate ───────────────────────────────────────────────────────────────
  echo "=== [\${TRAIN_NAME}] Evaluating (\${RET_NAME}) ==="
  python -m tevatron.utils.format.convert_result_to_trec \\
    --input "\${RANK_FILE}" \\
    --output "\${TREC_FILE}" \\
    --remove_query

  python -m pyserini.eval.trec_eval \\
    -m ndcg_cut.10 -m recall.100 \\
    "\${QRELS}" "\${TREC_FILE}"

  echo ""
done

# ── Pre-chunked corpus evaluation ────────────────────────────────────────────
RET_NAME="ret-prechunked"
CORPUS_PREFIX="\${ENCODE_DIR}/corpus-\${RET_NAME}"
RANK_FILE="\${RESULTS_DIR}/\${RET_NAME}.txt"
TREC_FILE="\${RESULTS_DIR}/\${RET_NAME}.trec"
PRECHUNKED_CORPUS="\${DATA_DIR}/prechunked-corpus.jsonl"

if [ -f "\${PRECHUNKED_CORPUS}" ]; then
  # ── Encode pre-chunked corpus (sharded) ──────────────────────────────────
  all_shards_exist=true
  for s in \$(seq 0 \$((NUM_GPUS-1))); do
    if [ ! -f "\${CORPUS_PREFIX}.\${s}.pkl" ]; then
      all_shards_exist=false
      break
    fi
  done

  if [ "\${all_shards_exist}" = false ]; then
    echo "=== [\${TRAIN_NAME}] Encoding pre-chunked corpus (\${NUM_GPUS} shards) ==="
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
        --dataset_path "\${PRECHUNKED_CORPUS}" \\
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

  # ── Search ───────────────────────────────────────────────────────────────
  if [ ! -f "\${RANK_FILE}" ]; then
    echo "=== [\${TRAIN_NAME}] Searching (\${RET_NAME}) ==="
    python -m tevatron.retriever.driver.search \\
      --query_reps "\${ENCODE_DIR}/queries.pkl" \\
      --passage_reps "\${CORPUS_PREFIX}.*.pkl" \\
      --depth ${SEARCH_DEPTH} --batch_size ${SEARCH_BATCH} --save_text \\
      --chunked --chunk_multiplier ${CHUNK_MULTIPLIER} \\
      --save_ranking_to "\${RANK_FILE}"
  else
    echo "=== [\${TRAIN_NAME}] Skipping search (\${RET_NAME}, exists) ==="
  fi

  # ── Evaluate ─────────────────────────────────────────────────────────────
  echo "=== [\${TRAIN_NAME}] Evaluating (\${RET_NAME}) ==="
  python -m tevatron.utils.format.convert_result_to_trec \\
    --input "\${RANK_FILE}" \\
    --output "\${TREC_FILE}" \\
    --remove_query

  python -m pyserini.eval.trec_eval \\
    -m ndcg_cut.10 -m recall.100 \\
    "\${QRELS}" "\${TREC_FILE}"

  echo ""
else
  echo "=== [\${TRAIN_NAME}] Skipping pre-chunked eval (${DATA_DIR}/prechunked-corpus.jsonl not found) ==="
fi

echo "=== All retrieval configs evaluated for \${TRAIN_NAME} ==="
SCRIPT_BODY
done

# ── Make all scripts executable ──────────────────────────────────────────────
chmod +x "${OUT_DIR}"/*.sh

# ── Summary ──────────────────────────────────────────────────────────────────
NUM_TRAIN=$(ls "${OUT_DIR}"/train_*.sh 2>/dev/null | wc -l)
NUM_EVAL=$(ls "${OUT_DIR}"/eval_*.sh 2>/dev/null | wc -l)
echo ""
echo "Generated ${NUM_TRAIN} train scripts and ${NUM_EVAL} eval scripts in ${OUT_DIR}/:"
echo ""
echo "Train scripts:"
ls -1 "${OUT_DIR}"/train_*.sh | sed 's/^/  /'
echo ""
echo "Eval scripts:"
ls -1 "${OUT_DIR}"/eval_*.sh | sed 's/^/  /'
echo ""
echo "Usage:"
echo "  # On training machine:"
echo "  bash ${OUT_DIR}/train_fixed-256.sh"
echo ""
echo "  # On eval machine (after copying exp/models/):"
echo "  bash ${OUT_DIR}/eval_fixed-256.sh"
