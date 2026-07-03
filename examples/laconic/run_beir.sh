#!/bin/bash
# Reproduce LACONIC (decoder-LM SPLADE) on a BEIR dataset:
#   encode corpus + queries -> seismic search -> BEIR score.
#
# Usage:
#   MODEL=utahnlp/laconic-1b TOKENIZER=unsloth/Llama-3.2-1B DATASET=scifact ./run_beir.sh
#
# Reproduces scifact NDCG@10 ~= 0.752 (paper 0.756). See README for the two
# encoding details (no BOS, query:/passage: prefixes) that this script sets and
# why they are critical.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NVTE_FUSED_ATTN=0            # this cluster: dual CUDA runtime conflict

MODEL="${MODEL:-utahnlp/laconic-1b}"
TOKENIZER="${TOKENIZER:-unsloth/Llama-3.2-1B}"   # checkpoint ships no tokenizer; use Llama-3 base
DATASET="${DATASET:-scifact}"
OUT="${OUT:-/tmp/laconic_repro/$DATASET}"
DEPTH="${DEPTH:-200}"
TOPK="${TOPK:-512}"
MAXLEN="${MAXLEN:-256}"
NSHARD="${NSHARD:-1}"               # raise for big corpora (one GPU shard each)

PY="${PY:-.venv/bin/python}"               # transformers 5.x + seismic
PY_EVAL="${PY_EVAL:-.venv-eval/bin/python}" # beir + pytrec_eval

mkdir -p "$OUT/corpus" "$OUT/query"

# Flags common to corpus + query encode. The 3 critical ones: causal model,
# no BOS, E5 prefixes (set per-side below).
COMMON=(--output_dir temp --model_name_or_path "$MODEL" --tokenizer_name "$TOKENIZER"
        --bf16 --attn_implementation sdpa
        --splade_model_type causal --is_bidirectional --pooling_strategy max
        --add_special_tokens False --splade_topk "$TOPK" --splade_weight_format float)

echo "================ LACONIC | $DATASET | $MODEL ================"

# ---- corpus encode (one GPU shard each) ----
if ! ls "$OUT"/corpus/split*.jsonl >/dev/null 2>&1; then
    echo "[encode] corpus ($NSHARD shard(s))"
    for s in $(seq 0 $((NSHARD-1))); do
        CUDA_VISIBLE_DEVICES=$s "$PY" -m tevatron.retriever.driver.encode_splade \
            "${COMMON[@]}" --passage_max_len "$MAXLEN" --per_device_eval_batch_size 32 \
            --passage_prefix "passage: " \
            --dataset_name Tevatron/beir-corpus --dataset_config "$DATASET" --dataset_split train \
            --dataset_number_of_shards "$NSHARD" --dataset_shard_index "$s" \
            --encode_output_path "$OUT/corpus/split$(printf %02d $s).jsonl" &
    done
    wait
else
    echo "[skip] corpus already encoded"
fi

# ---- query encode ----
# Queries come from a BEIR queries.jsonl ({query_id, query}); materialize via
# tevatron.utils.prepare_queries if you don't already have one.
QJ="${QUERY_JSONL:-$OUT/query/queries.jsonl}"
if [ ! -f "$QJ" ]; then
    echo "[materialize] queries.jsonl for $DATASET"
    "$PY" -m tevatron.utils.prepare_queries --dataset "$DATASET" --output "$QJ"
fi
if [ ! -f "$OUT/query/test.tsv" ]; then
    echo "[encode] queries"
    CUDA_VISIBLE_DEVICES=0 "$PY" -m tevatron.retriever.driver.encode_splade \
        "${COMMON[@]}" --query_max_len "$MAXLEN" --per_device_eval_batch_size 64 \
        --encode_is_query --query_prefix "query: " \
        --dataset_name json --dataset_path "$QJ" --dataset_split train \
        --encode_output_path "$OUT/query/test.tsv"
else
    echo "[skip] queries already encoded"
fi

# ---- seismic search ----
echo "[search] seismic (depth $DEPTH)"
"$PY" -m tevatron.retriever.driver.search_splade \
    --corpus_files "$OUT"/corpus/split*.jsonl \
    --query_file "$OUT/query/test.tsv" \
    --output_path "$OUT/rank.text" --depth "$DEPTH"

# ---- BEIR score ----
echo "[eval] BEIR"
OPENAI_API_KEY=dummy "$PY_EVAL" -m tevatron.eval.metrics \
    --dataset "$DATASET" --ranklist "$OUT/rank.text" --output "$OUT/metrics.json"

echo "================ $DATASET metrics ================"
grep -iE 'NDCG@10|Recall@100' "$OUT/metrics.json"
