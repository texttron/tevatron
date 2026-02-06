#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export LOCAL_RANK=-1

# ---- Config ----
base_model=Qwen/Qwen3-Embedding-0.6B
ft_dir=/root/autodl-tmp/tevatron/models/fixed-64-independent
output_dir=${ft_dir}/retrieval-fixed-chunk64-independent
data_dir=${output_dir}/data

corpus_jsonl=/home/ryan/quickdev/tevatron/paper-full-run/corpus-data/corpus.jsonl
queries_jsonl=/home/ryan/quickdev/tevatron/paper-full-run/corpus-data/queries.jsonl
qrels=/home/ryan/quickdev/tevatron/paper-full-run/corpus-data/qrels.txt
prechunked_corpus_jsonl=/home/ryan/quickdev/tevatron/paper-full-run/corpus-data/corpus.jsonl

export PYTHONPATH="/home/ryan/quickdev/tevatron/src:${PYTHONPATH:-}"
mkdir -p "${data_dir}"

# ---- Encode queries ----
python -m tevatron.retriever.driver.encode \
  --output_dir="${output_dir}" \
  --model_name_or_path "${base_model}" \
  --lora_name_or_path "${ft_dir}" \
  --bf16 \
  --per_device_eval_batch_size 32 \
  --normalize \
  --pooling last \
  --padding_side right \
  --query_prefix "Instruct: Given a question, retrieve documents that answer the question.\nQuery:" \
  --query_max_len 32 \
  --dataset_name json \
  --dataset_path "${queries_jsonl}" \
  --dataset_split train \
  --encode_output_path "${output_dir}/queries.mldr-en-dev.pkl" \
  --encode_is_query

# ---- Encode corpus (independent chunks, size 64) ----
python -m tevatron.retriever.driver.encode \
  --output_dir="${output_dir}" \
  --model_name_or_path "${base_model}" \
  --lora_name_or_path "${ft_dir}" \
  --bf16 \
  --per_device_eval_batch_size 8 \
  --normalize \
  --pooling last \
  --padding_side right \
  --passage_prefix "" \
  --passage_max_len 8192 \
  --passage_chunk_size 64 \
  --passage_chunk_independent \
  --dataset_name json \
  --dataset_path "${corpus_jsonl}" \
  --dataset_split train \
  --encode_output_path "${output_dir}/corpus.mldr-en-dev.pkl"

# ---- Search with MaxSim aggregation ----
python -m tevatron.retriever.driver.search \
  --query_reps "${output_dir}/queries.mldr-en-dev.pkl" \
  --passage_reps "${output_dir}/corpus.mldr-en-dev.pkl" \
  --depth 1000 \
  --batch_size 64 \
  --save_text \
  --save_ranking_to "${output_dir}/rank.mldr-en-dev.txt" \
  --chunked \
  --corpus_path "${corpus_jsonl}" \
  --tokenizer_name "${base_model}" \
  --passage_chunk_size 64 \
  --save_distribution "${output_dir}/best_chunk_distribution.json" \
  --qrels_path "${qrels}" \
  --queries_path "${queries_jsonl}"

# ---- Convert to TREC + evaluate ----
python -m tevatron.utils.format.convert_result_to_trec \
  --input "${output_dir}/rank.mldr-en-dev.txt" \
  --output "${output_dir}/rank.mldr-en-dev.trec"

python -m pyserini.eval.trec_eval \
  -c -mrecall.100 -mndcg_cut.10 \
  "${qrels}" \
  "${output_dir}/rank.mldr-en-dev.trec"
