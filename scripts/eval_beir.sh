#!/bin/bash

: '
Example usage:
./eval_beir.sh --dataset arguana \
                    --tokenizer mistralai/Mistral-7B-v0.1 \
                    --model_name_path mistralai/Mistral-7B-v0.1 \
                    --embedding_dir beir_embedding_arguana \
                    --query_prefix "Query: " \
                    --passage_prefix "Passage: " \
                    [--lora_name_path /retriever-mistral/checkpoint-7600] \
                    [--normalize]
'

# Default values
lora_name_path=""
dataset=""
tokenizer=""
model_name_path=""
embedding_dir=""
query_prefix="Query: "
passage_prefix="Passage: "
normalize_flag=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lora_name_path) lora_name_path="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --tokenizer) tokenizer="$2"; shift 2 ;;
    --model_name_path) model_name_path="$2"; shift 2 ;;
    --embedding_dir) embedding_dir="$2"; shift 2 ;;
    --query_prefix) query_prefix="$2"; shift 2 ;;
    --passage_prefix) passage_prefix="$2"; shift 2 ;;
    --normalize) normalize_flag="--normalize"; shift ;;
    --help) 
      echo "Usage: $0 --dataset <dataset> --tokenizer <tokenizer> --model_name_path <model> --embedding_dir <directory> --query_prefix <prefix> --passage_prefix <prefix> [--lora_name_path <path>] [--normalize]"
      exit 0 ;;
    *) 
      echo "Unknown argument: $1"; 
      echo "Use --help to see the valid arguments."; 
      exit 1 ;;
  esac
done

# Check if required arguments are provided
if [ -z "$dataset" ] || [ -z "$tokenizer" ] || [ -z "$model_name_path" ] || [ -z "$embedding_dir" ]; then
  echo "Missing required arguments. Please provide all necessary options."
  echo "Usage: $0 --dataset <dataset> --tokenizer <tokenizer> --model_name_path <model> --embedding_dir <directory> --query_prefix <prefix> --passage_prefix <prefix> [--lora_name_path <path>] [--normalize]"
  exit 1
fi

# Create the embedding directory if it doesn't exist
mkdir -p $embedding_dir

# Prepare optional lora arguments
lora_args=""
if [ -n "$lora_name_path" ]; then
  lora_args="--lora --lora_name_or_path ${lora_name_path}"
fi

# Encode passages
for s in $(seq -f "%02g" 0 7); do
  CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path ${model_name_path} \
    --tokenizer_name ${tokenizer} \
    --fp16 \
    ${lora_args} \
    ${normalize_flag} \
    --pooling eos \
    --passage_prefix "${passage_prefix}" \
    --per_device_eval_batch_size 64 \
    --passage_max_len 512 \
    --dataset_name Tevatron/beir-corpus \
    --dataset_config ${dataset} \
    --encode_output_path $embedding_dir/corpus_${dataset}.${s}.pkl \
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${s}
done

# Encode queries
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${model_name_path} \
  --tokenizer_name ${tokenizer} \
  --fp16 \
  ${lora_args} \
  ${normalize_flag} \
  --pooling eos \
  --query_prefix "${query_prefix}" \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/beir \
  --dataset_config ${dataset} \
  --dataset_split "test" \
  --encode_output_path $embedding_dir/query_${dataset}.pkl \
  --query_max_len 512 \
  --encode_is_query

# Perform retrieval
set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
    --query_reps $embedding_dir/query_${dataset}.pkl \
    --passage_reps $embedding_dir/corpus_${dataset}.*.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $embedding_dir/rank.${dataset}.txt

# Convert results to TREC format
python -m tevatron.utils.format.convert_result_to_trec \
    --input $embedding_dir/rank.${dataset}.txt \
    --output $embedding_dir/rank.${dataset}.trec \
    --remove_query

# Evaluate results using pyserini
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test $embedding_dir/rank.${dataset}.trec
