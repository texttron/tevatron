#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export HF_HOME=/u3/n3thakur/projects/cache
export DATASETS_HF_HOME=/u3/n3thakur/projects/cache
export WANDB_PROJECT=llm-filtering
today=$(date +"%Y-%m-%d")

deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train_distil \
  --deepspeed ds_config.json \
  --output_dir models/e5-base-kl-divergence-default-680K-bge-reranker-v2-gemma-test \
  --run_name e5-base-kl-divergence-default-680K-bge-reranker-v2-gemma-16x16x4 \
  --model_name_or_path intfloat/e5-base-unsupervised \
  --cache_dir /u3/n3thakur/projects/cache \
  --dataset_cache_dir /u3/n3thakur/projects/cache \
  --save_steps 5000 \
  --dataset_name rlhn/default-680K-bge-reranker-v2-gemma \
  --attn_implementation eager \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling mean \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 2e-5 \
  --query_max_len 350 \
  --passage_max_len 350 \
  --num_train_epochs 4 \
  --logging_steps 5 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4
