#!/bin/bash

# Multi-GPU training with torchrun + gradient_checkpointing
# Usage: bash run_multi_gpu.sh

cd /home/xinyu_shi/tevatron

uv run torchrun --nproc_per_node=4 --master_port=60001 -m tevatron.retriever.driver.train \
  --output_dir retriever-qwen3-emb-ft-bs2-5e-multigpu-new \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --lora=true \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name Tevatron/scifact \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.
Query:" \
  --passage_prefix "" \
  --bf16=true \
  --pooling last \
  --padding_side left \
  --normalize=true \
  --temperature 0.01 \
  --per_device_train_batch_size 2 \
  --gradient_checkpointing=true \
  --train_group_size 8 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --overwrite_output_dir=true \
  --gradient_accumulation_steps 1 \
  --attn_implementation sdpa
