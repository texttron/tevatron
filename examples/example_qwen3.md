# Qwen3-Embedding Finetuning Example

The doc provide examples to finetune [Qwen3-Embedding](https://qwenlm.github.io/blog/qwen3-embedding/).

## Finetune Qwen3-Embedding

```txt
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-qwen3-emb-ft \
  --model_name_or_path Qwen/Qwen3-Embedding-4B \ # replace by 0.6B/4B/8B
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name Tevatron/scifact \
  --query_prefix "Find a relevant scientific paper abstract to support or reject the claim. Query: " \
  --passage_prefix "" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 10 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1
```
