```
deepspeed --include localhost:4,5,6,7 --master_port 60000 --module tevatron.reranker.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir model_rankllama \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 200 \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --append_eos_token \
  --learning_rate 1e-4 \
  --rerank_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir
```