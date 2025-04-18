# MAGMaR workshop fine-tuning

## Step 1: download datasets
```bash
huggingface-cli download Tevatron/multivent-corpus-train-mini --local-dir multivent-corpus-train-mini --repo-type dataset
```
then decompress all tar.gz files in the `multivent-corpus-train-mini/video` folder and `multivent-corpus-train-mini/audio.tar.gz` folders.

## Step 2: train
```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60000 --module tevatron.retriever.driver.train_mm \
  --deepspeed deepspeed/ds_zero0_config.json \
  --output_dir retriever-omni-multivent \
  --model_name_or_path Tevatron/Qwen2.5-Omni-7B-Thinker \
  --lora_name_or_path Tevatron/OmniEmbed-v0.1 \
  --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_steps 50 \
  --train_yaml ./magmar/dataset_config.yaml \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --bf16 \
  --tf32 True \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --train_group_size 4 \
  --learning_rate 1e-4 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --num_train_epochs 4 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.005 \
  --report_to wandb \
  --num_proc 10 \
  --dataloader_num_workers 4
```