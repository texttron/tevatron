```
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir model_repllama \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name json \
  --dataset_path /store2/scratch/x93ma/main-tevatron/tevatron/experiments/tevatron-train-repllama.jsonl \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir
```


```
mkdir /mnt/users/x93ma/tevatron-v2-data/repllama-mistral-7b-4gpu-32bs-4acc-16gs-1e4lr-1epoch-embeddings
for s in 0 1 2 3
do
gpuid=$((s % 4))
CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_name_or_path model_repllama \
  --lora \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} \
  --encode_output_path /mnt/users/x93ma/tevatron-v2-data/repllama-mistral-7b-4gpu-32bs-4acc-16gs-1e4lr-1epoch-embeddings/corpus.${s}.pkl &
done
```