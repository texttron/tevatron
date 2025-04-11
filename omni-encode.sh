model=Tevatron/Qwen2.5-Omni-7B-Thinker
dataset_config=/mnt/users/x978zhan/paper-backup/multivent/tevatron/dataset_config.yaml

CUDA_VISIBLE_DEVICES=7 python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path $model --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --passage_prefix "" \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name Tevatron/msrvtt-corpus \
  --assets_path data/Tevatron/msrvtt-corpus/video \
  --dataset_split train \
  --encode_output_path test/corpus.0.pkl \
  --dataset_number_of_shards 1 \
  --dataset_shard_index 0 \
  --dataloader_num_workers 8 \