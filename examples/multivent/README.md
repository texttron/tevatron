# MAGMaR workshop fine-tuning


## Step 1: download datasets

Due to license issue, we are not able to share the processed dataset. Please contact the original author of multi-vent to get the raw data.

## Step 2: train
```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 60000 --module tevatron.retriever.driver.train_mm \
  --deepspeed deepspeed/ds_zero0_config.json \
  --output_dir retriever-omni-multivent \
  --model_name_or_path Tevatron/Qwen2.5-Omni-7B-Thinker \
  --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --lora_name_or_path Tevatron/OmniEmbed-v0.1 \
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
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.005 \
  --report_to wandb \
  --num_proc 10 \
  --dataloader_num_workers 4
```

## Step 3: evaluate
We release our trained checkpoint here: [Tevatron/OmniEmbed-v0.1-multivent](https://huggingface.co/Tevatron/OmniEmbed-v0.1-multivent)
```bash
CKPT=Tevatron/OmniEmbed-v0.1-multivent
# encode query
mkdir -p multivent-embedding/${CKPT}
CUDA_VISIBLE_DEVICES=1 python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Tevatron/Qwen2.5-Omni-7B-Thinker \
  --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --lora_name_or_path ${CKPT} \
  --bf16 \
  --per_device_eval_batch_size 8 \
  --normalize \
  --pooling last  \
  --query_prefix "Query: " \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name Tevatron/multivent-query-test \
  --dataset_split test \
  --encode_output_path multivent-embedding/${CKPT}/query-test.pkl \
  --dataloader_num_workers 8 \
  --encode_is_query
  
# dowanload the corpus and decompress
huggingface-cli download Tevatron/multivent-corpus-test --local-dir multivent-corpus-test --repo-type dataset
then decompress all tar.gz files in the `multivent-corpus-test/multi-media-data/video` folder and `multivent-corpus-test/multi-media-data/audio` into the same folders.

  
for gpu in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=$gpu python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Tevatron/Qwen2.5-Omni-7B-Thinker \
  --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --lora_name_or_path ${CKPT} \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --passage_prefix "" \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name Tevatron/multivent-corpus-test \
  --assets_path magmar/multivent-corpus-test \
  --dataset_split train \
  --encode_output_path multivent-embedding/${CKPT}/corpus.$gpu.pkl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index $gpu \
  --dataloader_num_workers 4 &
done

wait

```

Generate submission file
```bash
# generate trec run file
mkdir -p multivent-results/${CKPT}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.search \
    --query_reps multivent-embedding/${CKPT}/query-test.pkl \
    --passage_reps multivent-embedding/${CKPT}/'corpus.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to multivent-results/${CKPT}/rank-test.txt

python -m tevatron.utils.format.convert_result_to_trec --input multivent-results/${CKPT}/rank-test.txt \
                                                       --output multivent-results/${CKPT}/rank-test.trec

# convert trec to submit file                                                       
python convert_trec_to_submit.py --input multivent-results/${CKPT}/rank-test.trec \
--output multivent-results/${CKPT}/rank-test.json

```
