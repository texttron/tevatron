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

## Step 3: evaluate
```bash
CKPT=Tevatron/OmniEmbed-v0.1-multivent
# encode query
mkdir -p multivent-embedding/${CKPT}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode_mm  \
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

encode_num_shard=200
for i in $(seq 0 $((encode_num_shard-1)))
do
SHARD_INDEX=$i
NUM_SHARED=$encode_num_shard
if [ ! -f multivent-embedding/${CKPT}/corpus.${i}.pkl ]; then
    echo "Missing multivent-embedding/${CKPT}/corpus.${i}.pkl"
#    sbatch --job-name "encode-multivent-$i" --output "logs/multivent/print-encode-$i.txt" --error "logs/multivent/error-encode-$i.txt" ./encode_multivent_omni.sh "$CKPT" "$NUM_SHARED" "$SHARD_INDEX"
fi
done
  
for gpu in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=$gpu python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Tevatron/Qwen2.5-Omni-7B-Thinker \
  --tokenizer_name Qwen/Qwen2.5-Omni-7B \
  --lora_name_or_path Tevatron/OmniEmbed-v0.1-multivent \
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
  --dataset_number_of_shards 2 \
  --dataset_shard_index $gpu \
  --dataloader_num_workers 4 &
done

wait

```

Eval
```bash
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

python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 magmar/qrels_multivent-sample-train-test.txt multivent-results/${CKPT}/rank-sample-train-test.trec

```