# Document Screenshot Embedding

## Train
```bash
deepspeed --include localhost:0,1 --master_port 60000 train.py \
  --deepspeed ds_zero3_config.json \
  --output_dir dse-nq-retriever \
  --model_name_or_path Tevatron/Phi-3-vision-128k-instruct-clone \
  --lora \
  --save_steps 100 \
  --dataset_name Tevatron/wiki-ss-nq \
  --corpus_name Tevatron/wiki-ss-corpus \
  --bf16 \
  --pooling eos \
  --tf32 True \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 2 \
  --learning_rate 1e-4 \
  --query_max_len 128 \
  --passage_max_len 4096 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 8
```

## Encode
### Document Encode
```bash
for shard in 0 1
do
CUDA_VISIBLE_DEVICES=$shard python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path Tevatron/Phi-3-vision-128k-instruct-clone \
  --lora_name_or_path dse-nq-retriever \
  --lora \
  --bf16 \
  --tf32 True \
  --pooling eos \
  --normalize \
  --per_device_eval_batch_size 8 \
  --query_max_len 128 \
  --passage_max_len 4096 \
  --dataset_name Tevatron/wiki-ss-corpus \
  --dataset_number_of_shards 2 \
  --dataset_shard_index $shard \
  --encode_output_path corpus.shard.$shard.pkl &
done
wait
```

### Query Encode
```bash

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path Tevatron/Phi-3-vision-128k-instruct-clone \
  --lora_name_or_path dse-nq-retriever \
  --lora \
  --bf16 \
  --tf32 True \
  --pooling eos \
  --normalize \
  --per_device_eval_batch_size 16 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --dataset_name Tevatron/wiki-ss-nq \
  --dataset_split test \
  --encode_output_path query.nq.pkl \
  --encode_is_query
```

## Search
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.search \
    --query_reps query.nq.pkl \
    --passage_reps 'corpus.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to run.dse-nq.txt
```

## Evaluate
```bash
python eval_retrieval.py --run_file run.dse.nq.txt --k 1
```


## Train on Docmatix-IR for general purpose Document Visual Embedding
```bash
deepspeed --include localhost:0,1 --master_port 60000 train.py \
  --deepspeed ds_zero3_config.json \
  --output_dir dse-docmatix \
  --model_name_or_path Tevatron/Phi-3-vision-128k-instruct-clone \
  --lora \
  --save_steps 100 \
  --dataset_name Tevatron/docmatix-ir \
  --corpus_name HuggingFaceM4/Docmatix \
  --bf16 \
  --pooling eos \
  --tf32 True \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 2 \
  --learning_rate 1e-4 \
  --query_max_len 128 \
  --passage_max_len 4096 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 8
```


## Citation
```
@article{dse,
      title={Unifying Multimodal Retrieval via Document Screenshot Embedding}, 
      author={Xueguang Ma and Sheng-Chieh Lin and Minghan Li and Wenhu Chen and Jimmy Lin},
      year={2024},
      journal={arXiv:2406.11251}
}
```

