# Multimodal Retrieval

## Train
```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7,8 --master_port 60000 --module tevatron.retriever.driver.train_mm \
  --deepspeed deepspeed/ds_zero0_config.json \
  --output_dir retriever-qwen25vl-bge-pixmo-colpali-wiki \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_steps 500 \
  --train_yaml dataset_config.yaml \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --bf16 \
  --tf32 True \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 4 \
  --learning_rate 1e-4 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.005 \
  --report_to wandb \
  --dataloader_num_workers 4
```

## Inference and evaluation

### BEIR (textual modality)

#### Query Encode
```bash

CKPT=retriever-qwen25vl-bge-pixmo-colpali-wiki
DATASET=scifact

mkdir -p beir_embedding/${CKPT}/${DATASET}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --append_eos_token \
  --query_max_len 512 \
  --dataset_name Tevatron/beir \
  --dataset_config ${DATASET} \
  --dataset_split test \
  --encode_output_path beir_embedding/${CKPT}/${DATASET}/queries.pkl \
  --encode_is_query
```

#### Document Encode
```bash
for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --passage_prefix "" \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config ${DATASET} \
  --dataset_split train \
  --encode_output_path beir_embedding/${CKPT}/${DATASET}/corpus.${s}.pkl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} &
done
wait
```



#### Search
```bash
mkdir -p beir_results/retriever-qwen25vl-bge-pixmo-colpali-wiki/scifact
python -m tevatron.retriever.driver.search \
    --query_reps beir_embedding/${CKPT}/${DATASET}/queries.pkl \
    --passage_reps beir_embedding/${CKPT}/${DATASET}/'corpus.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to beir_results/${CKPT}/${DATASET}/rank.scifact.txt
```

#### Evaluate
```bash
python -m tevatron.utils.format.convert_result_to_trec \
--input beir_results/${CKPT}/${DATASET}/rank.scifact.txt \
--output beir_results/${CKPT}/${DATASET}/rank.scifact.trec \
--remove_query

python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test \
beir_results/${CKPT}/${DATASET}/rank.scifact.trec
```

### MIRACL (Multi-Lingual, Textual Modality)
#### Query Encode
```bash

CKPT=retriever-qwen25vl-bge-pixmo-colpali-wiki
DATASET=ar

mkdir -p miracl_embedding/${CKPT}/${DATASET}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --append_eos_token \
  --query_max_len 512 \
  --dataset_name miracl/miracl \
  --dataset_config $DATASET \
  --dataset_split test \
  --encode_output_path miracl_embedding/${CKPT}/${DATASET}/queries.pkl \
  --encode_is_query
```

#### Document Encode
```bash
for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode_mm  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last  \
  --passage_prefix "" \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name miracl/miracl-corpus \
  --dataset_config ${DATASET} \
  --dataset_split train \
  --encode_output_path miracl_embedding/${CKPT}/${DATASET}/corpus.${s}.pkl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} &
done
wait
```



#### Search
```bash
mkdir -p miracl_results/retriever-qwen25vl-bge-pixmo-colpali-wiki/$DATASET
python -m tevatron.retriever.driver.search \
    --query_reps miracl_embedding/${CKPT}/${DATASET}/queries.pkl \
    --passage_reps miracl_embedding/${CKPT}/${DATASET}/'corpus.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.txt
```

#### Evaluate
```bash
python -m tevatron.utils.format.convert_result_to_trec \
--input beir_results/${CKPT}/${DATASET}/rank.${DATASET}.txt \
--output beir_results/${CKPT}/${DATASET}/rank.${DATASET}.trec 

python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 miracl-v1.0-${DATASET}-dev \
miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.trec

### VIDORE Document screenshot retrieval (Cross modality)
```bash
CUDA_VISIBLE_DEVICES=0 python eval_vidore.py \
--model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
--lora_name_or_path ${CKPT} \
--batch_size 4 \
--pooling last \
--normalize \
--query_prefix "Query: "
```