## Train Retriever using Teacher distillation (using Reranker as Teacher)

1. Pre-compute the reranker scores for all training query & passage pairs using your favourite reranker as teacher, e.g., BAAI/bge-reranker-v2-gemma.

2. Construct a Training dataset in the default Tevatron format and add a score field with the reranker label/score!

```
{
   "query_id": "<query id>",
   "query": "<query>",
   "positive_passages": [
    {
      "docid": "<docid>", 
      "title": "<passage title>", 
      "text": "<passage text>", 
      "score": 1
    }, ...],
   "negative_passages": [
    {
      "docid": "<docid>", 
      "title": "<passage title>", 
      "text": "<passage text>", 
      "score": 0.75
    }, ...],
}
```

2. This example demonstrate train e5-base model using RLHN training data with labels from BAAI/bge-reranker-v2-gemma
```
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train_distil \
  --deepspeed ds_config.json \
  --output_dir e5-base-distill-bge-reranker-v2-gemma \
  --model_name_or_path intfloat/e5-base-unsupervised \
  --save_steps 1000 \
  --dataset_name rlhn/default-680K-bge-reranker-v2-gemma \
  --attn_implementation eager \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling mean \
  --normalize \
  --temperature 0.01 \
  --distil_temperature 0.02 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 2e-5 \
  --query_max_len 350 \
  --passage_max_len 350 \
  --num_train_epochs 5 \
  --logging_steps 5 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4
```