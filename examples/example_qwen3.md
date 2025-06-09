# Qwen3-Embedding Finetuning Example

The doc provide examples to finetune [Qwen3-Embedding](https://qwenlm.github.io/blog/qwen3-embedding/).
The prompt used for the dataset is follow the official prompts list [here](https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/task_prompts.json)
## Inference Qwen3-Embedding (BEIR scifact example)
### Encoding
```bash

# Encode queries
mkdir beir_embedding_scifact_qwen3
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-4B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last \
  --padding_side left \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --query_max_len 512 \
  --dataset_name Tevatron/beir \
  --dataset_config scifact \
  --dataset_split test \
  --encode_output_path beir_embedding_scifact_qwen3/queries_scifact.pkl \
  --encode_is_query


# Encode corpus
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path Qwen/Qwen3-Embedding-4B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last \
  --padding_side left \
  --passage_prefix "" \
  --passage_max_len 512 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config scifact \
  --dataset_split train \
  --encode_output_path beir_embedding_scifact_qwen3/corpus_scifact.pkl
```

### Search
```bash
mkdir beir_results_scifact_qwen3
python -m tevatron.retriever.driver.search \
    --query_reps beir_embedding_scifact_qwen3/queries_scifact.pkl \
    --passage_reps beir_embedding_scifact_qwen3/corpus_scifact.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to beir_results_scifact_qwen3/rank.scifact.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input beir_results_scifact_qwen3/rank.scifact.txt \
                                                       --output beir_results_scifact_qwen3/rank.scifact.trec \
                                                       --remove_query
```
### Eval
```bash
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-scifact-test beir_results_scifact_qwen3/rank.scifact.trec

# recall_100              all     0.9767
# ndcg_cut_10             all     0.7801


```


## Finetune Qwen3-Embedding

```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-qwen3-emb-ft \
  --model_name_or_path Qwen/Qwen3-Embedding-4B \ # replace by 0.6B/4B/8B
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name Tevatron/scifact \
  --query_prefix "Instruct: Given a scientific claim, retrieve documents that support or refute the claim.\nQuery:" \
  --passage_prefix "" \
  --bf16 \
  --pooling last \
  --padding_side left \
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
