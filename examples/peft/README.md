## Train

```
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_scifact_peft \
  --model_name_or_path bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/scifact \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 3e-4 \
  --q_max_len 64 \
  --p_max_len 512 \
  --num_train_epochs 80 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --peft \
  --grad_cache \
  --gc_p_chunk_size 16 \
  --gc_q_chunk_size 16 \
  --logging_steps 10
```

## Eval
```
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp_out \
  --model_name_or_path model_scifact_peft \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/scifact-corpus \
  --p_max_len 512 \
  --encoded_save_path corpus_emb.pt
```

```
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp_out \
  --model_name_or_path model_scifact_peft \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/scifact/dev \
  --encode_is_qry \
  --q_max_len 64 \
  --encoded_save_path queries_emb.pt 
```


```
python -m tevatron.faiss_retriever \
--query_reps queries_emb.pt \
--passage_reps corpus_emb.pt \
--depth 20 \
--batch_size -1 \
--save_text \
--save_ranking_to run.scifact.lora.dev.txt
```