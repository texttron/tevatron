# MS MARCO passage reranking example
In this example, we take the retrieval results from the first stage retriever (e.g. BM25 or neural bi-encoder) and do rerank with cross-encoder on MSMARCO Passage Ranking dataset.

## Train Reranker
```
CUDA_VISIBLE_DEVICES=0 python train_reranker.py \
  --output_dir reranker_msmarco \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 128 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```

## Inference
We use BM25 retrieval results as candidates for reranking.

### Get first stage retrieval results

We use `pyserini` to directly reproduce the default BM25 retrieval results on MS MARCO passage ranking development set.

```
python -m pyserini.search.lucene \
  --topics msmarco-passage-dev-subset \
  --index msmarco-v1-passage-slim \
  --output run.msmarco-v1-passage.bm25-default.dev.txt \
  --bm25 --k1 0.9 --b 0.4
```
The BM25 result has `topk=1000` passages for each query. It gives MRR@10=0.1840 on msmarco passage dev set.

### Prepare rerank input file

```
python prepare_rerank_file.py \
    --query_data_name Tevatron/msmarco-passage \
    --corpus Tevatron/msmarco-passage-corpus \
    --retrieval run.msmarco-v1-passage.bm25-default.dev.txt \
    --output_path rerank_input_file.bm25.jsonl
```

### Reranking

```
CUDA_VISIBLE_DEVICES=6 python reranker_inference.py \
  --output_dir=temp \
  --model_name_or_path reranker_msmarco \
  --tokenizer_name bert-base-uncased \
  --encode_in_path rerank_input_file.bm25.jsonl \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --q_max_len 16 \
  --p_max_len 128 \
  --dataset_name data_script.py \
  --encoded_save_path rerank_out_file.bm25.monobert.txt
```

### Evaluation
```
python -m tevatron.utils.format.convert_result_to_marco \
              --input rerank_out_file.bm25.monobert.txt \
              --output rerank_out_file.bm25.monobert.marco

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset rerank_out_file.bm25.monobert.marco
```

The reranker trained by following this instuction should gives following effectiveness:
```
#####################
MRR @10: 0.3775321212534682
QueriesRanked: 6980
#####################
```
