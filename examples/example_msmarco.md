# MS MARCO passage ranking example

In this doc, we show the steps to train dense retriever for MS MARCO passage ranking task

## Training
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_msmarco \
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

## Encoding
### Encode Corpus
```bash
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path corpus_emb.${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s}
done
```

### Encode Corpus
```bash
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_msmarco \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

## Search the Corpus
```bash
python -m tevatron.faiss_retriever \
--query_reps query_emb.pkl \
--passage_reps 'corpus_emb.*.pkl' \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to rank.txt
```

### Evaluation
Convert result to MARCO format
```bash
python -m tevatron.utils.format.convert_result_to_marco \
              --input rank.txt \
              --output rank.txt.marco
```

Evaluate with Pyserini (`pip install pyserini`)

```bash
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset rank.txt.marco
```

By following this doc, we are able to train a dense retriever for MS MARCO passage ranking
task with `MRR@10: 0.322`

## Reproduction Log
+ Results reproduced by [@Jasonwu-0803](https://github.com/Jasonwu-0803) on 2022-09-27 (commit [`b8f3390`](https://github.com/texttron/tevatron/commit/b8f33900895930f9886012580e85464a5c1f7e9a))
