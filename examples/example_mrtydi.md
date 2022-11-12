# Mr.TyDi

In this example, we use dataset [Mr.TyDi](https://github.com/castorini/mr.tydi) 
to show training dense retrieval model on non-English data.

## Dataset Preparation 
The Mr.TyDi dataset is self contain in our toolkit based on huggingface datasets.
The dataset name is `castorini/mr-tydi`. 

Please use the format `castorini/mr-tydi:<language>` to access different languages during
training and encoding with Tevatron.
See below for details.

In the example below, we use language `bengali` as example.
## Train
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_mrtydi_bengali \
  --model_name_or_path bert-base-multilingual-cased \
  --save_steps 20000 \
  --dataset_name castorini/mr-tydi:bengali \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 40 \
  --grad_cache \
  --gc_p_chunk_size 8 \
  --logging_steps 10 \
  --overwrite_output_dir
```

## Encode Corpus
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp_out \
  --model_name_or_path model_mrtydi_bengali \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --dataset_name castorini/mr-tydi-corpus:bengali \
  --p_max_len 256 \
  --encoded_save_path corpus_emb.pt 
```

## Encode Query
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp_out \
  --model_name_or_path model_mrtydi_bengali \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name castorini/mr-tydi:bengali/test \
  --encode_is_qry \
  --q_max_len 64 \
  --encoded_save_path queries_emb.pt 
```

## Search
```bash
python -m tevatron.faiss_retriever \
--query_reps queries_emb.pt \
--passage_reps corpus_emb.pt \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to run.mrtydi.bengali.test.txt
```

## Evaluate
Download qrels from [Mr.TyDi](https://github.com/castorini/mr.tydi) repo
```bash
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-bengali.tar.gz
tar -xvf mrtydi-v1.0-bengali.tar.gz
```

Evaluate
```bash
python -m tevatron.utils.format.convert_result_to_trec --input run.mrtydi.bengali.test.txt --output run.mrtydi.bengali.test.trec
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 mrtydi-v1.0-bengali/qrels.test.txt run.mrtydi.bengali.test.trec
```

Following results can be reproduced by following the instructions above:
```
recip_rank              all     0.6130
recall_100              all     0.9144
```
## Reproduction Log
+ Results reproduced by [@Jasonwu-0803](https://github.com/Jasonwu-0803) on 2022-09-27 (commit [`b8f3390`](https://github.com/texttron/tevatron/commit/b8f33900895930f9886012580e85464a5c1f7e9a))
