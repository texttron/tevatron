# MS-MARCO Passage Ranking
## Get Data
Run,
```
bash get_data.sh
```
This downloads the cleaned corpus, generate BM25 negatives and tokenize train/inference data using BERT tokenizer. The process could take up to tens of minutes depending on connection and hardware.

## Train a BERT Model
Train a BERT(`bert-base-uncased`) with mixed precision.
```
python -m tevatron.driver.train \  
  --output_dir ./retriever_model \  
  --model_name_or_path bert-base-uncased \  
  --save_steps 20000 \  
  --train_dir ./marco/bert/train \
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 2 \  
  --dataloader_num_workers 2
```

## Encode the Corpus and Query
```
mkdir encoding
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path encoding/split${i}.pt
done


python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path encoding/qry.pt
```

## Search the Corpus
```
mkdir -p ranking/intermediate

for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.faiss_retriever \
  --query_reps encoding/qry.pt \
  --passage_reps encoding/split${i}.pt \
  --depth 10 \
  --save_ranking_to ranking/intermediate/split${i}
done

python -m tevatron.faiss_retriever.reducer \
  --score_dir ranking/intermediate \
  --query encoding/qry.pt \
  --save_ranking_to ranking/rank.txt
```
Finally format the retrieval result,
```
python score_to_marco.py ranking/rank.txt
```
