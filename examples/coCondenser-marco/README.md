# coCondenser MS-MARCO Passage Retrieval
## coCondenser
You can find details about coCondenser pre-training in its [paper](https://arxiv.org/abs/2108.05540) and [open source code](https://github.com/luyug/Condenser),
```
@misc{gao2021unsupervised,
      title={Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval}, 
      author={Luyu Gao and Jamie Callan},
      year={2021},
      eprint={2108.05540},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
## Get Data
Run,
```
bash get_data.sh
```
This downloads the cleaned corpus hosted by RocketQA team, generate BM25 negatives and tokenize train/inference data using BERT tokenizer. 
The process could take up to tens of minutes depending on connection and hardware.

## Inference with Fine-tuned Checkpoint
You can obtain a fine-tuned retriever from HF hub using the identifier ` Luyu/co-condenser-marco-retriever`.
### Encode
```
mkdir -p encoding/corpus
mkdir -p encoding/query
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path Luyu/co-condenser-marco-retriever \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path encoding/corpus/split${i}.pt
done


python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path Luyu/co-condenser-marco-retriever \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path encoding/query/qry.pt
```
### Index Search
```
python -m tevatron.faiss_retriever \  
--query_reps encoding/query/qry.pt \  
--passage_reps encoding/corpus/'*.pt' \  
--depth 10 \
--batch_size -1 \
--save_text \
--save_ranking_to rank.tsv
```
And format the retrieval result,
```
python ../msmarco-passage-ranking/score_to_marco.py rank.txt
```
## Fine-tuning Stage 1
Pick a pre-trained condenser that is most suitable for the experiment from [Condenser Repo](https://github.com/luyug/Condenser#pre-trained-models).
Train
```
python -m tevatron.driver.train \  
  --output_dir ./retriever_model_s1 \  
  --model_name_or_path CONDENSER_MODEL_NAME \  
  --save_steps 20000 \  
  --train_dir ./marco/bert/train \
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 3 \  
  --dataloader_num_workers 2
```
## Mining Hard Negatives
### Encode
Encode corpus and train queries,
```
mkdir -p encoding/corpus
mkdir -p encoding/query
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model_s1 \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path encoding/corpus/split${i}.pt
done

python -m tevatron.driver.encode \  
  --output_dir ./retriever_model \
  --model_name_or_path ./retriever_model_s1 \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/train.query.json \
  --encoded_save_path encoding/query/train.pt
```

### Search
```
python -m tevatron.faiss_retriever \  
--query_reps encoding/query/train.pt \  
--passage_reps encoding/corpus/'*.pt' \  
--batch_size 5000 \
--save_text \
--save_ranking_to train.rank.tsv
```

### Build HN Train file
```
bash create_hn.sh
```

## Fine-tuning Stage 2
```
python -m tevatron.driver.train \  
  --output_dir ./retriever_model_s2 \  
  --model_name_or_path CONDENSER_MODEL_NAME \  
  --save_steps 20000 \  
  --train_dir ./marco/bert/train-hn \
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 2 \  
  --dataloader_num_workers 2
```

## Encode and Search
Do encoding,
```
mkdir -p encoding/corpus-s2
mkdir -p encoding/query-s2
for i in $(seq -f "%02g" 0 9)
do
python -m tevatron.driver.encode \  
  --output_dir ./retriever_model_s2 \
  --model_name_or_path ./retriever_model_s2 \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/corpus/split${i}.json \
  --encoded_save_path encoding/corpus-s2/split${i}.pt
done

python -m tevatron.driver.encode \  
  --output_dir  ./retriever_model_s2 \
  --model_name_or_path  ./retriever_model_s2 \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path encoding/query-s2/qry.pt
```
Run the retriever,
```
python -m tevatron.faiss_retriever \  
--query_reps encoding/query-s2/qry.pt \  
--passage_reps encoding/corpus-s2/'*.pt' \  
--depth 10 \
--batch_size -1 \
--save_text \
--save_ranking_to dev.rank.tsv
```
And format the retrieval result,
```
python ../msmarco-passage-ranking/score_to_marco.py dev.rank.tsv
```
