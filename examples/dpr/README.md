# DPR reproduction

## NQ
In this doc, we use NQ as an example to show the reproduction of [DPR](https://github.com/facebookresearch/DPR) work from Tevatron.

For other datasets, simply download the datasets from original [DPR repo](https://github.com/facebookresearch/DPR)
```bash
TriviaQA: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
WebQuestions: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-train.json.gz
CuratedTREC: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz
SQuAD: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
```
Simply replace the train data from the following command accordingly, you should be able to reproduce results for other datasets.

### 1. Prepare train data
We use the train data provided by [DPR repo](https://github.com/facebookresearch/DPR).
1. Download train data
```bash
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
$ gzip -d biencoder-nq-train.json.gz
```
2. Convert train data format & do tokenization
```bash
$ python prepare_wiki_train.py --input biencoder-nq-train.json \
                               --output nq-train \
                               --tokenizer bert-base-uncased
```

### 2. Train
```bash
TRAIN_DIR=nq-train
OUTDIR=model-nq

python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir $OUTDIR \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --train_dir $TRAIN_DIR \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --negatives_x_device
```

If GPU memory is limited, you can train using gradient cache updates. See its [example](../gradient-cache.md).

### Encode
Download wikipedia corpus
```bash
wget https://www.dropbox.com/s/8ocbt0qpykszgeu/wikipedia-corpus.tar.gz
tar -xvf wikipedia-corpus.tar.gz
```

Encode Corpus
```bash
ENCODE_DIR=embeddings-nq
OUTDIR=temp
MODEL_DIR=model-nq
CORPUS_DIR=wikipedia-corpus
mkdir $ENCODE_DIR
for s in $(seq -f "%02g" 0 21)
do
python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --encode_in_path $CORPUS_DIR/docs$s.json \
  --encoded_save_path $ENCODE_DIR/$s.pt
done
```

Download queries
```bash
wget https://www.dropbox.com/s/x4abrhszjssq6gl/nq-test-queries.json
wget https://www.dropbox.com/s/b64e07jzlji8zhl/trivia-test-queries.json
```

Encode Query
```bash
ENCODE_QRY_DIR=embeddings-nq-queries
OUTDIR=temp
MODEL_DIR=model-nq
QUERY=nq-test-queries.json
mkdir $ENCODE_QRY_DIR
python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --encode_in_path $QUERY \
  --encoded_save_path $ENCODE_QRY_DIR/query.pt
```


### Search
```bash
ENCODE_QRY_DIR=embeddings-nq-queries
ENCODE_DIR=embeddings-nq
DEPTH=100
RUN=run.nq.test.txt
python -m tevatron.faiss_retriever \
--query_reps $ENCODE_QRY_DIR/query.pt \
--passage_reps $ENCODE_DIR/'*.pt' \
--depth $DEPTH \
--batch_size -1 \
--save_text \
--save_ranking_to $RUN
```

### Evaluation
Convert result to trec format
```bash
RUN=run.nq.test.txt
TREC_RUN=run.nq.test.trec
python -m tevatron.utils.format.convert_result_to_trec --input $RUN --output $TREC_RUN
```

Evaluate with Pyserini for now, `pip install pyserini`
Recover query and passage contents
```bash
TREC_RUN=run.nq.test.trec
JSON_RUN=run.nq.test.json
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics dpr-nq-test \
                                                                --index wikipedia-dpr \
                                                                --input $TREC_RUN \
                                                                --output $JSON_RUN
```
> If you are working on `dpr-curated-test`, add `--regex` for the above command.

```bash
$ python -m pyserini.eval.evaluate_dpr_retrieval --retrieval $JSON_RUN --topk 20 100
Top20	accuracy: 0.8002770083102493
Top100	accuracy: 0.871191135734072
```

## Un-tie model
Un-tie model is that the query encoder and passage encoder do not share parameters.
To train untie models, simply add `--untie_encoder` option to the training command.
> Note: In original DPR work, passage and query encoders do not share parameters.

## (Alternatives) Train DPR with our self contained datasets
The above instructions uses train data downloaded from DPR. Tevatron also have self-contained
pre-processed datasets in HuggingFace [datasets hub](https://huggingface.co/Tevatron). 

To train and inference without saving additional dump of data, please following the commands below:

### Train (self-contained dataset)
```bash
python -m torch.distributed.launch --nproc_per_node=4 run.py \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 10 \
  --untie_encoder \
  --negatives_x_device \
  --overwrite_output_dir
```

### Encode (self-contained dataset)
Corpus:
```bash
ENCODE_DIR=embeddings-nq
OUTDIR=temp
MODEL_DIR=model-nq
mkdir $ENCODE_DIR
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path $ENCODE_DIR/$s.pt \
  --encode_num_shard 20 \
  --encode_shard_index $s
done
```

Queries:
```bash
ENCODE_QRY_DIR=embeddings-nq-queries
OUTDIR=temp
MODEL_DIR=model-nq
mkdir $ENCODE_QRY_DIR
python -m tevatron.driver.encode \
  --output_dir=$OUTDIR \
  --model_name_or_path $MODEL_DIR \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path $ENCODE_QRY_DIR/query.pt \
  --encode_is_qry
```

## Evaluation 
The evaluation process are same as above.

## Other self-contained datasets
If you want to work on other datasets (such as TriviaQA). Simply change the datasets name (`--dataset_name`) to other dataset.
We currently support following datasets:
- NQ: `Tevatron/wikipedia-nq`
- TriviaQA: `Tevatron/wikipedia-trivia`
- WebQuestions: `Tevatron/wikipedia-wq`
- CuratedTREC: `Tevatron/wikipedia-curated`
- SQuAD: `Tevatron/wikipedia-curated`
- MsMarco: `Tevatron/msmarco-passage`
- SciFact: `Tevatron/scifact`


## Summary
Using the process above should be able to obtain `top-k` retrieval accuracy as below:

| Dataset/Model  | Top20 | Top100 |
|----------------|-------|--------|
| NQ             | 0.81  | 0.86   |
| NQ-untie       | 0.80  | 0.87   |
| TriviaQA       | 0.81  | 0.86   |
| TriviaQA-untie | 0.81  | 0.86   |
| WebQuestion    | 0.75  | 0.83   |
| CuratedTREC    | 0.84  | 0.91   |

The above results successfully replicated numbers reported in the original [DPR paper](https://arxiv.org/pdf/2004.04906.pdf)
