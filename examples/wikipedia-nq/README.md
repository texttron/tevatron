# Wikipedia Natural Questions

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

python -m torch.distributed.launch --nproc_per_node=4 -m dense.driver.train \
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

### Encode (with Pyserini)
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
python -m dense.driver.encode \
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
python -m dense.driver.encode \
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
DEPTH=1000
RUN=run.nq.test.txt
python -m dense.faiss_retriever \
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
python -m dense.utils.format.result_to_trec --input $RUN --output $TREC_RUN
```

Evaluate with Pyserini for now, `pip install pyserini`
Recover query and passage contents
```bash
TREC_RUN=run.nq.test.trec
JSON_RUN=run.nq.test.json
$ python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics dpr-nq-test \
                                                                --index wikipedia-dpr \
                                                                --input $TREC_RUN \
                                                                --output $JSON_RUN
```
```bash
$ python -m pyserini.eval.evaluate_dpr_retrieval --retrieval $JSON_RUN --topk 20 100
Top20	accuracy: 0.8002770083102493
Top100	accuracy: 0.871191135734072
```