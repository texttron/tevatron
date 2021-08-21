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

The model checkpoint can be directly used in [Pyserini](https://github.com/castorini/pyserini/) for encoding and evaluation.
Please follow the guidance in Pyserini repo to install.
### Encode (with Pyserini)
Download wikipedia corpus
```bash
wget https://www.dropbox.com/s/n0c0ypuo25zdks0/wikipedia-dpr-jsonl.tar.gz
```

Run encoding and save as Faiss index
```bash
for i in $(seq -f "%02g" 0 21)
do
python -m pyserini.encode input   --corpus wikipedia-dpr-jsonl/docs${i}.json \
                                  --fields title text \
                          output  --embeddings dense-nq-faiss-${i} \
                                  --to-faiss \
                          encoder --encoder model-nq \
                                  --fields title text \
                                  --batch 32 \
                                  --fp16
done
```

Merge index
```bash
$ python -m pyserini.dindex.merge_indexes --prefix dense-nq-faiss- --shard-num 22
```

### Search (with Pyserini)
```bash
$ python -m pyserini.dsearch --topics dpr-nq-test \
                             --index dense-nq-faiss-full \
                             --encoder model-nq \
	                         --output runs/run.dense.nq.trec \
                             --batch-size 36 --threads 12
```

### Evaluation (with Pyserini)
```bash
$ python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run --topics dpr-nq-test \
                                                                --index wikipedia-dpr \
                                                                --input runs/run.dense.nq.trec \
                                                                --output runs/run.dense.nq.json
```
```bash
$ python -m pyserini.eval.evaluate_dpr_retrieval --retrieval runs/run.dense.nq.json --topk 20 100
Top20	accuracy: 0.8002770083102493
Top100	accuracy: 0.871191135734072
```