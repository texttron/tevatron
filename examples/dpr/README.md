# DPR replication

In this doc, we use NQ as an example to show the replication of [DPR](https://github.com/facebookresearch/DPR) work from Tevatron.

## Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --fp16 \
  --per_device_train_batch_size 32 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --negatives_x_device \
  --overwrite_output_dir
```

The above command train DPR with 4 GPUs.
If GPU memory is limited, you can train using [gradient cache]((../gradient-cache.md)) updates.

The command below train DPR on single GPU with gradient cache:
```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --fp16 \
  --per_device_train_batch_size 32 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --grad_cache \
  --overwrite_output_dir
```

### Un-tie model
Un-tie model is that the query encoder and passage encoder do not share parameters.
To train untie models, simply add `--untie_encoder` option to the training command.
> Note: In original DPR work, passage and query encoders do not share parameters.

## Encode
### Encode Corpus
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
  --encoded_save_path $ENCODE_DIR/$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done
```

### Encode Queries
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
  --encoded_save_path $ENCODE_QRY_DIR/query.pkl \
  --encode_is_qry
```

### Search
```bash
ENCODE_QRY_DIR=embeddings-nq-queries
ENCODE_DIR=embeddings-nq
DEPTH=100
RUN=run.nq.test.txt
python -m tevatron.faiss_retriever \
--query_reps $ENCODE_QRY_DIR/query.pkl \
--passage_reps $ENCODE_DIR/'*.pkl' \
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

Evaluate with Pyserini, `pip install pyserini`
Recover query and passage contents
```bash
TREC_RUN=run.nq.test.trec
JSON_RUN=run.nq.test.json
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
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

The above results successfully replicated numbers reported in the
original [DPR paper](https://arxiv.org/pdf/2004.04906.pdf)
