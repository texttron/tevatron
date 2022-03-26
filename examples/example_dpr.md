# DPR example

In this doc, we use NQ as an example to show the replication of [DPR](https://github.com/facebookresearch/DPR) work from Tevatron.

## Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
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
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --fp16 \
  --per_device_train_batch_size 128 \
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
mkdir $ENCODE_DIR
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus_emb.$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done
```

### Encode Queries
```bash
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path query_emb.pkl \
  --encode_is_qry
```

### Search
```bash
python -m tevatron.faiss_retriever \
--query_reps query_emb.pkl \
--passage_reps 'corpus_emb.*.pkl' \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to run.nq.test.txt
```

### Evaluation
Convert result to trec format
```bash
python -m tevatron.utils.format.convert_result_to_trec \
              --input run.nq.test.txt \
              --output run.nq.test.trec
```

Evaluate with Pyserini, `pip install pyserini`
Recover query and passage contents
```bash
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input run.nq.test.trec \
              --output run.nq.test.json
```
> If you are working on `dpr-curated-test`, add `--regex` for the above command.

```bash
$ python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval run.nq.test.json \
                --topk 20 100

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
