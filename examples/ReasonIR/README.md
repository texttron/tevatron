
# ReasonIR BRIGHT Retrieval Example

prompts follows: https://github.com/facebookresearch/ReasonIR/tree/main/evaluation/bright/configs/reasonir

In [bright_qrels](bright_qrels) we provided the qrels for BRIGHT datasets, which is used to evaluate the retrieval results.

```bash
dataset=biology
embedding_path=bright_embeddings/${dataset}/reasonir

mkdir -p ${embedding_path}
CUDA_VISIBLE_DEVICES=0 python encode.py  \
  --output_dir=temp \
  --model_name_or_path reasonir/ReasonIR-8B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --dataset_name Tevatron/bright \
  --dataset_config ${dataset} \
  --dataset_split test \
  --encode_is_query \
  --query_prefix "<|user|>\nGiven a Biology post, retrieve relevant passages that help answer the post\n<|embed|>\n" \
  --encode_output_path ${embedding_path}/queries.pkl

CUDA_VISIBLE_DEVICES=0 python encode.py  \
  --output_dir=temp \
  --model_name_or_path reasonir/ReasonIR-8B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --dataset_name Tevatron/bright-corpus \
  --dataset_config ${dataset} \
  --dataset_split train \
  --passage_prefix "<|embed|>\n" \
  --encode_output_path ${embedding_path}/corpus.pkl
```


### Search
```bash
mkdir -p bright_results/${dataset}/reasonir
python -m tevatron.retriever.driver.search \
    --query_reps ${embedding_path}/queries.pkl \
    --passage_reps ${embedding_path}/corpus.pkl \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to bright_results/${dataset}/reasonir/rank.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input bright_results/${dataset}/reasonir/rank.txt \
                                                       --output bright_results/${dataset}/reasonir/rank.trec
```
### Eval
```bash
python -m pyserini.eval.trec_eval -c -m recall.100 -m ndcg_cut.10 bright_qrels/biology.tsv bright_results/${dataset}/reasonir/rank.trec

recall_100              all     0.6390
ndcg_cut_10             all     0.2533

```