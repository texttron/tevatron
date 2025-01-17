# Colpali


## Encode
### Document Encode
We will score shards later, use large number of shards to split the corpus because we need to load shards in to gpu.
The index size is about 650GB.
```bash
encode_num_shard=100
for i in $(seq 0 $((encode_num_shard-1)))
do
CUDA_VISIBLE_DEVICES=0 python encode.py \
  --output_dir=temp \
  --model_name_or_path vidore/colpali-v1.2-hf \
  --bf16 \
  --per_device_eval_batch_size 8 \
  --dataset_name Tevatron/wiki-ss-corpus \
  --corpus_name Tevatron/wiki-ss-corpus \
  --dataset_number_of_shards $encode_num_shard \
  --dataset_shard_index $shard \
  --encode_output_path corpus.shard.$shard.pkl
done

```

### Query Encode
```bash
CUDA_VISIBLE_DEVICES=0 python encode.py \
  --output_dir=temp \
  --model_name_or_path vidore/colpali-v1.2-hf \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --query_max_len 128 \
  --dataset_name Tevatron/wiki-ss-nq \
  --corpus_name Tevatron/wiki-ss-corpus \
  --dataset_split test \
  --encode_output_path query.nq.pkl \
  --encode_is_query

```

## Search
```bash
CUDA_VISIBLE_DEVICES=0 python search.py \
    --query_reps query.nq.pkl \
    --passage_reps 'corpus.*.pkl' \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to run.colpali-nq.txt
```

## Evaluate
```bash
python eval_retrieval.py --run_file run.colpali-nq.txt --k 1

# top-1 score
Top-k Accuracy: 0.3518005540166205
```
