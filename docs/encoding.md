# Encoding

## Corpus encoding

Run the PyTorch driver [`tevatron.retriever.driver.encode`](../src/tevatron/retriever/driver/encode.py):

```bash
python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encode_output_path corpus_emb.pkl
```

### Sharded encoding

For a large corpus, split it with Hugging Face `datasets` sharding: `--dataset_number_of_shards` and `--dataset_shard_index` (see [`DataArguments`](../src/tevatron/retriever/arguments.py)).

```bash
for s in $(seq 0 19)
do
python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encode_output_path corpus_emb_$(printf '%02d' $s).pkl \
  --dataset_number_of_shards 20 \
  --dataset_shard_index ${s}
done
```

For self-contained hub corpora, keep `--dataset_name`. For local JSON/JSONL, set `--dataset_name json` (or the default) and pass the file(s) with `--dataset_path`.

## Query encoding

```bash
python -m tevatron.retriever.driver.encode \
  --output_dir temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq \
  --dataset_split test \
  --encode_output_path query_emb.pkl \
  --query_max_len 32 \
  --encode_is_query
```

> Here we are using our self-contained datasets to train. 
> To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq-corpus` by
> `--encode_in_path <file to encode>`. (see here for details)

## Encoding on TPU (JAX / Flax)

[`tevatron.retriever.driver.jax_encode`](../src/tevatron/retriever/driver/jax_encode.py) is an optional JAX path with a **different** CLI than the PyTorch encoder above. 

I.e. the following command will do same thing as above but with Jax/Flax:
```
python -m tevatron.driver.jax_encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encode_output_path corpus_emb.pkl
```