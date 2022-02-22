# Encoding
## Corpus Encoding
To encode, using the `tevatron.driver.encode` module. 
```bash
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus_emb.pkl
```

### Sharded Encoding
For large corpus, split the corpus into shards to parallelize can speed up the process.
Following code did same thing as above but splits the corpus into 20 shards.

```bash
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus_emb_${s}.pkl \
  --encode_num_shard 20 \
  --encode_shard_index ${s}
done
```

> Here we are using our self-contained datasets to train. 
To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq-corpus` by
> `--encode_in_path <file to encode>`. (see here for details)


## Query Encoding
```
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path query.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

> Here we are using our self-contained datasets to train. 
To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq-corpus` by
> `--encode_in_path <file to encode>`. (see here for details)


## Encoding on TPU
To encode with TPU, simply replace `tevatron.driver.encode` module with 
`tevatron.driver.jax_encode`

I.e. the following command will do same thing as above but with Jax/Flax:
```
python -m tevatron.driver.jax_encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus_emb.pkl
```