## Train UniCOIL on MS MARCO
```bash
CUDA_VISIBLE_DEVICES=0 python examples/unicoil/train_unicoil.py \
  --output_dir unicoil_distilbert \
  --model_name_or_path distilbert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --query_max_len 16 \
  --passage_max_len 128 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```

## Encode MS MARCO Corpus with UniCOIL
```bash
for s in $(seq -f "%02g" 0 19)
do
CUDA_VISIBLE_DEVICES=0 python examples/unicoil/encode_unicoil.py \
  --output_dir=temp \
  --model_name_or_path unicoil_distilbert \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encode_output_path corpus_emb.${s}.jsonl \
  --dataset_number_of_shards 20 \
  --dataset_shard_index ${s}
done

```
## Encode MS MARCO Query with UniCOIL
```bash
CUDA_VISIBLE_DEVICES=0 python examples/unicoil/encode_unicoil.py \
  --output_dir=temp \
  --model_name_or_path unicoil_distilbert \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --encode_is_query \
  --query_max_len 16 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_output_path queries_emb.tsv
```

## Indexing
```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input corpus_emb_unicoil_distilbert \
  --index index_unicoil_distilbert \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --impact --pretokenized --optimize
```

## Search
```bash
python -m pyserini.search.lucene \
  --index index_unicoil_distilbert \
  --topics queries_emb.tsv \
  --output run.msmarco-passage.unicoil-distilbert.tsv \
  --output-format msmarco \
  --batch 36 --threads 12 \
  --hits 1000 \
  --impact
```

## Evaluate 
```bash
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset run.msmarco-passage.unicoil-distilbert.tsv
#####################
MRR @10: 0.32835067312502864
QueriesRanked: 6980
#####################
```
