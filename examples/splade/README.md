## Train SPLADE on MS MARCO
```bash  
CUDA_VISIBLE_DEVICES=0 python train_splade.py \
  --output_dir model_msmarco_splade \
  --model_name_or_path Luyu/co-condenser-marco \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --query_max_len 128 \
  --passage_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --dataloader_num_workers 8 \
  --num_proc 8 \
  --logging_steps 500 \
  --attn_implementation sdpa \
  --overwrite_output_dir
```
----

## Evaluation 
Install Pyserini 
```bash
pip install pyserini
```
> [!NOTE]  
> The [psyserini](https://github.com/castorini/pyserini/tree/master) library as of v0.44.0 is using features that are no longer supported for python version >3.10; hence, configure your environment to use python3.10

### MSMARCO eval
SPLADE encoding can be done as follows:

#### Encode documents and queries
```bash
mkdir -p encoding_splade/corpus
mkdir -p encoding_splade/query
for i in $(seq -f "%02g" 0 4)
do
CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 128 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 5 \
  --dataset_shard_index ${i} \
  --encode_output_path encoding_splade/corpus/split${i}.jsonl
done

CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --query_max_len 128 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_output_path encoding_splade/query/dev.tsv
```



#### Index SPLADE with pyserini

```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/corpus \
  --index splade_pyserini_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### Retrieve SPLADE with pyserini

```bash
 python -m pyserini.search.lucene \
  --index splade_pyserini_index \
  --topics encoding_splade/query/dev.tsv \
  --output splade_results.txt \
  --batch 36 --threads 32 \
  --hits 1000 \
  --impact
```

#### Evaluate SPLADE with pyserini

```bash
python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset splade_results.txt

# recip_rank              all     0.3770
```
----

### BEIR eval
#### Encode documents and queries
```bash
dataset=nfcorpus
mkdir -p encoding_splade/${dataset}/corpus
mkdir -p encoding_splade/${dataset}/query

CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 512 \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config ${dataset} \
  --dataset_split train \
  --dataset_number_of_shards 1 \
  --dataset_shard_index 0 \
  --encode_output_path encoding_splade/${dataset}/corpus/split00.jsonl


CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --query_max_len 512 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/beir \
  --dataset_config ${dataset} \
  --dataset_split test \
  --encode_output_path encoding_splade/${dataset}/query/query.tsv
```



#### Index SPLADE with pyserini
```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/${dataset}/corpus \
  --index splade_pyserini_index_beir/${dataset} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### Retrieve SPLADE with pyserini

```bash
python -m pyserini.search.lucene \
  --index splade_pyserini_index_beir/${dataset} \
  --topics encoding_splade/${dataset}/query/query.tsv \
  --output splade_results_${dataset}.txt \
  --output-format msmarco \
  --batch 36 --threads 32 \
  --hits 1000 \
  --impact
  
python convert_result_to_trec.py \
    --input splade_results_${dataset}.txt \
    --output splade_results_${dataset}.trec \
    --remove_query
```

#### Evaluate SPLADE with pys## Train SPLADE on MS MARCO
```bash  
CUDA_VISIBLE_DEVICES=0 python train_splade.py \
  --output_dir model_msmarco_splade \
  --model_name_or_path Luyu/co-condenser-marco \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_group_size 8 \
  --learning_rate 5e-6 \
  --query_max_len 128 \
  --passage_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --dataloader_num_workers 8 \
  --num_proc 8 \
  --logging_steps 500 \
  --attn_implementation sdpa \
  --overwrite_output_dir
```
----

## Evaluation 
Install Pyserini 
```bash
pip install pyserini
```
> [!NOTE]  
> The [psyserini](https://github.com/castorini/pyserini/tree/master) library as of v0.44.0 is using features that are no longer supported for python version >3.10; hence, configure your environment to use python3.10

### MSMARCO eval
SPLADE encoding can be done as follows:

#### Encode documents and queries
```bash
mkdir -p encoding_splade/corpus
mkdir -p encoding_splade/query
for i in $(seq -f "%02g" 0 4)
do
CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 128 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 5 \
  --dataset_shard_index ${i} \
  --encode_output_path encoding_splade/corpus/split${i}.jsonl
done

CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --query_max_len 128 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_output_path encoding_splade/query/dev.tsv
```



#### Index SPLADE with pyserini

```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/corpus \
  --index splade_pyserini_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### Retrieve SPLADE with pyserini

```bash
 python -m pyserini.search.lucene \
  --index splade_pyserini_index \
  --topics encoding_splade/query/dev.tsv \
  --output splade_results.txt \
  --batch 36 --threads 32 \
  --hits 1000 \
  --impact --pretokenized

```

#### Evaluate SPLADE with pyserini

```bash
python -m pyserini.eval.trec_eval -c -M 10 -m recip_rank msmarco-passage-dev-subset splade_results.txt

# recip_rank              all     0.3770
```
----

### BEIR eval
#### Encode documents and queries
```bash
dataset=nfcorpus
mkdir -p encoding_splade/${dataset}/corpus
mkdir -p encoding_splade/${dataset}/query

CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 512 \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/beir-corpus \
  --dataset_config ${dataset} \
  --dataset_split train \
  --dataset_number_of_shards 1 \
  --dataset_shard_index 0 \
  --encode_output_path encoding_splade/${dataset}/corpus/split00.jsonl


CUDA_VISIBLE_DEVICES=0 python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --query_max_len 512 \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/beir \
  --dataset_config ${dataset} \
  --dataset_split test \
  --encode_output_path encoding_splade/${dataset}/query/query.tsv
```



#### Index SPLADE with pyserini
```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/${dataset}/corpus \
  --index splade_pyserini_index_beir/${dataset} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### Retrieve SPLADE with pyserini

```bash
python -m pyserini.search.lucene \
  --index splade_pyserini_index_beir/${dataset} \
  --topics encoding_splade/${dataset}/query/query.tsv \
  --output splade_results_${dataset}.txt \
  --batch 36 --threads 32 \
  --hits 1000 \
  --impact --pretokenized --remove-query
  
```

#### Evaluate SPLADE with pyserini

```bash
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test splade_results_${dataset}.txt

# recall_100              all     0.9153
# ndcg_cut_10             all     0.6864
```

```bash
python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test splade_results_${dataset}.trec

# recall_100              all     0.9153
# ndcg_cut_10             all     0.6864
```

