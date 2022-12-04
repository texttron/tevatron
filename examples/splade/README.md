## Train SPLADE on MS MARCO
```bash
CUDA_VISIBLE_DEVICES=0 python train_splade.py \
  --output_dir model_msmarco_splade \
  --model_name_or_path Luyu/co-condenser-marco \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 128 \
  --p_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```

## Encode SPLADE 

SPLADE encoding can be done as follows:

```bash
mkdir -p encoding_splade/corpus
mkdir -p encoding_splade/query
for i in $(seq -f "%02g" 0 9)
do
python encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encode_num_shard 10 \
  --encode_shard_index ${i} \
  --encoded_save_path encoding_splade/corpus/split${i}.jsonl
done

python -m encode_splade.py \
  --output_dir encoding_splade \
  --model_name_or_path model_msmarco_splade \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path encoding_splade/query/dev.tsv
```

## Index SPLADE with anserini
In the following, we consider that [ANSERINI](https://github.com/castorini/anserini) is installed, with all its tools, in $PATH_ANSERINI
```
sh $PATH_ANSERINI/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input encoding_splade/corpus \
 -index splade_anserini_index \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 16
```

## Retrieve SPLADE with anserini

```
sh $PATH_ANSERINI/target/appassembler/bin/SearchCollection -hits 1000 -parallelism 32 \
 -index splade_anserini_index \
 -topicreader TsvInt -topics encoding_splade/query/dev.tsv \
 -output splade_result.trec -format trec \
 -impact -pretokenized
```

## Evaluate SPLADE with anserini

```
$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
$PATH_ANSERINI/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
splade_result.trec

$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
$PATH_ANSERINI/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
splade_result.trec
```

## Index SPLADE with pyserini
In the following, we consider that [pyserini](https://github.com/castorini/pyserini) is installed (`pip install pyserini`).
```
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input encoding_splade/corpus \
  --index splade_anserini_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

## Retrieve SPLADE with pyserini

```
 python -m pyserini.search.lucene \
  --index splade_anserini_index \
  --topics encoding_splade/query/dev.tsv \
  --output splade_results.tsv \
  --output-format msmarco \
  --batch 36 --threads 32 \
  --hits 1000 \
  --impact
```

## Evaluate SPLADE with pyserini

```
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset splade_results.tsv
```
By following this doc, we are able to train a SPLADE for MS MARCO passage ranking task with MRR@10: 0.355
