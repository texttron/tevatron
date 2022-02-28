# Datasets

## Dataset types
There are usually two types of dataset format for dense retrieval training based on
whether the relevancy of document is human judged or by answer exactly matching.

### 1. Relevancy Judged Dataset
If the relevancy of a passage is annotated, (e.g. MS MARCO passage ranking),
an instance in the dataset can usually be organized in following format:
```json
{
   "query_id": "<query id>",
   "query": "<query text>",
   "positive_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ],
   "negative_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ]
}
```
where the passages in `positive_passages` are the annotated relevant passages of the `query` 
and passages in `negative_passages` are usually non-relevant passages from top results of a retrieval system (e.g. BM25).

### 2.Exactly Matched Dataset
If the relevancy of a passage is judged by answer exactly matching, (e.g. Natural Question),
an instance in the dataset can usually be organized in following format:
```json
{
   "query_id": "<query id>",
   "query": "<query text>",
   "answers": ["<answer>"],
   "positive_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ],
   "negative_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ]
}
```
where the passages in `positive_passages` has subsequence that exactly matches one of the answer string in `answers`.
And passages in `negative_passages` are usually passages from top results of a retrieval system but doesn't have 
subsequence exactly matches any of answer in `answers`.

## Self-Contained Dataset
Tevatron self-contained following common use datasets for dense retrieval. 
(via [HuggingFace](https://huggingface.co/Tevatron)).
These datasets will be downloaded and tokenized automatically during training and encoding
by setting `--dataset_name <hgf dataset name>`.

| dataset      | dataset HuggingFace name     | type             |
|--------------|------------------------------|------------------|
| MS MARCO     | `Tevatron/msmarco-passage`   | Relevancy Judged |
| SciFact      | `Tevatron/scifact`           | Relevancy Judged |
| NQ           | `Tevatron/wikipedia-nq`      | Exactly Match    |
| TriviaQA     | `Tevatron/wikipedia-trivia`  | Exactly Match    |
| WebQuestions | `Tevatron/wikipedia-wq`      | Exactly Match    |
| CuratedTREC  | `Tevatron/wikipedia-curated` | Exactly Match    |
| SQuAD        | `Tevatron/wikipedia-squad`   | Exactly Match    |

> Note: the self-contained datasets come with BM25 negative passages by default

Take SciFact as an example:

We can directly train with self-contained dataset by:

```bash
python -m tevatron.driver.train \
  --do_train \
  --output_dir model_scifact \
  --dataset_name Tevatron/scifact \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5
```

Then we can encode corresponding self-contained corpus by:
```bash
python tevatron.driver.encode \
  --do_encode \
  --output_dir=temp_out \
  --model_name_or_path model_scifact \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/scifact-corpus \
  --p_max_len 512 \
  --encoded_save_path corpus_emb.pkl
```

And encode corresponding self-contained topics by:
```bash
python tevatron.driver.encode \
  --do_encode \
  --output_dir=temp_out \
  --model_name_or_path model_scifact \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/scifact/dev \
  --encode_is_qry \
  --q_max_len 64 \
  --encoded_save_path queries_emb.pkl 
```

## Custom dataset
To use custom dataset with Tevatron, there are two ways:
### 1. Raw data
The first method is to prepare dataset in the same format as one of the above two dataset types.
- If the dataset was prepared in the `Relevancy Judged` format, then we can directly use the data load process
defined by `Tevatron/msmarco-passage`.  
- If the dataset was prepared in the `Exactly Match` format, then we can directly use the data load process
defined by `Tevatron/wikipedia-nq`.

For example, if we have prepared a dataset in Exactly Match format (same as `Tevatron/wikipedia-nq`), with:
- train data: `train_dir/train_data.jsonl`
- dev data: `dev_dir/dev_data.jsonl`
- corpus: `corpus_dir/corpus_jsonl`

We can train by:
```bash
python -m tevatron.driver.train \
  ... \
  --dataset_name Tevatron/wikipedia-nq \
  --train_dir train_dir \
  ...
```

Then we can encode corpus by:
```bash
python tevatron.driver.encode \
  ... \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encode_in_path corpus_dir/corpus_jsonl \
  ...
```

And encode query by:
```bash
python tevatron.driver.encode \
  ... \
  --dataset_name Tevatron/wikipedia-nq \
  --encode_in_path dev_dir/dev_data.jsonl \
  --encode_is_qry \
  ...
```
> Note: we use `...` here to hide the arguments that irrelevant to dataset setting for a more clear comperision.
> Please see training and encoding document for detailed arguments.


### 2. Pre-tokenized data
Tevatron also accept pre-tokenized custom dataset.
By doing this, Tevatron will skip the tokenization step during training or encoding.

The datasets need to be crafted in the format below:
- Training: `jsonl` file with each line is a training instance,
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
```
- Encoding: `jsonl` file with each line is a piece of text to be encoded,
```
{text_id: "xxx", 'text': TEXT_TYPE}
```
The `TEXT_TYPE` here can be either `List[int]` (pre-tokenized) or `string` (non-pretokenized).
Here we encourage user to use pre-tokenized (i.e. `TEXT_TYPE=List[int]`) 
as `TEXT_TYPE=string` is not supported for some tokenizer.

To use custom data in pre-tokenized format, use `--dataset_name json` (or leave it as empty)
during training and encoding.

For example, if we have prepared a pre-tokenized dataset, with:
- train data: `train_dir/train_data.jsonl`
- dev data: `dev_dir/dev_data.jsonl`
- corpus: `corpus_dir/corpus_jsonl`

We can train by:
```bash
python -m tevatron.driver.train \
  ... \
  --train_dir train_dir \
  ...
```

Then we can encode corpus by:
```bash
python tevatron.driver.encode \
  ... \
  --encode_in_path corpus_dir/corpus_jsonl \
  ...
```

And encode query by:
```bash
python tevatron.driver.encode \
  ... \
  --encode_in_path dev_dir/dev_data.jsonl \
  --encode_is_qry \
  ...
```
> Note: we use `...` here to hide the arguments that irrelevant to dataset setting for a more clear comperision.
> Please see training and encoding document for detailed arguments.
