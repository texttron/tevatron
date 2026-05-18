# Datasets

## Dataset types

There are usually two types of dataset format for dense retrieval training, depending on whether document relevance is human-judged or decided by answer exact matching.

### 1. Relevance-judged dataset

If passages are labeled relevant or not (e.g. MS MARCO passage ranking), an instance often looks like:

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

`positive_passages` are annotated relevant documents to the `query`; `negative_passages` are usually hard negatives from the top results of a retriever (e.g. BM25).

### 2. Exact matching dataset

If relevance is defined by an answer span (e.g. Natural Questions), an instance often looks like:

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

`positive_passages` should contain text that matches an answer string; negatives are typically top-k retriever results that do not.

## Self-contained datasets

Tevatron self-contains common retrieval benchmarks on the Hugging Face Hub under the [Tevatron](https://huggingface.co/Tevatron) org. They are downloaded and cached automatically when you set `--dataset_name` during training or encoding.

| Dataset      | Hugging Face name            | Type             |
|--------------|-------------------------------|------------------|
| MS MARCO     | `Tevatron/msmarco-passage`   | Relevance judged |
| SciFact      | `Tevatron/scifact`           | Relevance judged |
| NQ           | `Tevatron/wikipedia-nq`      | Answer overlap   |
| TriviaQA     | `Tevatron/wikipedia-trivia`  | Answer overlap   |
| WebQuestions | `Tevatron/wikipedia-wq`      | Answer overlap   |
| CuratedTREC  | `Tevatron/wikipedia-curated` | Answer overlap   |
| SQuAD        | `Tevatron/wikipedia-squad`   | Answer overlap   |

> Self-contained training splits include BM25 negatives by default.

### SciFact example

Train:

```bash
python -m tevatron.retriever.driver.train \
  --do_train \
  --output_dir model_scifact \
  --dataset_name Tevatron/scifact \
  --model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 5
```

Encode corpus:

```bash
python -m tevatron.retriever.driver.encode \
  --output_dir temp_out \
  --model_name_or_path model_scifact \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/scifact-corpus \
  --passage_max_len 512 \
  --encode_output_path corpus_emb.pkl
```

Encode queries (dev split):

```bash
python -m tevatron.retriever.driver.encode \
  --output_dir temp_out \
  --model_name_or_path model_scifact \
  --per_device_eval_batch_size 64 \
  --dataset_name Tevatron/scifact \
  --dataset_split dev \
  --encode_is_query \
  --query_max_len 64 \
  --encode_output_path queries_emb.pkl
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
python -m tevatron.retriever.driver.train \
  ... \
  --dataset_name Tevatron/wikipedia-nq \
  --dataset_path train_dir/train_data.jsonl \
  ...
```

Encode corpus (reuse NQ *corpus* schema, local file):

```bash
python -m tevatron.retriever.driver.encode \
  ... \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --dataset_path corpus_dir/corpus.jsonl \
  ...
```

Encode dev queries:

```bash
python -m tevatron.retriever.driver.encode \
  ... \
  --dataset_name Tevatron/wikipedia-nq \
  --dataset_path dev_dir/dev_data.jsonl \
  --encode_is_query \
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

- train: `train_dir/train_data.jsonl`
- dev: `dev_dir/dev_data.jsonl`
- corpus: `corpus_dir/corpus.jsonl`

Train:

```bash
python -m tevatron.retriever.driver.train \
  ... \
  --dataset_path train_dir/train_data.jsonl \
  ...
```

Encode corpus:

```bash
python -m tevatron.retriever.driver.encode \
  ... \
  --dataset_path corpus_dir/corpus.jsonl \
  ...
```

Encode queries:

```bash
python -m tevatron.retriever.driver.encode \
  ... \
  --dataset_path dev_dir/dev_data.jsonl \
  --encode_is_query \
  ...
```

> `...` contains model path, batch sizes, lengths, `encode_output_path`, and other flags from the training/encoding docs.
