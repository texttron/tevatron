# Tevatron
Tevatron is a simple and efficient toolkit for training and running dense retrievers with deep language models. The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast development and testing. A set of easy-to-use interfaces to Huggingfac's state-of-the-art pre-trained transformers ensures Tevatron's superior performance.

*Tevatron is currently under initial development stage. We will be actively adding new features and API changes may happen. Suggestions, feature requests and PRs are welcomed.*

## Features
- Command line interface for dense retriever training/encoding and dense index search.
- Flexible and extendable Pytorch retriever models. 
- Highly efficient Trainer, a subclass of  Huggingface Trainer, that naively support training performance features like mixed precision and distributed data parallel.
- Fast and memory-efficient train/inference data access based on memory mapping with Apache Arrow through Huggingface datasets.

## Installation
First install neural network and similarity search backends, namely Pytorch and FAISS. Check out the official installation guides for [Pytorch](https://pytorch.org/get-started/locally/#start-locally) and for [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Then install Tevatron with pip,
```bash
pip install tevatron
```

Or typically for develoment/research, clone this repo and install as editable,
```
git https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
```

> Note: The current code base has been tested with, `torch==1.8.2`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`


## Data Format
Training: Each line of the the Train file is a training instance,
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
...
```
Inference/Encoding: Each line of the the encoding file is a piece of text to be encoded,
```
{text_id: "xxx", 'text': TEXT_TYPE}
...
```
Here `TEXT_TYPE` can be either raw string or pre-tokenized ids, i.e. `List[int]`. Using the latter can help lower data processing latency during training to reduce/eliminate GPU wait. **Note**: the current code requires text_id of passages/contexts to be convertible to integer, e.g. integers or string of integers.

## Training (Simple)
To train a simple dense retriever, call the `tevatron.driver.train` module,
```
python -m tevatron.driver.train \  
  --output_dir $OUTDIR \  
  --model_name_or_path bert-base-uncased \  
  --do_train \  
  --save_steps 20000 \  
  --train_dir $TRAIN_DIR \
  --fp16 \  
  --per_device_train_batch_size 8 \  
  --learning_rate 5e-6 \  
  --num_train_epochs 2 \  
  --dataloader_num_workers 2
```
Here we picked `bert-base-uncased` BERT weight from Huggingface Hub and turned on AMP with `--fp16` to speed up training. Several command flags are provided in addition to configure the learned model, e.g. `--add_pooler` which adds an linear projection. A full list command line arguments can be found in `tevatron.arguments`.

## Training (Research)
Check out the [run.py](examples/run.py) in examples directory for a fully configurable train/test loop. Typically you will do,
```
from tevatron.modeling import DenseModel
from tevatron.trainer import DenseTrainer as Trainer

...
model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
...
trainer.train()
```


## Encoding
To encode, call the `tevatron.driver.encode` module. For large corpus, split the corpus into shards to parallelize.
```
for s in shard1 shar2 shard3
do
python -m tevatron.driver.encode \  
  --output_dir=$OUTDIR \  
  --tokenizer_name $TOK \  
  --config_name $CONFIG \  
  --model_name_or_path $MODEL_DIR \  
  --fp16 \  
  --per_device_eval_batch_size 128 \  
  --encode_in_path $CORPUS_DIR/$s.json \  
  --encoded_save_path $ENCODE_DIR/$s.pt
done
```
## Index Search
Call the `tevatron.faiss_retriever` module,
```
python -m tevatron.faiss_retriever \  
--query_reps $ENCODE_QRY_DIR/qry.pt \  
--passage_reps $ENCODE_DIR/'*.pt' \  
--depth $DEPTH \
--batch_size -1 \
--save_text \
--save_ranking_to rank.tsv
```
Encoded corpus or corpus shards are loaded based on glob pattern matching of argument `--passage_reps`. Argument `--batch_size` controls number of queries passed to the FAISS index each search call and `-1` will pass all queries in one call. Larger batches typically run faster (due to better memory access patterns and hardware utilization.) Setting flag `--save_text` will save the ranking to a tsv file with each line being `qid pid score`.

Alternatively paralleize search over the shards,
```
for s in shard1 shar2 shard3
do
python -m tevatron.faiss_retriever \  
--query_reps $ENCODE_QRY_DIR/qry.pt \  
--passage_reps $ENCODE_DIR/$s.pt \  
--depth $DEPTH \  
--save_ranking_to $INTERMEDIATE_DIR/$s
done
```
Then combine the results using the reducer module,
```
python -m tevatron.faiss_retriever.reducer \  
--score_dir $INTERMEDIATE_DIR \  
--query $ENCODE_QRY_DIR/qry.pt \  
--save_ranking_to rank.txt  
```

## Contacts
If you have a toolkit specific question, feel free to open an issue. 

You can also reach out to us for general comments/suggestions/questions through email.
- Luyu Gao luyug@cs.cmu.edu
- Xueguang Ma x93ma@uwaterloo.ca
