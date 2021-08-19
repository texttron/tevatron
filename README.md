# Dense
Dense is a simple and efficient toolkit for training and running dense retrievers with deep language models. The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast development and testing. A set of easy-to-use interfaces to Huggingface🤗's state-of-the-art pre-trained transformers ensures Dense's superior performance.

*Dense is currently under initial development stage. We will be actively adding new features and API changes may happen.*

## Features
- Command line interface for dense retriever training/encoding and dense index search.
- Flexible and extendable Pytorch retriever models. 
- Highly efficient Trainer, a subclass of  Huggingface Trainer, that naively support training performance features like mixed precision and distributed data parallel.
- Fast and memory-efficient train/inference data access based on memory mapping with Apache Arrow through Huggingface datasets.

## Installation
First install the dependencies. The current code base has been testes with,
```
pytorch==1.8.0  
faiss-cpu==1.6.5  
transformers==4.2.0  
datasets==1.1.3
```
Then clone this repo and run pip.
```
git https://github.com/luyug/Dense
cd Dense
pip install .
```
Or typically for research, install as editable,
```
pip install --editable .
```

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
To train a simple dense retriever, call the `dense.driver.train` module,
```
python -m dense.driver.train \  
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
Here we picked `bert-base-uncased` BERT weight from Huggingface Hub and turned on AMP with `--fp16` to speed up training. Several command flags are provided in addition to configure the learned model, e.g. `--add_pooler` which adds an linear projection. A full list command line arguments can be found in `dense.arguments`.

## Training (Research)
Check out the [run.py](examples/run.py) in examples directory for a fully configurable train/test loop. Typically you will do,
```
from dense.modeling import DenseModel
from dense.trainer import DenseTrainer as Trainer

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
To encode, call the `dense.driver.encode` module. For large corpus, split the corpus into shards to parallelize.
```
for s in shard1 shar2 shard3
do
python -m dense.driver.encode \  
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
Search over the shards,
```
for s in shard1 shar2 shard3
do
python -m dense.faiss_retriever \  
--query_reps $ENCODE_QRY_DIR/qry.pt \  
--passage_reps $ENCODE_DIR/$s.pt \  
--depth $DEPTH \  
--save_ranking_to $INTERMEDIATE_DIR/$s
done
```
Then combine the results,
```
python -m dense.faiss_retriever.reducer \  
--score_dir $INTERMEDIATE_DIR \  
--query $ENCODE_QRY_DIR/qry.pt \  
--save_ranking_to rank.txt  
```
For a single shard, you can still run the reducer to get results in a text file.
