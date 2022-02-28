# Examples
Here we provide examples for running tevatron on various datasets/models.

## Example Index
- [Web Retrieaval - MS-MARCO passage ranking](msmarco-passage-ranking)
- [DPR - Natural Question & TriviaQA](dpr)
- [Beyond BERT - Condenser/coCondenser on MS-MARCO](coCondenser-marco)
- [Beyond BERT - Condenser/coCondenser on Natural Question](coCondenser-nq)
- [Large Batch Training with Limited Memory - Gradiend Cache](gradient-cache.md)
- [Scifact - scientific fact retrieval](scifact)

## Research
Researchers are recommended to start with the [run.py](run.py) under this directory. It includes logics in `tevatron.driver.train` and `tevatron.driver.encode` for training and encoding. 
Adjustments can then be made into `tevatron.modeling`, `tevatron.trainer` and `tevatron.data`; either create sub-classes or make direct edits.

In particular,
- better models can go into `tevatron.modeling`
- better training technique can go into `tevatron.trainer`
- better data control go into `tevatron.data`

To change retriever behaviors, check out its [main function](../src/tevatron/faiss_retriever/__main__.py), 
and also the entire `faiss_retriever` [submodule](../src/tevatron/faiss_retriever). 


## Checkpoints Validation
All the DR checkpoints generated during Tevatron training (checkpoints saved in `--output_dir`) can be automatically evaluated on the retrieval task by using [Asyncval](https://github.com/ielab/asyncval) toolkit with another GPU. 

If you want to use this feature, then install Asyncval by: `pip install asyncval`

After the installation, you can simply run the following command line to kick-off validation:

```
python -m asyncval \
	--query_file List[str] \
	--candidate_file str \
	--ckpts_dir str \
	--tokenizer_name_or_path str \
	--qrel_file str \
	--output_dir str
```
where `--query_file` is the path to query JSON file; `--candidate_file` is the path to the folder that stores corpus JSON splits; `--ckpts_dir` is the folder that saves checkpoints; `--tokenizer_name_or_path` is your DR tokenizer; `--qrel_file` is the path to the TREC standard qrel file; `--output_dir` is the path to the folder that saves run files of checkpoints.

Asyncval also supports different evaluation metrics and commonly used loggers (e.g. Tensorboard and WandB), also the corpus subset sampling methods for fast validation. We refer to the instructions in the original [repository](https://github.com/ielab/asyncval) for more advanced features.