# Examples
Here we provide examples for running tevatron on various datasets/models.

## Example Index
- [Web Retrieaval - MS-MARCO passage ranking](msmarco-passage-ranking)
- [DPR - Natural Question & TriviaQA](dpr)
- [Beyond BERT - Condenser/coCondenser on MS-MARCO](coCondenser-marco)
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
