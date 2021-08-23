# Examples
Here we provide examples for running Dense on various datasets/models.

## Research
Researchers are recommended to start with the [run.py](run.py) under this directory. It includes logics in `dense.driver.train` and `dense.driver.encode` for training and encoding. 
Adjustments can then be made into `dense.modeling`, `dense.trainer` and `dense.data`; either create sub-classes or make direct edits.

In particular,
- better models can go into `dense.modeling`
- better training technique can go into `dense.trainer`
- better data control go into `dense.data`

To change retriever behaviors, check out its [main function](../src/dense/faiss_retriever/__main__.py), 
and also the entire `faiss_retriever` [submodule](../src/dense/faiss_retriever). 
## Example Index
- [MS-MARCO passage ranking](msmarco-passage-ranking)
