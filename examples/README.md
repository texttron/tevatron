# Examples
Here we provides examples for running Dense on various datasets/models.

## Research
Researchers are recommended to start with the [run.py](run.py) under this directory. It includes logics in `dense.driver.train` and `dense.driver.encode` for training and encoding. 
Adjustments can then be made into `dense.modeling`, `dense.trainer` and `dense.data`; either create sub-classes or make direct edits.

In particular,
- better models can go into `dense.modeling`
- better training technique can go into `dense.trainer`
- better data control go into `dense.data`

## Example Index
- [MS-MARCO passage ranking](msmarco-passage-ranking)
