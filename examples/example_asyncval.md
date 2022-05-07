# Acyncval validation example
[Asyncval](https://github.com/ielab/asyncval) is a DR validation toolkit that can asynchronously validating dense retriever checkpoints during training. All the DR checkpoints generated during Tevatron training (checkpoints saved in `--output_dir`) can be automatically evaluated on the retrieval task by using Asyncval toolkit with another GPU. 

If you want to use this feature, you can install Asyncval by: `pip install asyncval`

After the installation, you can simply run the following command line to kick-off validation:

```
python -m asyncval \
	--query_file List[str] \
	--candidate_dir str \
	--ckpts_dir str \
	--tokenizer_name_or_path str \
	--qrel_file str \
	--output_dir str
```
where `--query_file` is the path to query JSON file; `--candidate_dir` is the path to the folder that stores corpus JSON splits; `--ckpts_dir` is the folder that saves checkpoints; `--tokenizer_name_or_path` is your DR tokenizer; `--qrel_file` is the path to the TREC standard qrel file; `--output_dir` is the path to the folder that saves run files of checkpoints.

Asyncval also supports different IR evaluation metrics and commonly used loggers (e.g. Tensorboard and WandB), also the corpus subset sampling methods for fast validation. We refer to the instructions in the original [repository](https://github.com/ielab/asyncval) for more advanced features.

For a more concrete example, you can check out this [link](https://github.com/ielab/asyncval/tree/main/examples/msmarco-passage) which contains the instructions for using Asyncval to validate Tevatron checkpoints on the MS MARCO passage ranking task.
