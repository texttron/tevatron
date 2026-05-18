# Training

## Basic Training

To train a dense retriever, run the module `tevatron.retriever.driver.train`.

Here we use Natural Questions (nq) as an example.

We train on a machine with 4× V100 GPUs. If GPU memory is limited, use [GradCache](#gradcache).

```bash
torchrun --nproc_per_node=4 -m tevatron.retriever.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_group_size 2 \
  --learning_rate 1e-5 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --num_train_epochs 40
```

Here we are using our self-contained datasets to train, using Hugging Face `datasets` loading: `--dataset_name` selects a hub dataset. For local JSON/JSONL, set `--dataset_name json` (or the default) and pass files via `--dataset_path` (see [datasets.md](datasets.md)).

> We use `bert-base-uncased` from the Hugging Face Hub and `--fp16` AMP where appropriate. Extra model options live on `ModelArguments` (for example `--pooling`, `--normalize`, `--lora`). The full CLI is `python -m tevatron.retriever.driver.train --help`. Shared dataclasses are defined in [`tevatron.retriever.arguments`](../src/tevatron/retriever/arguments.py).

## GradCache

Tevatron can use gradient caching to train with a large effective batch on limited GPU memory.

> Described in [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983).

Enable it with:

- `--grad_cache`
- `--gc_q_chunk_size`: sub-batch size for queries
- `--gc_p_chunk_size`: sub-batch size for passages

Example (single GPU, large batch):

```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 128 \
  --train_group_size 2 \
  --learning_rate 1e-5 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --num_train_epochs 40 \
  --grad_cache \
  --gc_q_chunk_size 32 \
  --gc_p_chunk_size 16
```

GradCache also works with multi-GPU `torchrun` setups.

## Training with TPU
Tevatron implements TPU training via Jax/Flax.
We provide a separate module `tevatron.driver.jax_train` to train on TPU.
The arguments managements aligns with above Pytorch training driver.

By running the following commands on a V3-8 TPU VM is equivalent to the commands above.
```bash
python -m tevatron.driver.jax_train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --per_device_train_batch_size 16 \
  --train_group_size 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40
```
> Note that our Jax training driver also support gradient cache by adding `--grad_cache` option.

## Arguments Description
Our Argument parser inherits from `TrainingArguments` from HuggingFace's `transformers`.
For the common use training arguments such as learning rate and batch size configuration, 
please check [document](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) from HuggingFace for details.

Here we describe the details of the arguments additionally defined for Tevatron's CLI

| name                      | description                                                                                                                                                          | type   | default                      | supported driver |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|------------------------------|------------------|
| `do_train`                | Whether to run training                                                                                                                                              | `bool` | required                     | pytorch, jax     |
| `model_name_or_path`      | Model backbone to initialize dense retriever.  It can be either a model name that avaliable in Huggingface model hub. Or a path to a model directory                 | `str`  | required                     | pytorch, jax     |
| `tokenizer_name`          | Tokenizer name or path if not the same as `model_name_or_path`                                                                                                       | `str`  | same as `model_name_or_path` | pytorch, jax     |
| `cache_dir`               | Path to the directory to save the cache of models and datasets                                                                                                       | `str`  | `~/.cache/`                  | pytorch, jax     |
| `untie_encoder`           | Whether query encoder and passage encoder share same parameter                                                                                                       | `bool` | `False`                      | pytorch, jax     |
| `add_pooler`              | Whether add pooler on top of last layer output                                                                                                                       | `bool` | `False`                      | pytorch          |
| `projection_in_dim`       | The input dim of pooler                                                                                                                                              | `int`  | `768`                        |                  |
| `projection_out_dim`      | The output dim of pooler                                                                                                                                             | `int`  | `768`                        | pytorch          |
| `dataset_name`            | Dataset name that avaliable on HuggingFace                                                                                                                           | `str`  | `json`                       | pytorch, jax     |
| `train_dir`               | Directory that stores custom training data                                                                                                                           | `str`  | `None`                       | pytorch, jax     |
| `dataset_proc_num`        | Number of threads to use to preprocess/tokenize data                                                                                                                 | `int`  | `12`                         | pytorch, jax     |
| `train_group_size`        | Number of passages for each anchor query during training. It will load 1 positive passage + (`train_group_size`-1) negative passages for each example during training | `int`  | `8`                          | pytorch, jax     |
| `passage_field_separator` | The token to seperate `title` and `text` field for passages                                                                                                          | `str`  | `" "`                        | pytorch          |
| `query_max_len`               | Maximum query length                                                                                                                                                 | `int`  | `32`                         | pytorch, jax     |
| `passage_max_len`               | Maximum passage length                                                                                                                                               | `int`  | `128`                        | pytorch, jax     |
| `grad_cache`              | Whether use gradient cache feature. This can be used to support large batch size while GPU/TPU memory are limited.                                                   | `bool` | `False`                      | pytorch, jax     |
| `gc_q_chunk_size`         | Sub-batch size for queries with `grad_cache`                                                                                                                         | `int`  | `4`                          | pytorch, jax     |
| `gc_p_chunk_size`         | Sub-batch size for passages with  `grad_cache`                                                                                                                       | `int`  | `32`                         | pytorch, jax     |