# Training

## Basic Training
To train a simple dense retriever, call the `tevatron.driver.train` module.

Here we use Natural Questions as example.

We train on a machine with 4xV100 GPU, if the GPU resources are limited for you, please train with gradient cache.

```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --negatives_x_device
```


Here we are using our self-contained datasets to train. 
To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq` by 
`--train_dir <train data dir>`, (see here for details).

>Here we picked `bert-base-uncased` BERT weight from Huggingface Hub and turned 
> on AMP with `--fp16` to speed up training. Several command flags are provided 
> in addition to configure the learned model, e.g. `--add_pooler` which adds an 
> linear projection. A full list command line arguments can be found in 
> `tevatron.arguments`.


## GradCache
Tevatron adopts gradient cache technique to allow large batch training of dense retriever on memory limited GPU.

> Details is described in paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup
](https://arxiv.org/abs/2101.06983).

Adding following three flags to training command to enable gradient cache:
- `--grad_cache`: enable gradient caching
- `--gc_q_chunk_size`: sub-batch size for query 
- `--gc_p_chunk_size`: sub-batch size for passage

For example, the following command can train dense retrieval model for Natural Question in 128 batch size
but only with one GPU.
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 128 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --grad_cache \
  --gc_q_chunk_size 32 \
  --gc_p_chunk_size 16
```
> Notice that GradCache also support multi-GPU setting.
