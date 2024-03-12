# Tevatron
Tevatron is a simple and efficient toolkit for training and running dense retrievers with deep language models. 
The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast
development and testing. A set of easy-to-use interfaces to Huggingface's state-of-the-art pre-trained transformers
ensures Tevatron's superior performance.

*Tevatron is currently under initial development stage. We will be actively adding new features and API changes
may happen. Suggestions, feature requests and PRs are welcomed.*

## Features
- Command line interface for dense retriever training/encoding and dense index search.
- Flexible and extendable Pytorch retriever models. 
- Highly efficient Trainer, a subclass of  Huggingface Trainer, that naively support training performance features like mixed precision and distributed data parallel.
- Fast and memory-efficient train/inference data access based on memory mapping with Apache Arrow through Huggingface datasets.
- Jax/Flax training/encoding on TPU

## Installation

## Usage-PyTorch (GPU)


<details><h3><summary>Training</summary></h3>

<details><summary><h4>Mistral-7B</h4></summary>

```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-mistral \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --save_steps 50 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 8 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4
```

In batch passages per query: 8x4x16 = 512

Number of queries per update: 8x4x4 = 128

The training tooks about 70 hours on 4xA6000 GPU.

Equivalent training tooks about 110 hours on 1xA100 GPU.

</details>

<details><summary><h4>BERT</h4></summary></details>

```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir retriever-bert \
  --model_name_or_path bert-base-uncased \
  --dataset_name Tevatron/msmarco-passage \
  --bf16 \
  --pooling cls \
  --per_device_train_batch_size 32 \
  --train_group_size 16 \
  --learning_rate 1e-5 \
  --query_max_len 32 \
  --passage_max_len 128 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --overwrite_output_dir
```

</details>

## Usage-JAX (TPU/GPU)

## Retrieval


## Citation
If you find Tevatron helpful, please consider citing our [paper](https://arxiv.org/abs/2203.05765).
```
@article{Gao2022TevatronAE,
  title={Tevatron: An Efficient and Flexible Toolkit for Dense Retrieval},
  author={Luyu Gao and Xueguang Ma and Jimmy J. Lin and Jamie Callan},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.05765}
}
```

## Contacts
If you have a toolkit specific question, feel free to open an issue. 

You can also reach out to us for general comments/suggestions/questions through email.
- Luyu Gao luyug@cs.cmu.edu
- Xueguang Ma x93ma@uwaterloo.ca
