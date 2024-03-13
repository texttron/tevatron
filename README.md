# Tevatron V2
Tevatron aims to provide a flexible and efficient toolkit that enables training and inference for neural retrieval models at scale.

> Some of the features in Tevatron v1 is not yet migrated to Tevatron v2. We are working on it.
> If you are looking for the Tevatron v1 features, please pull the [v1 branch]().

## Features
- Training billion-scale LLM neural retriever on GPUs and TPUs.
- Parameter efficient tuning with LoRA.
- Integration with DeepSpeed, flash attention, gradient accumulation, and other efficient training techniques.
- Self-contained datasets for neural retrieval and open-domain QA tasks.
- Direct loading and finetuning SoTA pre-trained models (BGE-Embbedding, Instruct-E5) from HuggingFace.

## Installation

<details><summary><b>PyTorch (GPU)</b></summary></details>
<details><summary><b>JAX (TPU)</b></summary></details>
<details><summary><b>JAX (GPU)</b></summary></details>



## Toolkit Usage

<details><summary><b>PyTorch (GPU)</b></summary>

### Training

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
  --passage_max_len 156 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4
```

In batch passages per query: 8x4x16 = 512

Number of queries per update: 8x4x4 = 128

The above training setting tooks about 70 hours on 4xA6000 GPU.

Equivalent training tooks about 110 hours on 1xA100 GPU.



### Encoding

#### Query Encoding
```bash
EMBEDDING_OUTPUT_DIR=<folder to save query embedding>
CUDA_VISIBLE_DEVICES=4 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_name_or_path retriever-mistral \
  --lora \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/query-dev.pkl
```

#### Corpus Encoding
```bash
EMBEDDING_OUTPUT_DIR=<folder to save query embedding>
for s in 0 1 2 3
do
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_name_or_path retriever-mistral \
  --lora \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.${s}.pkl
done
```
> add & to the end of the command to run in the background in parallel.

### Retrieval
```bash
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-dev.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus*.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.dev.txt
```

The output file is in the format of `<query_id> <passage_id> <score>` in each line.

</details>

<details><summary><b>Jax (TPU/GPU)</b></summary>

### Training

> For GPU training, set `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` and make sure the query and passage length are multiples of 64 if TransformersEngine is installed.

```bash
python -m tevatron.tevax.experimental.mp.train_lora  \
   --checkpoint_dir retriever-mistral-jax \
   --train_file Tevatron/msmarco-passage-aug \
   --model_name mistralai/Mistral-7B-v0.1 \
   --model_type mistral \
   --batch_size 128 \
   --num_target_passages 16 \
   --learning_rate 1e-4 \
   --seed 12345 \
   --mesh_shape 1 -1 \
   --weight_decay 0.00001 \
   --num_epochs 1 \
   --max_query_length 64 \
   --max_passage_length 128 \
   --pooling eos \
   --scale_by_dim True \
   --grad_cache \
   --passage_num_chunks 32 \
   --query_num_chunks 4
```

In batch passages per query: 128x16 = 2048

Number of queries per update: 128

The above training setting tooks about 42 hours on a v4-8 TPU VM.

Equivalent training tooks about 80 hours on 1xA100 GPU.

### Encoding

#### Query Encoding
```bash
python -m tevatron.tevax.experimental.mp.encode  \
   --model_type mistral \
   --model_name_or_path mistralai/Mistral-7B-v0.1 \
   --model_config_name_or_path mistralai/Mistral-7B-v0.1 \
   --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
   --dataset_name_or_path Tevatron/msmarco-passage \
   --split dev \
   --output_dir $EMBEDDING_OUTPUT_DIR/query-embedding \
   --batch_size 32 \
   --input_type query \
   --max_seq_length 64 \
   --mesh_shape 1 -1 \
   --lora retriever-mistral-jax/lora \
   --scale_by_dim
```

#### Corpus Encoding
```bash
python -m tevatron.tevax.experimental.mp.encode  \
   --model_type mistral \
   --model_name_or_path mistralai/Mistral-7B-v0.1 \
   --model_config_name_or_path mistralai/Mistral-7B-v0.1 \
   --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
   --dataset_name_or_path Tevatron/msmarco-passage-corpus \
   --output_dir $EMBEDDING_OUTPUT_DIR/corpus-embedding \
   --batch_size 32 \
   --input_type passage \
   --max_seq_length 128 \
   --mesh_shape 1 -1 \
   --lora retriever-mistral-jax/lora \
   --scale_by_dim
```

### Retrieval
```bash
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-embedding/*.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/corpus-embedding/*.pkl \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/run.dev.txt
```

The output file is in the format of `<query_id> <passage_id> <score>` in each line.

</details>


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


## Acknowledgement

* We thank all the contributors of dependency libraries.
* We thank Google's [TPU research cloud](https://sites.research.google/trc/about/) for providing TPU resources.
