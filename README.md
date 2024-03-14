# Tevatron V2
Tevatron aims to provide a flexible and efficient toolkit that enables training and inference for neural retrieval models at scale.

> Some of the features in Tevatron v1 is not yet migrated to Tevatron v2. We are working on it.
> If you are looking for the Tevatron v1 features, please pull the [v1 branch](https://github.com/texttron/tevatron/tree/tevatron-v1).

## Features
- Training billion-scale LLM neural retriever on GPUs and TPUs.
- Parameter efficient tuning with LoRA.
- Integration with DeepSpeed, flash attention, gradient accumulation, and other efficient training techniques.
- Self-contained datasets for neural retrieval and open-domain QA tasks.
- Direct loading and finetuning SoTA pre-trained models (BGE-Embbedding, Instruct-E5) from HuggingFace.

## Installation

<details><summary><b>PyTorch (GPU)</b></summary>

0. Clone the repository.
1. Install PyTorch based on your CUDA version from [PyTorch](https://pytorch.org/get-started/locally/).
2. Install dependencies and Tevatron.
```bash
pip install transformers datasets peft
pip install deepspeed accelerate
pip install faiss
pip install -e .
```


</details>
<details><summary><b>JAX (TPU)</b></summary>

0. Clone the repository.
1. Install JAX by following the [official guide](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-google-cloud-tpu)
2. Install dependencies
```bash
pip install transformers datasets
pip install flax optax
```
3. Install Magix and GradCache
```bash
git clone https://github.com/luyug/magix.git
cd magix && pip install -e . && cd ..
git clone https://github.com/luyug/GradCache.git
cd GradCache && pip install -e . && cd ..
```

4. Install Tevatron
```bash
pip install -e .
```

</details>
<details><summary><b>JAX (GPU)</b></summary>

To run the JAX implementation of Tevatron on GPU, we encourage using the jax-toolbox [jax container](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) image from NVIDIA.

Below is a Dockerfile example to set up Tevatron on top of the jax container.
```Dockerfile
FROM ghcr.io/nvidia/jax:jax-2024-03-08

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir transformers sentencepiece simple_parsing datasets orbax==0.4.8 && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/luyug/magix.git && \
    cd magix && pip install -e . && cd .. && \
    git clone https://github.com/luyug/GradCache.git \
    cd GradCache && pip install -e . && cd .. \
    git clone https://github.com/texttron/tevatron.git && \
    cd tevatron && pip install -e .
```




</details>



## Tevatron 101
In this example, we will demonstrate how to use Tevatron to LoRA fine-tune a Mistral-7B model on the MSMARCO passage dataset. The obtained LLM Retriever is expected to have `MRR@10=42.3` on the MS MARCO dev set with straightforward training.

<details><summary><b>Data Preparation</b></summary>

Tevatron takes training or inference data in `jsonl` format with each line organized as a json object as follows:
### 1. Training Data
```json
{
   "query_id": "<query id>",
   "query": "<query text>",
   "positive_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"},
     ...
   ],
   "negative_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"},
     ...
   ]
}
```
where the passages in `positive_passages` are the annotated relevant passages of the `query` 
and passages in `negative_passages` are usually non-relevant (hard negative) passages from top results of a retrieval system (e.g. BM25, DPR). Additional fields such as `answers` for QA datasets can be included as well.

#### 2. Corpus Data
```json
{
   "docid": "<passage id>",
   "title": "<passage title>",
   "text": "<passage body>"
}
```
where each line represents a passage in the corpus.

### Self-Contained Dataset
Tevatron self-contained several commonlly used datasets for neural retrieval. 
(via [HuggingFace](https://huggingface.co/Tevatron)).
These datasets can downloaded automatically during training and encoding
by setting `--dataset_name <hgf dataset name>`.

In this example, we will use the self-contained dataset `Tevatron/msmarco-passage-aug` for training, whose hard negative passages are sampled from the mix of top200 BM25 and top200 CoCondenser results.

</details>


<details><summary><b>Run with PyTorch (GPU)</b></summary>

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

<details><summary><b>Run with JAX (TPU/GPU)</b></summary>

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

The above training setting tooks about 35 hours on a v4-8 TPU VM.

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
