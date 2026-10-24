# Tevatron

<div align="center">
<a href="https://arxiv.org/abs/2608.00916" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/Tevatron" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=red"></a>
<a href="https://pepy.tech/projects/tevatron"><img src="https://static.pepy.tech/badge/tevatron" alt="PyPI Downloads"></a>
<a href="https://star-history.com/#texttron/tevatron"> <img src="https://img.shields.io/github/stars/texttron/tevatron?style=social" alt="GitHub stars"> </a>
<!--   -->
</div>

Tevatron: Unified Document Retrieval Toolkit across Scale, Language, and Modality.

> For legacy Tevatron v1 features, see the [v1 branch](https://github.com/texttron/tevatron/tree/tevatron-v1).

## Features

- Training billion-scale LLM neural retriever on GPUs and TPUs.
- Parameter efficient tuning with LoRA.
- Integration with vLLM, DeepSpeed, FlashAttention, gradient accumulation, and other efficient training and inference techniques.
- Self-contained [huggingface datasets](https://huggingface.co/Tevatron) for multi-modal and multilingual neural retrieval and open-domain QA tasks.
- Direct loading and finetuning SoTA pre-trained models (BGE-Embbedding, Instruct-E5) from HuggingFace.
- LLM reranker training with Megatron-Core, including expert parallelism for MoE models and support for full-parameter tuning and LoRA.
- Unified retrieval and reranker evaluation with Hugging Face and vLLM backends, plus offline teacher annotation for distillation.

## Installation

For Megatron reranker training, see the [installation guide](./docs/megatron_installation.md) and [environment notes](./docs/environments.md).

<details><summary><b>PyTorch (GPU)</b></summary>

0. Clone the repository.
1. Install PyTorch based on your CUDA version from [PyTorch](https://pytorch.org/get-started/locally/).
2. Install dependencies and Tevatron.
```bash
pip install git+https://github.com/luyug/GradCache.git
pip install transformers datasets peft
pip install deepspeed accelerate
pip install faiss-cpu
pip install pillow qwen-omni-utils
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
   "query_text": "<query text>",
   "query_image": "<query image>",
   "positive_document_ids": ["<passage id>", ...],
   "negative_document_ids": ["<passage id>", ...],
}
```
where the passages in `positive_passages` are the annotated relevant passages of the `query` 
and passages in `negative_passages` are usually non-relevant (hard negative) passages from top results of a retrieval system (e.g. BM25, DPR). Additional fields such as `answers` for QA datasets can be included as well.

#### 2. Corpus Data
```json
{
   "docid": "<document id>",
   "document_text": "<document text>",
   "document_image": "<document image>",
}
```
where each line represents a document in the corpus. 

Note that the image field for both training and corpus data are optional and can be omitted (i.e., pure textual modality retrieval).

### Self-Contained Dataset
Tevatron self-contained several commonlly used datasets for neural retrieval. 
(via [HuggingFace](https://huggingface.co/Tevatron)).
These datasets can downloaded automatically during training and encoding
by setting `--dataset_name <hgf dataset name>`.

In this example, we will use the self-contained dataset `Tevatron/msmarco-passage-aug` for training, whose hard negative passages are sampled from the mix of top200 BM25 and top200 CoCondenser results.

</details>


<details><summary><b>Run with PyTorch (GPU)</b></summary>

> Some examples may be outdated; use each command's `--help` output as the reference for current CLI options.

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
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir temp \
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

## Megatron Training Backend and Reranker Support

Tevatron supports cross-encoder reranker training through Hugging Face Trainer and Megatron-Core. The Megatron backend adds tensor, pipeline, and expert parallelism for dense and MoE LLM rerankers, with full-parameter tuning, LoRA, and offline distillation. It exports Hugging Face-compatible checkpoints for inference and evaluation.

- **Training recipes:** see the [Megatron reranker examples](./examples/megatron/README.md) and [configuration reference](./docs/megatron_reranker_usage.md), or the [Hugging Face reranker example](./examples/reranker/README.md).
- **Setup and dependencies:** see the [Megatron installation guide](./docs/megatron_installation.md) and [environment notes](./docs/environments.md), which explain the pinned [Megatron](./requirements-megatron.txt) and [LoRA](./requirements-lora.txt) dependency manifests.
- **Inference and evaluation:** see the [unified reranker evaluation guide](./src/tevatron/eval/README.md) for Hugging Face and vLLM workflows.

## vLLM Integration

We are actively ramping up vLLM integration. Here are the functionalities already supported:

- **Text embedding:** encode queries and documents with configurable pooling, normalization, and optional LoRA, then search with the existing FAISS driver. See the [RepLLaMA example](./examples/example_repllama_vllm.md).
- **Multimodal embedding:** encode text and image inputs with optional LoRA through [`tevatron.retriever.driver.vllm_encode_mm`](./src/tevatron/retriever/driver/vllm_encode_mm.py).
- **Offline reranking:** score causal-LM rerankers using yes/no log-probabilities, with Tevatron and Qwen3 reranker prompt templates, tensor parallelism, and optional LoRA. See the [reranking example](./src/tevatron/eval/README.md#2-rerank-then-score-one-checkpoint-one-dataset).
- **Persistent scoring servers:** keep models loaded and distribute benchmark evaluation requests across an HTTP scoring pool. See the [serving example](./src/tevatron/eval/README.md#3-rerank--score-a-full-sweep-over-a-persistent-backend-pool).
- **Offline teacher annotation:** precompute teacher reranker scores for distillation training. See the [annotation and distillation recipe](./examples/megatron/README.md#2-distillation-training-listwise-kl-vs-a-teacher).

Embedding drivers currently support one visible GPU per process; reranking supports tensor parallelism. See the [environment notes](./docs/environments.md#4-reranking--teacher-annotation-vllm--venv-eval) for the vLLM evaluation setup, and use each command's `--help` output for current CLI options.

## Examples

+ [Unified multi-modal and multilingual retrieval](./examples/multimodal)
+ [vLLM encoding and retrieval](./examples/example_repllama_vllm.md)
+ [Megatron reranker training and distillation](./examples/megatron/README.md)
+ [Unified retrieval and reranker evaluation](./src/tevatron/eval/README.md)
+ [Decoder-LM SPLADE / LACONIC](./examples/laconic/README.md) · [LACONIC repository](https://github.com/zhichaoxu-shufe/laconic-sparse-retrieval)
+ [Tevatron-Elastic](https://github.com/zhichaoxu-shufe/tevatron-elastic)

## Citation

If you find Tevatron helpful, please consider citing the relevant papers below. The [Tevatron 3.0 paper](https://arxiv.org/abs/2608.00916) is accepted to SIGIR-AP 2026.

```bibtex
@inproceedings{Gao2022TevatronAE,
  title={Tevatron: An efficient and flexible toolkit for neural retrieval},
  author={Gao, Luyu and Ma, Xueguang and Lin, Jimmy and Callan, Jamie},
  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={3120--3124},
  year={2023}
}
```

```bibtex
@inproceedings{ma2025tevatron,
  title={Tevatron 2.0: Unified document retrieval toolkit across scale, language, and modality},
  author={Ma, Xueguang and Gao, Luyu and Zhuang, Shengyao and Zhan, Jiaqi Samantha and Callan, Jamie and Lin, Jimmy},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={4061--4065},
  year={2025}
}
```

```bibtex
@misc{xu2026tevatronmeetsmegatronexpertparallel,
  title={Tevatron Meets Megatron: Expert-Parallel LLM Reranker Training on an Academic Budget},
  author={Zhichao Xu and Xueguang Ma and Shengyao Zhuang and Luyu Gao and Wenqian Ye and Yu Wang and Jamie Callan and Jimmy Lin},
  year={2026},
  eprint={2608.00916},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2608.00916},
}
```


## Contacts
If you have a toolkit specific question, feel free to open an issue. 

You can also reach out to us for general comments/suggestions/questions through email.
- Luyu Gao luyug@cs.cmu.edu
- Xueguang Ma x93ma@uwaterloo.ca
- Shengyao Zhuang shengyaozhuang@gmail.com


## Acknowledgement

* We thank all the contributors of dependency libraries.
* We thank Google's [TPU research cloud](https://sites.research.google/trc/about/) for providing TPU resources.
