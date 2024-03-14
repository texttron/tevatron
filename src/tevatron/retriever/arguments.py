import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    pooling: str = field(
        default='cls',
        metadata={"help": "pooling method for query and passage encoder"}
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "normalize query and passage representations"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for softmax"}
    )

    # for lora
    lora: bool = field(default=False,
        metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )

    lora_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained lora model or model identifier from huggingface.co/models"}
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "lora r"}
    )

    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )

    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "lora target modules"}
    )

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default='json', metadata={"help": "huggingface dataset name"}
    )

    dataset_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    dataset_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    dataset_number_of_shards: int = field(
        default=1, metadata={"help": "number of shards to split the dataset into"}
    )

    dataset_shard_index: int = field(
        default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"}
    )

    train_group_size: int = field(
        default=8, metadata={"help": "number of passages used to train for each query"}
    )

    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage for training"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first n negative passages for training"})

    encode_is_query: bool = field(default=False)
    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})


    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    query_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for query"}
    )

    passage_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for passage"}
    )

    append_eos_token: bool = field(
        default=False, metadata={"help": "append eos token to query and passage, this is currently used for repllama"}
    )

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
