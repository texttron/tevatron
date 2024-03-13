import os
import logging
import pickle

from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import jax
import jax.numpy as jnp
import optax
import flax
import orbax
import numpy as np
import math

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

import datasets
from dataclasses import dataclass
from transformers import AutoTokenizer
import simple_parsing.helpers as parsing_helpers
from simple_parsing import ArgumentParser
from tqdm import trange

import magix
from magix.models import ENCODER_MODEL_MAPPING
from magix.lora import Lora

# logger with date and time
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def pad_to_bsz(x, bsz, pad_str=""):
    if len(x) > bsz:
        raise ValueError(f"Length of x ({len(x)}) is greater than bsz ({bsz})")
    return x + [pad_str] * (bsz - len(x))


@dataclass
class EncoderArguments:
    """ Arguments for the encoder """
    model_type: str
    model_name_or_path: str
    model_config_name_or_path: str
    tokenizer_name_or_path: str
    dataset_name_or_path: str
    output_dir: str
    max_seq_length: int = 256
    batch_size: int = 32
    input_type: str = "passage"
    split: str = "train"
    num_shards: Optional[int] = None
    shard_id: Optional[int] = None
    mesh_shape: List[int] = parsing_helpers.list_field(default=[1, 8])
    lora: Optional[str] = None
    lora_alpha: float = 30
    scale_by_dim: bool = False
    hf_format: bool = False
    
def main():
    parser = ArgumentParser()
    parser.add_arguments(EncoderArguments, dest="args")
    args = parser.parse_args().args
    logger.info(f"Arguments: {args}")
    
    # Create mesh
    mesh = magix.create_device_mesh((args.mesh_shape[0], args.mesh_shape[1]))

    # Load the model
    _model_cls = ENCODER_MODEL_MAPPING[args.model_type]
    if os.path.isdir(args.model_name_or_path) and not args.hf_format:
        model, model_params = magix.load_model_local(
            _model_cls,
            args.model_name_or_path,
            sharding_config=_model_cls.partition_rules,
            mesh=mesh,
            model_config=_model_cls.config_class.from_pretrained(args.model_config_name_or_path),
        )
    else:
        model, model_params = magix.load_model_hub(
            _model_cls, args.model_name_or_path,
            sharding_config=_model_cls.partition_rules,
            mesh=mesh,
            half=True,
        )
        
    if args.lora is not None:
        lora = Lora(
            args.lora_alpha,
            rules={
                'layers/.*/kernel': 16,
            }
        )
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        lora_params = ckptr.restore(args.lora, item=None)
        lora_params = jax.device_put(
            lora_params,
            NamedSharding(mesh, PartitionSpec(None, None)))
        model_params = jax.jit(
            lora.apply, 
            in_shardings=(magix.item_sharding(model_params),magix.item_sharding(lora_params)),
            out_shardings=magix.item_sharding(model_params)
            ) (model_params, lora_params)
        del lora_params
    

    # Load the dataset
    if os.path.exists(args.dataset_name_or_path) and args.dataset_name_or_path.endswith(".jsonl"):
        dataset = datasets.load_dataset(
            "json",
            data_files=args.dataset_name_or_path,
            split=args.split,
        )
    else:
        dataset = datasets.load_dataset(
            args.dataset_name_or_path,
            split=args.split,
        )
        
    if args.num_shards is not None:
        dataset = dataset.shard(args.num_shards, args.shard_id)
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        add_eos_token=True, use_fast=True, padding_side='left', legacy=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    @jax.jit
    def encode(params, batch):
        reps = model(**batch, params=params, train=False)[0][:, -1, :]
        if args.scale_by_dim:
            reps = reps / math.sqrt(reps.shape[-1])
        return reps
    
    
    all_representations = []
    
    assert args.input_type in ["passage", "question", "query"]
    
    def format_batch(xx, input_type):
        if input_type == "passage":
            return [title + " " + passage for title, passage in zip(xx['title'], xx['text'])], xx['docid']
        elif input_type == "query":
            return xx['query'], xx['query_id']
        elif input_type == "question":
            return xx['question'], xx['question_id']
        
        raise ValueError(f"Unknown input type: {input_type}")
    
    all_ids = []
    
    with mesh:
        for idx in trange(0, len(dataset), args.batch_size):
            batch, batch_ids = format_batch(
                dataset[idx:idx+args.batch_size],
                args.input_type
            )
            batch = jax.tree_map(
                lambda xx: pad_to_bsz(xx, args.batch_size),
                batch, is_leaf=lambda xx: isinstance(xx, list))
            
            batch = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="np",
            )

            encodings = encode(model_params, dict(batch))

            representations = jax.device_put(encodings, jax.devices("cpu")[0])
            all_representations.append(representations[:len(batch_ids)])
            all_ids.extend(batch_ids)
            
        
    all_representations = jnp.concatenate(all_representations, axis=0)
    logger.info(f"Shape of all representations: {all_representations.shape}")
    logger.info(f"Saving representations to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.shard_id is None:
        output_path = os.path.join(args.output_dir, "emb.pkl")
    else:
        output_path = os.path.join(args.output_dir, f"emb_{args.shard_id}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump((all_representations, all_ids), f)
    
if __name__ == "__main__":
    main()