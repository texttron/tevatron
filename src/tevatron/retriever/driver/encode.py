import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from rich import print

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator, ChunkedEncodeCollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = DenseModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    use_chunked = not data_args.encode_is_query and data_args.passage_chunk_size > 0
    
    if use_chunked:
        logger.info(f"Using chunked passage encoding with chunk_size={data_args.passage_chunk_size}")
        encode_collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=tokenizer)
    else:
        encode_collator = EncodeCollator(data_args=data_args, tokenizer=tokenizer)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    for batch in tqdm(encode_loader):
        with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                if use_chunked:
                    doc_ids, batch_inputs, sep_positions, chunk_counts = batch
                    print(batch_inputs)
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.to(training_args.device)
                    chunk_embs, chunk_mask = model.encode_passage(batch_inputs, sep_positions)
                    
                    # Flatten chunk embeddings and create lookup indices
                    batch_size, max_chunks, hidden_size = chunk_embs.shape
                    for i, doc_id in enumerate(doc_ids):
                        for chunk_idx in range(max_chunks):
                            if chunk_mask[i, chunk_idx] > 0:  # Valid chunk
                                encoded.append(chunk_embs[i, chunk_idx].cpu().detach().numpy())
                                lookup_indices.append((doc_id, chunk_idx))
                else:
                    batch_ids, batch_inputs = batch
                    lookup_indices.extend(batch_ids)
                    
                    for k, v in batch_inputs.items():
                        batch_inputs[k] = v.to(training_args.device)
                    
                    if data_args.encode_is_query:
                        model_output: EncoderOutput = model(query=batch_inputs)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: EncoderOutput = model(passage=batch_inputs)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

    # Combine encoded embeddings
    if use_chunked:
        encoded = np.stack(encoded)
        logger.info(f"Encoded {len(set(d for d, c in lookup_indices))} docs into {len(lookup_indices)} chunks")
    else:
        encoded = np.concatenate(encoded)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)
    
    logger.info(f"Saved embeddings to {data_args.encode_output_path}, shape: {encoded.shape}")


if __name__ == "__main__":
    main()
