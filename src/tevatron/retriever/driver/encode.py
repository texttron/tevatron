import logging
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel
from tevatron.retriever.chunk_utils import chunk_passage_text

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

    encode_collator = EncodeCollator(data_args=data_args, tokenizer=tokenizer)

    chunk_mode = data_args.use_chunk_maxsim and not data_args.encode_is_query

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    if chunk_mode:
        logger.info("Encoding corpus with chunk+MaxSim: passages will be split into chunks.")
        chunk_buffer = []

        def flush_chunk_buffer():
            nonlocal chunk_buffer
            if not chunk_buffer:
                return
            features = chunk_buffer
            chunk_buffer = []
            batch_ids, batch = encode_collator(features)
            lookup_indices.extend(batch_ids)
            with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

        per_device_bs = training_args.per_device_eval_batch_size or 1
        for idx in tqdm(range(len(encode_dataset))):
            content_id, content_text, _, _, _ = encode_dataset[idx]
            if content_text is None:
                continue
            chunks = chunk_passage_text(content_text, tokenizer, data_args)
            for chunk_text in chunks:
                chunk_buffer.append((content_id, chunk_text, None, None, None))
                if len(chunk_buffer) >= per_device_bs:
                    flush_chunk_buffer()
        flush_chunk_buffer()

    else:
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=encode_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        for (batch_ids, batch) in tqdm(encode_loader):
            lookup_indices.extend(batch_ids)
            with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(training_args.device)
                    if data_args.encode_is_query:
                        model_output: EncoderOutput = model(query=batch)
                        encoded.append(model_output.q_reps.cpu().detach().numpy())
                    else:
                        model_output: EncoderOutput = model(passage=batch)
                        encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded) if encoded else np.empty((0, model.config.hidden_size))

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
