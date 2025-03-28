import logging
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import VllmMultiModalEncodeCollator
from vllm import LLM
from vllm.config import PoolerConfig
from PIL import Image
from vllm.lora.request import LoRARequest

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

    processor = AutoProcessor.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    min_pixels = 1 * 28 * 28
    max_pixels = 2560 * 28 * 28

    if training_args.bf16:
        torch_dtype = 'bfloat16'
    elif training_args.fp16:
        torch_dtype = 'float16'
    else:
        torch_dtype = 'float32'


    pooler_config = PoolerConfig(pooling_type=model_args.pooling.upper(),
                                 normalize=model_args.normalize)

    model = LLM(
        model=model_args.model_name_or_path,
        task="embed",
        enforce_eager=True,
        override_pooler_config=pooler_config,
        dtype=torch_dtype,
        # mm_processor_kwargs={"use_fast": True},
        enable_lora=True if model_args.lora_name_or_path else False,
        max_lora_rank=model_args.lora_r,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = VllmMultiModalEncodeCollator(
        data_args=data_args,
        processor=processor,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    lookup_indices = []
    encoded = []
    for (batch_ids, texts, images) in tqdm(encode_loader, desc="Encoding"):
        lookup_indices.extend(batch_ids)
        vllm_inputs = []
        for prompt, image in zip(texts, images):
            vllm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {'image': image} if image is not None else None,
            })
        outputs = model.embed(vllm_inputs,
                              use_tqdm=False,
                              lora_request=LoRARequest("emb_adapter",
                                                       1,
                                                       model_args.lora_name_or_path) if model_args.lora_name_or_path else None)

        for output in outputs:
            encoded.append(output.outputs.embedding)

    encoded = np.stack(encoded, dtype=np.float16)

    with open(data_args.encode_output_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
