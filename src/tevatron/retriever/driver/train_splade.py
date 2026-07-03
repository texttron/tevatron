"""Train a causal-LM SPLADE retriever (the LACONIC recipe).

Mirrors ``tevatron.retriever.driver.train`` but swaps in:
  - ``SpladeModelForCausalLM`` (decoder backbone, optional bidirectional
    attention, max/mean/last pooling) instead of ``DenseModel``;
  - ``SpladeTrainer`` (adds FLOPS sparsity regularization) instead of
    ``TevatronTrainer``;
  - ``SpladeTrainingArguments`` (adds the FLOPS knobs).

Everything else — dataset, collator, tokenizer setup, checkpointing — is the
stock Tevatron path, so a SPLADE checkpoint loads through the normal HF route
and encodes via ``encode_splade.py``.

Example (decoder SPLADE, bidirectional, LoRA, on rlhn-680K):

    torchrun --nproc_per_node=8 -m tevatron.retriever.driver.train_splade \\
        --model_name_or_path meta-llama/Llama-3.1-8B \\
        --dataset_name rlhn/rlhn-680K \\
        --is_bidirectional --pooling_strategy max \\
        --lora --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \\
        --q_flops_loss_factor 1e-3 --p_flops_loss_factor 1e-3 --flops_warmup 100 \\
        --bf16 --train_group_size 16 --query_max_len 192 --passage_max_len 192 \\
        --learning_rate 1e-4 --num_train_epochs 1 --output_dir model_laconic
"""

import logging
import os
import sys

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from tevatron.retriever.arguments import (
    ModelArguments,
    DataArguments,
    SpladeTrainingArguments as TrainingArguments,
)
from tevatron.retriever.dataset import TrainDataset
from tevatron.retriever.collator import TrainCollator
from tevatron.retriever.modeling import SpladeModelForCausalLM
from tevatron.retriever.splade_trainer import SpladeTrainer

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

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank, training_args.device, training_args.n_gpu,
        bool(training_args.local_rank != -1), training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right' if data_args.padding_side == 'right' else 'left'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = SpladeModelForCausalLM.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    train_dataset = TrainDataset(data_args)
    collator = TrainCollator(data_args, tokenizer)

    trainer = SpladeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.set_trainer(trainer)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer.train(resume_from_checkpoint=(last_checkpoint is not None))
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
