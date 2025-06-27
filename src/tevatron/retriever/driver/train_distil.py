import logging
import os
import sys
import torch
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import DistilTrainDataset
from tevatron.retriever.collator import DistilTrainCollator
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.trainer import DistilTevatronTrainer as DistilTrainer

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
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
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
    
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
    )

    # Train dataset and collator
    train_dataset = DistilTrainDataset(data_args)
    collator = DistilTrainCollator(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer_cls = DistilTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
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
