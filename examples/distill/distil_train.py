import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import DataArguments
from tevatron.modeling import DenseModel
from tevatron.reranker.modeling import RerankerModel
from tevatron.distillation.data import DistilTrainDataset, DistilTrainCollator, HFDistilTrainDataset
from tevatron.distillation.trainer import DistilTrainer
from tevatron.distillation.arguments import DistilModelArguments, DistilTrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DistilModelArguments, DataArguments, DistilTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: DistilModelArguments
        data_args: DataArguments
        training_args: DistilTrainingArguments

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

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = DenseModel.build(
        model_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_config_name if model_args.teacher_config_name else model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        model_args.teacher_tokenizer_name if model_args.teacher_tokenizer_name else model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    teacher_model = RerankerModel.load(
        model_name_or_path=model_args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
    
    )
    teacher_model.to(training_args.device)
    teacher_model.eval()

    train_dataset = HFDistilTrainDataset(student_tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = DistilTrainDataset(data_args, train_dataset.process(), tokenizer, teacher_tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer = DistilTrainer(
        teacher_model=teacher_model,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DistilTrainCollator(
            tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
