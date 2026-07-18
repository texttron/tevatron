import logging
import os
import sys

import torch
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import TrainingArguments

from tevatron.reranker.arguments import ModelArguments, DataArguments

from tevatron.reranker.modeling import RerankerModel
from tevatron.reranker.dataset import RerankerTrainDataset
from tevatron.reranker.trainer import RerankerTrainer
from tevatron.reranker.collator import RerankerTrainCollator

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'right'
    # Load weights in the training compute dtype (mirrors retriever/driver/train.py).
    # The base model is frozen under LoRA, so fp32 master weights are pure memory
    # overhead; bf16 load also matches the historical DeepSpeed-bf16 runs, where
    # params were kept in bf16 as well.
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    model = RerankerModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    # The HF SequenceClassification head reads the last non-pad position by
    # comparing input_ids to config.pad_token_id. Models like Qwen3 ship
    # config.pad_token_id=None, in which case .build() defaults it to 0 — but
    # the tokenizer pads with a different id (e.g. 151643), so the head ends up
    # reading position 0 instead of the true last token. Pin the config to the
    # tokenizer's pad id so training and saved-checkpoint inference agree.
    model.hf_model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = RerankerTrainDataset(data_args)
    train_collator = RerankerTrainCollator(data_args, tokenizer)

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        processing_class=tokenizer,
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
