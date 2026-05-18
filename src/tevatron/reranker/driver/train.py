import logging
import os
import sys
import torch
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tevatron.reranker.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.reranker.modeling import RerankerModel
from tevatron.reranker.dataset import RerankerTrainDataset
from tevatron.reranker.collator import RerankerTrainCollator
from tevatron.reranker.trainer import RerankerTrainer

logger = logging.getLogger(__name__)


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # We're running in a distributed environment
        import torch.distributed as dist
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend="nccl")
        return rank
    else:
        # We're not running in a distributed environment
        return -1


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = setup_ddp()
    training_args.local_rank = local_rank

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(local_rank != -1),
        training_args.fp16 or training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'right'

    model = RerankerModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    # Move model to GPU
    if local_rank != -1:
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = RerankerTrainDataset(data_args)
    train_collator = RerankerTrainCollator(data_args, tokenizer)

    training_args.gc_chunk_size = getattr(training_args, 'gc_chunk_size', 2)
    training_args.grad_cache = getattr(training_args, 'grad_cache', False)

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collator
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
