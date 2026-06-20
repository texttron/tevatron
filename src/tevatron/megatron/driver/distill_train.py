"""Megatron distillation trainer (listwise KL vs teacher logits).

Mirrors `tevatron.megatron.driver.train` with three differences:
  - builds `MegatronRerankerDistilDataset` + `MegatronRerankerDistilCollator`
    over a HF dataset produced by `tevatron.utils.annotate_with_teacher`;
  - sets `config.loss_kind = "distill"` before engine init;
  - exposes `--teacher_temp` / `--student_temp` for scaling the listwise KL.

Launch (single node, EP=8 for a 30B-A3B student or DP=8 for a dense student):

    torchrun --nproc_per_node=8 -m tevatron.megatron.driver.distill_train \\
        --model_name_or_path Qwen/Qwen3-0.6B-Base \\
        --distill_dataset_path /path/to/distill_cache/rlhn-680K-qwen3-reranker-8b \\
        --train_group_size 8 --teacher_temp 2.0 --student_temp 1.0 \\
        --learning_rate 3e-6 --num_epochs 1 --warmup_ratio 0.1 \\
        --save_dir output/qwen3-0.6b-distill
"""

import argparse
import logging
import os

import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from tevatron.megatron.config import (
    LORA_TARGET_GROUPS,
    MegatronRerankerConfig,
    expand_lora_target_groups,
)
from tevatron.megatron.distill_data import (
    MegatronRerankerDistilCollator,
    MegatronRerankerDistilDataset,
)
from tevatron.megatron.engine import MegatronRerankerEngine
from tevatron.megatron.driver.train import resolve_yes_no_token_ids

logger = logging.getLogger(__name__)


def parse_args() -> tuple[MegatronRerankerConfig, str]:
    parser = argparse.ArgumentParser(description="Megatron Reranker Distillation Training")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    # Parallelism
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    parser.add_argument("--expert_model_parallel_size", type=int, default=1)

    # Training
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_micro_batches", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--total_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA (uses megatron-bridge path; .venv-lora). See docs/LORA_ROADMAP.md.
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_groups",
        nargs="+",
        default=None,
        choices=list(LORA_TARGET_GROUPS.keys()),
        help="Named LoRA target groups. Mutually exclusive with "
             "--lora_target_modules. Picking groups by role avoids the "
             "MoE foot-gun where leaf names match every expert FFN.",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        help="Raw megatron-bridge target_modules patterns. Prefer "
             "--lora_target_groups for clarity; this flag is the escape "
             "hatch for power users who need patterns the registry "
             "doesn't expose.",
    )

    # MoE
    parser.add_argument("--moe_router_load_balancing_type", type=str, default="aux_loss")
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.000001)
    parser.add_argument("--moe_z_loss_coeff", type=float, default=None)

    # Distill data + loss
    parser.add_argument("--distill_dataset_path", type=str, required=True,
                        help="Path to the HF dataset produced by "
                             "tevatron.utils.annotate_with_teacher (save_to_disk).")
    parser.add_argument("--query_prefix", type=str, default="query:")
    parser.add_argument("--passage_prefix", type=str, default="passage:")
    parser.add_argument("--pad_to_multiple_of", type=int, default=16)
    parser.add_argument("--teacher_temp", type=float, default=1.0)
    parser.add_argument("--student_temp", type=float, default=1.0)

    # Checkpoint
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="tevatron-megatron")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    if args.lora_target_groups:
        lora_targets = expand_lora_target_groups(tuple(args.lora_target_groups))
    else:
        lora_targets = tuple(args.lora_target_modules)

    config = MegatronRerankerConfig(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        train_group_size=args.train_group_size,
        micro_batch_size=args.micro_batch_size,
        num_micro_batches=args.num_micro_batches,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
        num_epochs=args.num_epochs,
        max_seq_len=args.max_seq_len,
        grad_clip=args.grad_clip,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        moe_router_load_balancing_type=args.moe_router_load_balancing_type,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
        moe_z_loss_coeff=args.moe_z_loss_coeff,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        pad_to_multiple_of=args.pad_to_multiple_of,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        loss_kind="distill",
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    )
    return config, args.distill_dataset_path


def build_dataloader(
    config: MegatronRerankerConfig,
    distill_dataset_path: str,
    engine: MegatronRerankerEngine,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    dataset = MegatronRerankerDistilDataset(
        dataset_path=distill_dataset_path,
        train_group_size=config.train_group_size,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        seed=config.seed,
    )

    collator = MegatronRerankerDistilCollator(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )

    dp_rank = engine.get_data_parallel_rank()
    dp_size = engine.get_data_parallel_size()

    sampler = DistributedSampler(
        dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, drop_last=True
    )

    batch_size_per_step = config.micro_batch_size * config.num_micro_batches

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_step,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler, tokenizer


def split_micro_batches(batch: dict, num_micro_batches: int, train_group_size: int) -> list:
    """Split a batch into micro-batches for pipeline parallelism.

    `teacher_scores` is flat shape (n_queries * G,) — slicing must use the
    same per-microbatch query span × G stride as input_ids.
    """
    total_queries = batch["input_ids"].shape[0] // train_group_size
    queries_per_micro = total_queries // num_micro_batches
    pairs_per_micro = queries_per_micro * train_group_size
    micro_batches = []
    for i in range(num_micro_batches):
        s = i * pairs_per_micro
        e = s + pairs_per_micro
        micro_batches.append({
            "input_ids": batch["input_ids"][s:e],
            "attention_mask": batch["attention_mask"][s:e],
            "position_ids": batch["position_ids"][s:e],
            "teacher_scores": batch["teacher_scores"][s:e],
        })
    return micro_batches


def main():
    config, distill_dataset_path = parse_args()

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if torch.distributed.get_rank() == 0 else logging.WARN,
    )

    use_wandb = torch.distributed.get_rank() == 0
    if use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config={
                "model": config.model_name_or_path,
                "loss_kind": config.loss_kind,
                "teacher_temp": config.teacher_temp,
                "student_temp": config.student_temp,
                "train_group_size": config.train_group_size,
                "micro_batch_size": config.micro_batch_size,
                "num_micro_batches": config.num_micro_batches,
                "learning_rate": config.learning_rate,
                "min_lr": config.min_lr,
                "weight_decay": config.weight_decay,
                "warmup_steps": config.warmup_steps,
                "total_steps": config.total_steps,
                "max_seq_len": config.max_seq_len,
                "grad_clip": config.grad_clip,
                "tp": config.tensor_model_parallel_size,
                "pp": config.pipeline_model_parallel_size,
                "ep": config.expert_model_parallel_size,
                "moe_load_balancing": config.moe_router_load_balancing_type,
                "moe_aux_loss_coeff": config.moe_aux_loss_coeff,
                "distill_dataset_path": distill_dataset_path,
            },
        )

    resolve_yes_no_token_ids(config)

    engine = MegatronRerankerEngine(config)
    engine.initialize_parallel_and_model()

    dataloader, sampler, tokenizer = build_dataloader(config, distill_dataset_path, engine)

    class FakeTrainer:
        class state:
            epoch = 0
        class args:
            seed = config.seed
    dataloader.dataset.trainer = FakeTrainer()

    dp_rank = engine.get_data_parallel_rank()
    is_logging = dp_rank == 0

    steps_per_epoch = len(dataloader)
    if config.total_steps <= 0:
        config.total_steps = steps_per_epoch * config.num_epochs

    if config.warmup_steps <= 0 and config.warmup_ratio > 0:
        config.warmup_steps = int(config.total_steps * config.warmup_ratio)

    if is_logging:
        logger.info(f"Config: {config}")
        logger.info(f"Distill dataset: {distill_dataset_path} (n={len(dataloader.dataset)})")
        logger.info(f"Steps per epoch: {steps_per_epoch}, total steps: {config.total_steps}")

    engine.build_optimizer()

    global_step = 0
    epoch = 0
    while global_step < config.total_steps:
        sampler.set_epoch(epoch)
        FakeTrainer.state.epoch = epoch

        for batch in dataloader:
            if global_step >= config.total_steps:
                break

            micro_batches = split_micro_batches(
                batch, config.num_micro_batches, config.train_group_size
            )
            metrics = engine.train_step(micro_batches)

            global_step += 1

            if is_logging and global_step % config.log_interval == 0 and metrics:
                loss_str = f"loss={metrics.get('loss', 0):.4f}"
                lr_str = f"lr={metrics.get('lr', 0):.2e}"
                moe_str = ""
                for k, v in metrics.items():
                    if k.startswith("moe/"):
                        moe_str += f" {k}={v:.6f}"
                logger.info(f"step={global_step} {loss_str} {lr_str}{moe_str}")
                if use_wandb:
                    wandb.log(metrics, step=global_step)

            if global_step % config.save_interval == 0:
                engine.save_checkpoint(config.save_dir, global_step)

        epoch += 1

    engine.save_checkpoint(config.save_dir, global_step)

    if is_logging:
        logger.info(f"Training complete. Final step: {global_step}")
    if use_wandb:
        wandb.finish()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
