"""
Megatron-based pointwise reranker training.

Launch:
    torchrun --nproc_per_node=8 -m tevatron.megatron.driver.train \
        --model_name_or_path Qwen/Qwen3.5-35B-A3B \
        --expert_model_parallel_size 8 \
        --dataset_path /path/to/data.jsonl \
        --train_group_size 8 \
        --total_steps 5000
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from tevatron.megatron.config import (
    LORA_TARGET_GROUPS,
    MegatronRerankerConfig,
    expand_lora_target_groups,
)
from tevatron.megatron.data import MegatronRerankerCollator
from tevatron.megatron.engine import MegatronRerankerEngine
from tevatron.reranker.dataset import RerankerTrainDataset
from tevatron.reranker.arguments import DataArguments

logger = logging.getLogger(__name__)


def parse_args() -> MegatronRerankerConfig:
    parser = argparse.ArgumentParser(description="Megatron Reranker Training")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    # Parallelism
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    parser.add_argument("--expert_model_parallel_size", type=int, default=1)
    parser.add_argument("--use_megatron_fsdp", action="store_true",
                        help="Enable the Megatron-FSDP data-parallel path "
                             "(ZeRO-2/3-style grad/param sharding). Default off "
                             "uses the distributed optimizer (ZeRO-1).")
    parser.add_argument("--dp_sharding_strategy", type=str, default="optim_grads_params",
                        choices=["optim", "optim_grads", "optim_grads_params"],
                        help="With --use_megatron_fsdp: optim=ZeRO-1, "
                             "optim_grads=ZeRO-2, optim_grads_params=ZeRO-3.")
    parser.add_argument("--recompute_enabled", action="store_true",
                        help="Activation recompute (gradient checkpointing): "
                             "recompute layer activations in backward to save "
                             "memory. Matches HF FSDP's activation_checkpointing.")
    parser.add_argument("--recompute_granularity", type=str, default="full",
                        choices=["full", "selective"])
    parser.add_argument("--recompute_method", type=str, default="uniform",
                        choices=["uniform", "block"])
    parser.add_argument("--recompute_num_layers", type=int, default=1)

    # Training
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_micro_batches", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Fraction of total_steps for LR warmup. Used when warmup_steps==0.")
    parser.add_argument("--total_steps", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")

    # LoRA (uses megatron-bridge path; .venv-lora). See docs/LORA_ROADMAP.md.
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # Preferred: pick semantic groups, e.g. `--lora_target_groups attn mlp`
    # for dense models or `--lora_target_groups attn moe_experts` for MoE.
    # See LORA_TARGET_GROUPS in tevatron.megatron.config for the full list.
    parser.add_argument(
        "--lora_target_groups",
        nargs="+",
        default=None,
        choices=list(LORA_TARGET_GROUPS.keys()),
        help="Named LoRA target groups. Mutually exclusive with "
             "--lora_target_modules. Picking groups by role avoids the "
             "MoE foot-gun where leaf names match every expert FFN.",
    )
    # Escape hatch / backwards-compat: raw bridge module-name patterns.
    # Default kept for backwards compatibility with existing scripts; this
    # default is dense-only — on MoE it will silently adapt every expert
    # FFN (use --lora_target_groups instead).
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

    # Data
    parser.add_argument("--dataset_name", type=str, default="json")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--query_prefix", type=str, default="query:")
    parser.add_argument("--passage_prefix", type=str, default="passage:")
    parser.add_argument("--append_eos_token", action="store_true")
    parser.add_argument("--pad_to_multiple_of", type=int, default=16)

    # Checkpoint
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="tevatron-megatron")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    # Resolve LoRA targets: groups take precedence over the raw flag when
    # set explicitly. The raw flag's default-of-four patterns is kept for
    # backwards compatibility with existing scripts.
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
        use_megatron_fsdp=args.use_megatron_fsdp,
        dp_sharding_strategy=args.dp_sharding_strategy,
        recompute_enabled=args.recompute_enabled,
        recompute_granularity=args.recompute_granularity,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
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
        freeze_backbone=args.freeze_backbone,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        moe_router_load_balancing_type=args.moe_router_load_balancing_type,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
        moe_z_loss_coeff=args.moe_z_loss_coeff,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        append_eos_token=args.append_eos_token,
        pad_to_multiple_of=args.pad_to_multiple_of,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
    )
    return config


def resolve_yes_no_token_ids(config: MegatronRerankerConfig):
    """Resolve yes/no token IDs.

    The prompt ends with "?" (no trailing space) so the next token the BPE
    produces is " yes"/" no" with leading space, which has different ids than
    "yes"/"no". Resolve the contextually-correct ids by tokenizing the full
    prompt and taking the appended token.
    """
    from tevatron.megatron.data import RERANKER_PROMPT_SUFFIX
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    base_ids = tokenizer.encode(RERANKER_PROMPT_SUFFIX, add_special_tokens=False)
    yes_full = tokenizer.encode(RERANKER_PROMPT_SUFFIX + " yes", add_special_tokens=False)
    no_full = tokenizer.encode(RERANKER_PROMPT_SUFFIX + " no", add_special_tokens=False)
    assert yes_full[: len(base_ids)] == base_ids, f"yes tokenization changed prefix: {yes_full} vs {base_ids}"
    assert no_full[: len(base_ids)] == base_ids, f"no tokenization changed prefix: {no_full} vs {base_ids}"
    assert len(yes_full) == len(base_ids) + 1, f"' yes' is multi-token: {yes_full[len(base_ids):]}"
    assert len(no_full) == len(base_ids) + 1, f"' no' is multi-token: {no_full[len(base_ids):]}"
    config.yes_token_id = yes_full[-1]
    config.no_token_id = no_full[-1]
    if torch.distributed.get_rank() == 0:
        logger.info(
            f"yes_token_id={config.yes_token_id} ({tokenizer.decode([config.yes_token_id])!r}), "
            f"no_token_id={config.no_token_id} ({tokenizer.decode([config.no_token_id])!r})"
        )


def build_dataloader(config: MegatronRerankerConfig, engine: MegatronRerankerEngine):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    data_args = DataArguments(
        dataset_name=config.dataset_name,
        dataset_path=config.dataset_path,
        dataset_split=config.dataset_split,
        train_group_size=config.train_group_size,
        rerank_max_len=config.max_seq_len,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        append_eos_token=config.append_eos_token,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )

    dataset = RerankerTrainDataset(data_args)

    collator = MegatronRerankerCollator(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )

    dp_rank = engine.get_data_parallel_rank()
    dp_size = engine.get_data_parallel_size()

    sampler = DistributedSampler(
        dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, drop_last=True
    )

    # Each dataloader batch = micro_batch_size queries × num_micro_batches
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


def split_micro_batches(batch: dict, num_micro_batches: int) -> list:
    """Split a batch into micro-batches for pipeline parallelism."""
    total_size = batch["input_ids"].shape[0]
    micro_batch_size = total_size // num_micro_batches
    micro_batches = []
    for i in range(num_micro_batches):
        start = i * micro_batch_size
        end = start + micro_batch_size
        micro_batches.append({
            "input_ids": batch["input_ids"][start:end],
            "attention_mask": batch["attention_mask"][start:end],
            "position_ids": batch["position_ids"][start:end],
        })
    return micro_batches


def main():
    config = parse_args()

    # Init distributed
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Set seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if torch.distributed.get_rank() == 0 else logging.WARN,
    )

    # Wandb (rank 0 only)
    use_wandb = torch.distributed.get_rank() == 0
    if use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config={
                "model": config.model_name_or_path,
                "train_group_size": config.train_group_size,
                "micro_batch_size": config.micro_batch_size,
                "num_micro_batches": config.num_micro_batches,
                "global_batch_size": config.global_batch_size,
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
            },
        )

    # Resolve yes/no token IDs
    resolve_yes_no_token_ids(config)

    # Build engine (parallel groups + model only; optimizer built after total_steps is known)
    engine = MegatronRerankerEngine(config)
    engine.initialize_parallel_and_model()

    # Build dataloader
    dataloader, sampler, tokenizer = build_dataloader(config, engine)

    # Assign trainer reference for dataset epoch tracking
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
        logger.info(f"DP size: {dp_rank}/{engine.get_data_parallel_size()}, "
                    f"World size: {torch.distributed.get_world_size()}")
        logger.info(f"Dataset size: {len(dataloader.dataset)}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Num epochs: {config.num_epochs}")
        logger.info(f"Total steps: {config.total_steps}")
        logger.info(f"Warmup steps: {config.warmup_steps}")

    engine.build_optimizer()

    # Training loop
    global_step = 0
    epoch = 0

    while global_step < config.total_steps:
        sampler.set_epoch(epoch)
        FakeTrainer.state.epoch = epoch

        for batch in dataloader:
            if global_step >= config.total_steps:
                break

            micro_batches = split_micro_batches(batch, config.num_micro_batches)
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

    # Final save
    engine.save_checkpoint(config.save_dir, global_step)

    if is_logging:
        logger.info(f"Training complete. Final step: {global_step}")

    if use_wandb:
        wandb.finish()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
