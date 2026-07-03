from dataclasses import dataclass, field
from typing import Optional


# Named LoRA target groups. Each maps a semantic role ("attention",
# "dense MLP", "MoE expert FFN", ...) to the megatron-bridge module-name
# patterns that match it. The bridge resolves these against decoder module
# names with glob semantics — `linear_fc1` is a leaf-name match that hits
# *every* MLP linear, including the 128 expert FFNs in Qwen3-30B-A3B; the
# `*.experts.*` patterns scope an adapter to MoE-only or non-MoE-only
# linears.
#
# Pick groups by semantic role rather than typing module strings — that
# makes MoE-vs-dense decisions explicit and prevents the 128× blow-up of
# accidentally adapting every expert.
#
# Use `--lora_target_groups attn mlp` for dense models and
# `--lora_target_groups attn` (or `attn moe_experts`) for MoE.
# Power-user override: `--lora_target_modules` accepts raw bridge patterns
# and bypasses the registry entirely.
LORA_TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    # Attention projections — safe for any architecture.
    "attn":        ("linear_qkv", "linear_proj"),
    # Dense MLP. Matches the *non-MoE* MLP block on dense models. On MoE
    # models the leaf names `linear_fc1/fc2` also match the per-expert
    # FFNs, so prefer `moe_experts` / `moe_shared` / `moe_router` there.
    "mlp":         ("linear_fc1", "linear_fc2"),
    # MoE expert FFNs (one adapter pair per expert, scoped by parent path).
    "moe_experts": ("*.experts.*.linear_fc1", "*.experts.*.linear_fc2"),
    # MoE shared expert FFN (Qwen3-MoE has a `shared_experts` branch
    # alongside the routed ones; harmless if absent — the bridge silently
    # skips non-matching patterns).
    "moe_shared":  ("*.shared_experts.linear_fc1", "*.shared_experts.linear_fc2"),
    # MoE router (top-k gate). Tiny but sometimes worth adapting.
    "moe_router":  ("*.router.weight",),
}


def expand_lora_target_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Expand a list of group names into the flat module-pattern list that
    megatron-bridge LoRA expects. Order is preserved; duplicates removed."""
    seen: dict[str, None] = {}
    for g in groups:
        if g not in LORA_TARGET_GROUPS:
            raise ValueError(
                f"Unknown LoRA target group {g!r}. "
                f"Known: {sorted(LORA_TARGET_GROUPS)}"
            )
        for pat in LORA_TARGET_GROUPS[g]:
            seen.setdefault(pat, None)
    return tuple(seen.keys())


@dataclass
class MegatronRerankerConfig:
    # Model
    model_name_or_path: str = ""
    dtype: str = "bfloat16"

    # Parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False

    # Data-parallel sharding. By default we use Megatron's distributed optimizer
    # (ZeRO-1: optimizer state sharded across DP, params/grads replicated).
    # Setting use_megatron_fsdp=True activates the Megatron-FSDP path, where
    # dp_sharding_strategy controls the ZeRO stage:
    #   "optim"               -> ZeRO-1 (optimizer only; == use_distributed_optimizer)
    #   "optim_grads"         -> ZeRO-2 (optimizer + gradients)
    #   "optim_grads_params"  -> ZeRO-3 (optimizer + gradients + parameters)
    # This gives FSDP-equivalent param/grad sharding without leaving the Megatron
    # backend, on top of TP/PP/EP.
    use_megatron_fsdp: bool = False
    dp_sharding_strategy: str = "optim_grads_params"

    # Activation recompute (a.k.a. gradient/activation checkpointing). Trades
    # compute for memory by recomputing layer activations in the backward pass
    # instead of storing them. Needed to match HF FSDP's activation_checkpointing
    # at low TP, where un-sharded activations otherwise dominate memory.
    # granularity: "full" (whole transformer layer) or "selective"; method:
    # "uniform" or "block". recompute_enabled gates the whole feature.
    recompute_enabled: bool = False
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    # Training
    train_group_size: int = 8
    micro_batch_size: int = 1
    num_micro_batches: int = 1
    global_batch_size: int = 32
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    total_steps: int = -1
    num_epochs: int = 1
    max_seq_len: int = 512
    grad_clip: float = 1.0
    use_distributed_optimizer: bool = True
    seed: int = 42

    # MoE load balancing
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 0.000001
    moe_z_loss_coeff: Optional[float] = None

    # Token IDs for yes/no scoring
    yes_token_id: int = 9693
    no_token_id: int = 2152

    # Data
    dataset_name: str = "json"
    dataset_path: Optional[str] = None
    dataset_split: str = "train"
    query_prefix: str = "query:"
    passage_prefix: str = "passage:"
    append_eos_token: bool = False
    pad_to_multiple_of: int = 16

    # Freezing
    freeze_backbone: bool = False

    # LoRA (Phase 1+2 of docs/LORA_ROADMAP.md)
    # When use_lora=True the engine builds the model via megatron-bridge
    # (instead of mbridge directly), registers the PEFT pre-wrap hook so the
    # adapters land in DDP's grad buffers, and the optimizer only sees adapter
    # params (which are the only ones with requires_grad=True after the hook).
    #
    # `lora_target_modules` is the flat module-pattern list passed straight
    # to megatron-bridge `LoRA(target_modules=...)`. The drivers expand
    # `--lora_target_groups` (see LORA_TARGET_GROUPS above) into this list;
    # power-user `--lora_target_modules` overrides the registry. Default
    # covers both attention and dense MLP; on MoE models pick groups
    # explicitly to avoid the leaf names `linear_fc1/fc2` matching every
    # one of the 128 expert FFNs.
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"
    )

    # Loss
    # "contrastive": listwise CE with target=0 (positive at index 0).
    # "distill"    : listwise KL vs teacher_scores (provided per pair by the
    #                distill collator). Temperatures scale logits before the
    #                softmax; student_temp ** 2 reapplies the Hinton gradient
    #                rescaling.
    loss_kind: str = "contrastive"
    teacher_temp: float = 1.0
    student_temp: float = 1.0

    # Checkpoint
    save_dir: str = "output"
    save_interval: int = 1000
    log_interval: int = 10

    # Wandb
    wandb_project: str = "tevatron-megatron"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True
