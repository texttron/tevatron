import logging
import os
from typing import Dict, List, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from transformers import AutoConfig

from megatron.core.transformer.enums import AttnBackend

from tevatron.megatron.config import MegatronRerankerConfig

logger = logging.getLogger(__name__)


def last_token_pool(output: torch.Tensor, attention_mask: torch.Tensor, yes_token_id: int, no_token_id: int):
    """Extract yes/no relevance score from the last token's logits.

    Returns:
        (yes_logits, no_logits) tuple, each shape (batch,). Caller computes the
        log-odds and listwise CE.
    """
    output = output.transpose(0, 1)  # -> (batch, seq_len, vocab_size)
    seq_lengths = attention_mask.sum(dim=-1) - 1
    last_hidden = output[torch.arange(output.size(0), device=output.device), seq_lengths]  # (batch, vocab_size)
    return last_hidden[:, yes_token_id], last_hidden[:, no_token_id]


class MegatronRerankerEngine:
    """Minimal Megatron engine for pointwise reranker training with EP/TP/PP."""

    def __init__(self, config: MegatronRerankerConfig):
        self.config = config
        self.param_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.bridge = None
        self.tf_config = None
        self.yes_token_id = config.yes_token_id
        self.no_token_id = config.no_token_id

    def initialize(self):
        """Initialize parallel groups, build model, load weights, build optimizer."""
        self.initialize_parallel_and_model()
        self.build_optimizer()

    def initialize_parallel_and_model(self):
        self._init_parallel_groups()
        self._build_model()

    def build_optimizer(self):
        self._build_optimizer()

    def _init_parallel_groups(self):
        if mpu.is_initialized():
            return
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.config.expert_model_parallel_size,
        )
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(self.config.seed)

    def _build_model(self):
        # FSDP-style ZeRO-2/3 is not supported on either build path; gate once
        # here. See docs/bugs-and-fixes.md + roadmap in docs/megatron_reranker_usage.md.
        if self.config.use_megatron_fsdp:
            raise NotImplementedError(
                "use_megatron_fsdp (ZeRO-2/3) is not supported: mbridge/mcore "
                "version coupling blocks wiring the FSDP wrapper (see "
                "docs/bugs-and-fixes.md). Use TP/EP for memory relief instead."
            )
        if self.config.use_lora:
            self._build_model_with_megatron_bridge()
        else:
            self._build_model_with_mbridge()

        if torch.distributed.get_rank() == 0:
            total_params = sum(p.numel() for model_chunk in self.model for p in model_chunk.parameters())
            trainable_params = sum(
                p.numel() for model_chunk in self.model for p in model_chunk.parameters() if p.requires_grad
            )
            logger.info(f"Total params: {total_params / 1e9:.2f}B, Trainable: {trainable_params / 1e9:.2f}B")

    def _build_model_with_mbridge(self):
        """Original mbridge path. Used for full-FT (use_lora=False)."""
        from mbridge import AutoBridge

        hf_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
        self.bridge = AutoBridge.from_config(hf_config, dtype=self.param_dtype)

        extra_args = {
            "variable_seq_lengths": False,
            "sequence_parallel": self.config.sequence_parallel,
            "attention_backend": AttnBackend.flash,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": self.config.moe_router_load_balancing_type,
            "moe_aux_loss_coeff": self.config.moe_aux_loss_coeff,
        }
        if self.config.moe_z_loss_coeff is not None:
            extra_args["moe_z_loss_coeff"] = self.config.moe_z_loss_coeff
        if self.config.recompute_enabled:
            # Activation recompute (matches HF FSDP's activation_checkpointing):
            # recompute layer activations in backward instead of storing them.
            extra_args["recompute_granularity"] = self.config.recompute_granularity
            extra_args["recompute_method"] = self.config.recompute_method
            extra_args["recompute_num_layers"] = self.config.recompute_num_layers

        self.bridge.set_extra_args(**extra_args)
        self.tf_config = self.bridge.config
        self.tf_config.fp16 = self.param_dtype == torch.float16
        self.tf_config.bf16 = self.param_dtype == torch.bfloat16

        from megatron.core.distributed import finalize_model_grads
        self.tf_config.finalize_model_grads_func = finalize_model_grads

        # Data-parallel sharding: ZeRO-1 (distributed optimizer; params/grads
        # replicated). FSDP-style ZeRO-2/3 is gated off in _build_model().
        ddp_config = {
            "grad_reduce_in_fp32": True,
            "use_distributed_optimizer": self.config.use_distributed_optimizer,
        }

        self.model = self.bridge.get_model(
            weight_path=self.config.model_name_or_path,
            wrap_with_ddp=True,
            bf16=self.tf_config.bf16,
            fp16=self.tf_config.fp16,
            ddp_config=ddp_config,
        )

    def _build_model_with_megatron_bridge(self):
        """megatron-bridge path used for LoRA. The bridge owns the pre-wrap
        hook lifecycle, which is what makes LoRA work under DDP + distributed
        optimizer (adapters must exist BEFORE DDP wraps grad buffers; see
        docs/LORA_ROADMAP.md §3).

        The provider already has one pre-wrap hook registered by
        to_megatron_provider() to load HF weights. We append a second hook
        that applies the PEFT transform; the bridge composes them in order so
        weights load → adapters get attached → DDP wraps the (small) trainable
        param set.
        """
        from megatron.bridge import AutoBridge as _MBAutoBridge
        from megatron.bridge.peft.lora import LoRA
        from megatron.core.distributed import (
            DistributedDataParallelConfig,
            finalize_model_grads,
        )

        # Build the bridge from a real HF checkpoint dir (the bridge needs
        # weights to load them via the pre-wrap hook).
        self.bridge = _MBAutoBridge.from_hf_pretrained(self.config.model_name_or_path)
        provider = self.bridge.to_megatron_provider(load_weights=True)

        # Override parallelism + runtime fields on the provider.
        provider.tensor_model_parallel_size = self.config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = self.config.pipeline_model_parallel_size
        provider.expert_model_parallel_size = self.config.expert_model_parallel_size
        provider.sequence_parallel = self.config.sequence_parallel
        provider.bf16 = self.param_dtype == torch.bfloat16
        provider.fp16 = self.param_dtype == torch.float16
        provider.params_dtype = self.param_dtype
        provider.attention_backend = AttnBackend.flash
        if getattr(provider, "num_moe_experts", None):
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_router_load_balancing_type = self.config.moe_router_load_balancing_type
            provider.moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
            if self.config.moe_z_loss_coeff is not None:
                provider.moe_z_loss_coeff = self.config.moe_z_loss_coeff
        provider.variable_seq_lengths = False
        if self.config.recompute_enabled:
            # Activation recompute (matches HF FSDP's activation_checkpointing).
            provider.recompute_granularity = self.config.recompute_granularity
            provider.recompute_method = self.config.recompute_method
            provider.recompute_num_layers = self.config.recompute_num_layers
        provider.finalize_model_grads_func = finalize_model_grads

        # megatron-bridge defers MCore TransformerConfig.__post_init__ (which
        # computes init_method/output_layer_init_method from init_method_std)
        # into finalize(). AutoBridge.get_model() calls it for us, but
        # provide_distributed_model() does not — so we must call it explicitly
        # after setting parallelism/MoE fields, before model construction.
        if hasattr(provider, "finalize"):
            provider.finalize()
        self.tf_config = provider

        # PEFT pre-wrap hook: freeze base, attach adapters. Runs AFTER the
        # bridge's weight-loading hook so the base weights are real.
        self._lora = LoRA(
            target_modules=list(self.config.lora_target_modules),
            dim=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
        )

        def _peft_hook(model_chunks):
            transformed = self._lora(model_chunks, training=True)
            self._lora.set_params_to_save(transformed)
            return transformed

        provider.register_pre_wrap_hook(_peft_hook)

        ddp_kwargs = dict(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            use_distributed_optimizer=self.config.use_distributed_optimizer,
            check_for_nan_in_grad=True,
            bucket_size=None,
            average_in_collective=True,
        )
        ddp_config = DistributedDataParallelConfig(**ddp_kwargs)

        self.model = provider.provide_distributed_model(
            ddp_config=ddp_config,
            wrap_with_ddp=True,
            bf16=self.param_dtype == torch.bfloat16,
            fp16=self.param_dtype == torch.float16,
        )

    def _build_optimizer(self):
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

        opt_config = OptimizerConfig(
            optimizer="adam",
            lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            weight_decay=self.config.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=self.config.grad_clip,
            bf16=self.param_dtype == torch.bfloat16,
            fp16=self.param_dtype == torch.float16,
            use_distributed_optimizer=self.config.use_distributed_optimizer,
        )

        self.optimizer = get_megatron_optimizer(opt_config, self.model)

        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        self.lr_scheduler = OptimizerParamScheduler(
            optimizer=self.optimizer,
            init_lr=0.0,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            lr_warmup_steps=self.config.warmup_steps,
            lr_decay_steps=self.config.total_steps,
            lr_decay_style="cosine",
            start_wd=self.config.weight_decay,
            end_wd=self.config.weight_decay,
            wd_incr_steps=0,
            wd_incr_style="constant",
        )

    def train_step(self, micro_batches: List[Dict[str, torch.Tensor]]) -> dict:
        """Execute one training step with gradient accumulation over micro-batches."""
        for model_chunk in self.model:
            model_chunk.train()
            if hasattr(model_chunk, 'zero_grad_buffer'):
                model_chunk.zero_grad_buffer()

        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=self._forward_step,
            data_iterator=iter(micro_batches),
            model=self.model,
            num_microbatches=len(micro_batches),
            seq_length=micro_batches[0]["input_ids"].shape[1],
            micro_batch_size=micro_batches[0]["input_ids"].shape[0],
            forward_only=False,
        )

        step_result = self.optimizer.step()
        grad_norm = step_result[1] if isinstance(step_result, tuple) and len(step_result) > 1 else None
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(increment=1)
        self.optimizer.zero_grad()

        output = {}
        if mpu.is_pipeline_last_stage():
            avg_loss = sum(l["loss"].item() for l in losses_reduced) / len(losses_reduced)
            output["loss"] = avg_loss
            output["lr"] = self.optimizer.param_groups[0].get("lr", self.config.learning_rate)
            if grad_norm is not None:
                output["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

            moe_metrics = self._get_moe_metrics(len(micro_batches))
            if moe_metrics:
                output.update(moe_metrics)

        return output

    @torch.no_grad()
    def eval_step(self, micro_batches: List[Dict[str, torch.Tensor]]) -> dict:
        """Forward-only evaluation step."""
        for model_chunk in self.model:
            model_chunk.eval()

        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=self._forward_step,
            data_iterator=iter(micro_batches),
            model=self.model,
            num_microbatches=len(micro_batches),
            seq_length=micro_batches[0]["input_ids"].shape[1],
            micro_batch_size=micro_batches[0]["input_ids"].shape[0],
            forward_only=True,
        )

        output = {}
        if mpu.is_pipeline_last_stage():
            avg_loss = sum(l["loss"].item() for l in losses_reduced) / len(losses_reduced)
            output["loss"] = avg_loss

        return output

    def _forward_step(self, batch_iter, model):
        """Forward step: run causal LM, extract yes/no scores, compute reranker loss."""
        batch = next(batch_iter)
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        position_ids = batch["position_ids"].cuda()
        teacher_scores = batch.get("teacher_scores")
        if teacher_scores is not None:
            teacher_scores = teacher_scores.cuda()

        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        def loss_func(output_tensor):
            # if torch.distributed.get_rank() == 0:
            #     logger.info(f"output_tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

            output_bsv = output_tensor  # (B, S, V)
            B = output_bsv.size(0)
            seq_lengths = attention_mask.sum(dim=-1) - 1
            last_logits = output_bsv[torch.arange(B, device=output_bsv.device), seq_lengths]  # (B, V)

            yes_logits = last_logits[:, self.yes_token_id]
            no_logits = last_logits[:, self.no_token_id]
            scores = yes_logits - no_logits  # (B,)

            G = self.config.train_group_size
            grouped_scores = scores.view(-1, G)  # (n_groups, G); positive at index 0

            if self.config.loss_kind == "contrastive":
                labels = torch.zeros(grouped_scores.size(0), dtype=torch.long, device=scores.device)
                loss = torch.nn.functional.cross_entropy(grouped_scores, labels)
            elif self.config.loss_kind == "distill":
                if teacher_scores is None:
                    raise RuntimeError(
                        "loss_kind='distill' but batch has no 'teacher_scores'. "
                        "Use MegatronRerankerDistilCollator."
                    )
                t_temp = self.config.teacher_temp
                s_temp = self.config.student_temp
                student = grouped_scores / s_temp
                teacher = teacher_scores.view(-1, G) / t_temp
                student_lp = torch.nn.functional.log_softmax(student, dim=-1)
                teacher_p = torch.nn.functional.softmax(teacher, dim=-1)
                # batchmean: averages KL over groups, matching the contrastive
                # CE which is also a per-group-then-mean reduction.
                loss = torch.nn.functional.kl_div(
                    student_lp, teacher_p, reduction="batchmean"
                ) * (s_temp ** 2)
            else:
                raise ValueError(f"Unknown loss_kind: {self.config.loss_kind!r}")

            # if torch.distributed.get_rank() == 0:
            #     logger.info(
            #         f"yes_logits min/max/mean: {yes_logits.min():.2f}/{yes_logits.max():.2f}/{yes_logits.mean():.2f}, "
            #         f"no_logits min/max/mean: {no_logits.min():.2f}/{no_logits.max():.2f}/{no_logits.mean():.2f}"
            #     )
            #     logger.info(f"grouped[0] scores (yes-no): {[f'{x:.2f}' for x in grouped_scores[0].tolist()]}")
            #     logger.info(f"loss: {loss.item():.6f}")

            # Megatron's pipeline scheduler does `output_tensor /= num_microbatches`
            # in-place after we return; `.detach()` shares storage and would be
            # mutated too. Clone so the logged value is the raw per-microbatch loss.
            return loss, {"loss": loss.detach().clone()}

        return output, loss_func

    def _get_moe_metrics(self, n_micro_batches: int) -> dict:
        """Drain Megatron's MoE aux loss tracker and return metrics for logging."""
        try:
            from megatron.core.transformer.moe.moe_utils import (
                clear_aux_losses_tracker,
                track_moe_metrics,
            )
        except ImportError:
            return {}

        if getattr(self.tf_config, "num_moe_experts", None) is None:
            return {}

        lb_type = getattr(self.tf_config, "moe_router_load_balancing_type", "none")
        if lb_type == "none":
            return {}

        _LB_MAP = {
            "aux_loss": "load_balancing_loss",
            "seq_aux_loss": "seq_load_balancing_loss",
            "global_aux_loss": "global_load_balancing_loss",
        }

        if isinstance(lb_type, list):
            track_names = [_LB_MAP[t] for t in lb_type if t in _LB_MAP]
        elif lb_type in _LB_MAP:
            track_names = [_LB_MAP[lb_type]]
        else:
            return {}

        if getattr(self.tf_config, "moe_z_loss_coeff", None):
            track_names.append("z_loss")

        total_loss_dict = {}
        track_moe_metrics(
            loss_scale=1.0 / n_micro_batches,
            iteration=0,
            writer=None,
            wandb_writer=None,
            total_loss_dict=total_loss_dict,
            per_layer_logging=False,
            track_names=track_names,
            num_layers=getattr(self.tf_config, "num_layers", None),
            moe_layer_freq=getattr(self.tf_config, "moe_layer_freq", None),
        )
        clear_aux_losses_tracker()

        output = {}
        for key, value in total_loss_dict.items():
            output[f"moe/{key}"] = value.cpu().item() if hasattr(value, "cpu") else float(value)
        return output

    def save_checkpoint(self, save_dir: str, step: int):
        """Save model weights back to HF format with config and tokenizer.

        For LoRA runs (use_lora=True) we let the megatron-bridge save path
        merge adapters into the base on the fly (merge_adapter_weights=True),
        producing a regular HF checkpoint that the existing eval pipeline
        (vLLM / HF + DDP) can load with no special handling. The bridge's
        save_hf_pretrained(merge_adapter_weights=False) path doesn't actually
        emit a PEFT-format adapter dir — it tries to write base_layer/lora_A/B
        keys against an HF state schema that only has 'weight' keys, which
        fails strict matching. Adapter-only export goes through
        export_adapter_weights, which we'd have to wrap ourselves; not worth
        it given the eval workflow is full-checkpoint anyway.
        For full-FT runs we use the mbridge save_weights path.
        """
        if torch.distributed.get_rank() == 0:
            os.makedirs(save_dir, exist_ok=True)

        torch.distributed.barrier()

        step_dir = os.path.join(save_dir, f"step_{step}")

        if self.config.use_lora:
            self.bridge.save_hf_pretrained(
                self.model, step_dir, merge_adapter_weights=True
            )
        else:
            self.bridge.save_weights(self.model, step_dir)

        if torch.distributed.get_rank() == 0:
            from transformers import AutoTokenizer
            hf_config = AutoConfig.from_pretrained(self.config.model_name_or_path)
            hf_config.save_pretrained(step_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            tokenizer.save_pretrained(step_dir)
            logger.info(f"Saved checkpoint at step {step} to {step_dir}")

    def get_data_parallel_rank(self) -> int:
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self) -> int:
        return mpu.get_data_parallel_world_size()
