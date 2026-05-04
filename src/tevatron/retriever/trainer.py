import os
from typing import Optional

import torch

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from .modeling import EncoderModel

import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _wrap_model(self, model, training=True, dataloader=None):
        """Override to enable static_graph for DDP with gradient checkpointing."""
        wrapped = super()._wrap_model(model, training, dataloader)
        # Enable static graph to handle gradient checkpointing
        if self.is_ddp and self.args.gradient_checkpointing and hasattr(wrapped, '_set_static_graph'):
            wrapped._set_static_graph()
            logger.info("Enabled DDP static graph mode")
        return wrapped

    def _dist_gather_tensor(self, t: Optional[torch.Tensor], name: str = "tensor") -> Optional[torch.Tensor]:
        """Gather tensor from all ranks, concatenating along dim 0 (batch dimension).

        CRITICAL: All ranks must call this with either ALL None or ALL non-None tensors.
        Otherwise NCCL will deadlock!

        Args:
            t: Tensor to gather (or None)
            name: Name of tensor for debugging
        """
        if not self.is_ddp:
            return t

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Check if this rank has None - need to sync this info across all ranks
        has_tensor = torch.tensor([t is not None], dtype=torch.int, device='cuda')
        has_tensor_list = [torch.zeros_like(has_tensor) for _ in range(world_size)]
        dist.all_gather(has_tensor_list, has_tensor)

        # If ALL ranks have None, return None (no need to gather)
        if not any(ht.item() for ht in has_tensor_list):
            return None

        # If SOME ranks have None but others don't, this is a bug!
        all_have_tensor = all(ht.item() for ht in has_tensor_list)

        if not all_have_tensor:
            if t is None:
                raise RuntimeError(
                    f"[Rank {rank}] {name} is None but other ranks have non-None tensors. "
                    "This causes NCCL deadlock. Ensure all ranks produce consistent outputs."
                )

        # All ranks have non-None tensors - safe to gather
        t = t.contiguous()

        # First, verify all ranks have the same shape (except dim 0)
        # This is critical for all_gather to work correctly
        shape_tensor = torch.tensor(list(t.shape), dtype=torch.long, device=t.device)
        shape_list = [torch.zeros_like(shape_tensor) for _ in range(world_size)]
        dist.all_gather(shape_list, shape_tensor)

        # Check shape consistency (all dims except dim 0 must match)
        local_shape = list(t.shape)
        for other_rank, other_shape_tensor in enumerate(shape_list):
            other_shape = other_shape_tensor.tolist()
            if len(local_shape) != len(other_shape):
                raise RuntimeError(
                    f"[Rank {rank}] {name} has {len(local_shape)} dims but rank {other_rank} "
                    f"has {len(other_shape)} dims. Shapes: local={local_shape}, other={other_shape}"
                )
            # Check all dims except dim 0
            for dim_idx in range(1, len(local_shape)):
                if local_shape[dim_idx] != other_shape[dim_idx]:
                    raise RuntimeError(
                        f"[Rank {rank}] {name} shape mismatch at dim {dim_idx}: "
                        f"local={local_shape}, rank {other_rank}={other_shape}. "
                        "All ranks must have the same shape except dim 0 for gathering."
                    )

        # Create placeholders for gathering
        all_tensors = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(all_tensors, t)

        # Keep local tensor for gradient flow (important for backprop!)
        all_tensors[rank] = t

        # Concatenate along dim 0 (batch dimension)
        gathered = torch.cat(all_tensors, dim=0)

        return gathered

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=getattr(self.args, "save_safetensors", False)
            )

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, *rest = inputs

        # Unwrap DDP model first (important for attribute access)
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Set eos_token_id from tokenizer if available (for EOS verification in encode_passage)
        if self.tokenizer is not None and hasattr(self.tokenizer, 'eos_token_id'):
            unwrapped_model.eos_token_id = self.tokenizer.eos_token_id

        # Determine whether rest[0] is eos_positions or chunk_counts
        extra = rest[0] if rest else None
        if extra is not None and getattr(unwrapped_model, 'passage_chunk_independent', False):
            # Independent chunk mode: rest[0] is chunk_counts (List[int])
            unwrapped_model.chunk_counts = extra

            if self.is_ddp:
                logger.debug(
                    f"[Rank {dist.get_rank()}] chunk_counts: {extra}, "
                    f"total_chunks={sum(extra)}, max_chunks={max(extra) if extra else 0}"
                )
        elif extra is not None:
            # Standard chunked mode: rest[0] is eos_positions
            eos_positions = extra
            unwrapped_model.eos_positions = eos_positions

            # Debug logging for DDP
            if self.is_ddp:
                chunk_counts = [len(ep) for ep in eos_positions]
                max_local_chunks = max(chunk_counts) if chunk_counts else 0
                logger.debug(
                    f"[Rank {dist.get_rank()}] eos_positions: {len(eos_positions)} passages, "
                    f"chunk_counts={chunk_counts}, max_local_chunks={max_local_chunks}"
                )

        # Set static graph on first call if needed (for gradient_checkpointing + DDP)
        if self.is_ddp and self.args.gradient_checkpointing and not hasattr(self, '_static_graph_set'):
            if hasattr(model, '_set_static_graph'):
                model._set_static_graph()
                logger.info("Enabled DDP static graph mode in compute_loss")
            self._static_graph_set = True

        # Get embeddings from model (forward() will read eos_positions from unwrapped_model)
        output = model(query=query, passage=passage)
        q_reps = output.q_reps
        p_reps = output.p_reps
        chunk_mask = output.chunk_mask

        # Debug logging for shapes before gathering
        if self.is_ddp:
            logger.debug(
                f"[Rank {dist.get_rank()}] Before gather - q_reps: {q_reps.shape}, "
                f"p_reps: {p_reps.shape}, chunk_mask: {chunk_mask.shape if chunk_mask is not None else None}"
            )

        # Gather embeddings from all ranks for in-batch negatives
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps, name="q_reps")
            p_reps = self._dist_gather_tensor(p_reps, name="p_reps")
            chunk_mask = self._dist_gather_tensor(chunk_mask, name="chunk_mask")
        
        # Compute similarity (use maxsim for chunked passages)
        # Note: unwrapped_model was already defined above
        
        # Defensive check: ensure p_reps shape matches expected usage
        if unwrapped_model.passage_chunk_size > 0 and chunk_mask is not None:
            # Chunked passage mode: p_reps should be [P, C, H]
            if p_reps.dim() != 3:
                raise ValueError(
                    f"Expected 3D p_reps [P, C, H] for chunked passages, got shape {p_reps.shape}. "
                    f"passage_chunk_size={unwrapped_model.passage_chunk_size}, chunk_mask shape={chunk_mask.shape}"
                )
            scores = unwrapped_model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
        else:
            # Regular mode: p_reps should be [P, H]
            if p_reps.dim() == 3:
                # Edge case: chunked but no mask - shouldn't happen but handle gracefully
                p_reps = p_reps.mean(dim=1)  # [P, C, H] -> [P, H]
            elif p_reps.dim() != 2:
                raise ValueError(f"Expected 2D p_reps [P, H], got shape {p_reps.shape}")
            
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        
        scores = scores.view(q_reps.size(0), -1)
        
        # Create target labels: each query should match its corresponding passage
        # After DDP gather, we have Q total queries and P total passages
        # If we have N passages per query, then query i should match passage i*N
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        num_passages_per_query = p_reps.size(0) // q_reps.size(0)
        target = target * num_passages_per_query
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(scores / unwrapped_model.temperature, target)
        if self.is_ddp:
            loss = loss * dist.get_world_size()
        return loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


class DistilTevatronTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, reranker_labels = inputs
        scores = model(query=query, passage=passage).scores
        
        if model.is_ddp:
            # reranker_scores are gathered across all processes
            reranker_labels = model._dist_gather_tensor(reranker_labels)
        
        # Derive student_scores [batch, num_labels]
        batch_size, total_passages = scores.size()
        num_labels = reranker_labels.size(1)
        start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
        idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
        student_scores = scores.gather(1, idx_matrix)

        # Temperature‐scaled soft distributions
        T = self.args.distil_temperature
        student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
        teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)

        # KL Divergence loss (shapes now [batch, num_labels])
        loss = torch.nn.functional.kl_div(
            student_log,
            teacher_probs,
            reduction="batchmean"
        ) * self._dist_loss_scale_factor

        return loss

    def training_step(self, *args):
        return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor
