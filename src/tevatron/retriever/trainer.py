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

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Gather tensor from all ranks, concatenating along dim 0 (batch dimension)."""
        if t is None:
            return None
        t = t.contiguous()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Create placeholders for gathering
        all_tensors = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(all_tensors, t)
        
        # Keep local tensor for gradient flow (important for backprop!)
        all_tensors[rank] = t
        
        # Concatenate along dim 0 (batch dimension)
        # Works for 2D [B, H], 3D [B, C, H], etc.
        gathered = torch.cat(all_tensors, dim=0)
        
        logger.debug(f"[Gather] rank={rank}, input shape={t.shape}, output shape={gathered.shape}")
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
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, *rest = inputs
        eos_positions = rest[0] if rest else None
        print(f"[DEBUG Trainer] Received {len(rest)} extra inputs, eos_positions: {eos_positions}")
        
        # Unwrap DDP model first (important for attribute access)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        
        # Attach eos_positions to the unwrapped model (for passage chunking)
        # This is critical: the forward() method reads this attribute via getattr(self, "eos_positions", None)
        if eos_positions is not None:
            unwrapped_model.eos_positions = eos_positions
            logger.info(f"[Chunking] Set eos_positions with {len(eos_positions)} passages, "
                       f"passage_chunk_size={unwrapped_model.passage_chunk_size}")
        else:
            logger.info(f"[Chunking] No eos_positions provided, passage_chunk_size={unwrapped_model.passage_chunk_size}")
        
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
        
        # Log what we got back from forward()
        logger.info(f"[Forward] Got q_reps: {q_reps.shape if q_reps is not None else None}, "
                   f"p_reps: {p_reps.shape if p_reps is not None else None}, "
                   f"chunk_mask: {chunk_mask.shape if chunk_mask is not None else None}")
        
        # Gather embeddings from all ranks for in-batch negatives
        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            chunk_mask = self._dist_gather_tensor(chunk_mask)
            logger.info(f"[DDP] After gather - q_reps: {q_reps.shape}, p_reps: {p_reps.shape}, chunk_mask: {chunk_mask.shape if chunk_mask is not None else None}")
        
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
            logger.info(f"[MaxSim] Computing with q_reps: {q_reps.shape}, p_reps: {p_reps.shape}, chunk_mask: {chunk_mask.shape}")
            scores = unwrapped_model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
            logger.info(f"[MaxSim] Output scores shape: {scores.shape}")
        else:
            # Regular mode: p_reps should be [P, H]
            if p_reps.dim() == 3:
                # Edge case: chunked but no mask - shouldn't happen but handle gracefully
                logger.warning(
                    f"Got 3D p_reps {p_reps.shape} but chunk_mask is None. "
                    f"Averaging over chunks as fallback."
                )
                p_reps = p_reps.mean(dim=1)  # [P, C, H] -> [P, H]
            elif p_reps.dim() != 2:
                raise ValueError(f"Expected 2D p_reps [P, H], got shape {p_reps.shape}")
            
            logger.info(f"[Regular] Computing with q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            logger.info(f"[Regular] Output scores shape: {scores.shape}")
        
        scores = scores.view(q_reps.size(0), -1)
        
        # Create target labels: each query should match its corresponding passage
        # After DDP gather, we have Q total queries and P total passages
        # If we have N passages per query, then query i should match passage i*N
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        num_passages_per_query = p_reps.size(0) // q_reps.size(0)
        target = target * num_passages_per_query
        
        logger.info(f"[Loss] scores shape: {scores.shape}, target shape: {target.shape}, "
                   f"passages_per_query: {num_passages_per_query}")
        
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
