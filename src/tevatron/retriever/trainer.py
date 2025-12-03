import os
from typing import Optional

import torch
import torch.nn.functional as F

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
        data_args = getattr(self.args, "data_args", None)
        if data_args is not None and getattr(data_args, "use_chunk_maxsim", False):
            loss = self.compute_loss_chunk_maxsim(model, inputs)
            return (loss, None) if return_outputs else loss

        query, passage = inputs
        return model(query=query, passage=passage).loss

    def compute_loss_chunk_maxsim(self, model, inputs):
        query_inputs = {
            "input_ids": inputs["query_input_ids"],
            "attention_mask": inputs["query_attention_mask"],
        }
        if "query_token_type_ids" in inputs:
            query_inputs["token_type_ids"] = inputs["query_token_type_ids"]

        passage_inputs = {
            "input_ids": inputs["passage_input_ids"],
            "attention_mask": inputs["passage_attention_mask"],
        }
        if "passage_token_type_ids" in inputs:
            passage_inputs["token_type_ids"] = inputs["passage_token_type_ids"]

        chunk_to_passage = inputs["chunk_to_passage"]

        query_reps = model.encode_query(query_inputs)
        chunk_reps = model.encode_passage(passage_inputs)

        if model.is_ddp:
            query_reps = model._dist_gather_tensor(query_reps)
            chunk_reps = model._dist_gather_tensor(chunk_reps)

            local_passages = chunk_to_passage.max().item() + 1 if chunk_to_passage.numel() else 0
            counts_tensor = chunk_to_passage.new_tensor([local_passages])
            gathered_counts = [torch.zeros_like(counts_tensor) for _ in range(model.world_size)]
            dist.all_gather(gathered_counts, counts_tensor)
            counts = torch.cat(gathered_counts).tolist()
            passage_offset = int(sum(counts[:model.process_rank]))
            chunk_to_passage = chunk_to_passage + passage_offset
            chunk_to_passage = model._dist_gather_tensor(chunk_to_passage.unsqueeze(1)).squeeze(1)

        num_passages = chunk_to_passage.max().item() + 1 if chunk_to_passage.numel() else 0
        if num_passages == 0:
            raise ValueError("chunk_to_passage is empty; cannot compute MaxSim loss.")

        sim_q_chunk = model.compute_similarity(query_reps, chunk_reps)

        B = query_reps.size(0)
        device = query_reps.device
        scores = torch.full(
            (B, num_passages),
            fill_value=-1e9,
            dtype=sim_q_chunk.dtype,
            device=device,
        )

        passage_idx = chunk_to_passage.unsqueeze(0).expand(B, -1)
        if hasattr(scores, "scatter_reduce_"):
            scores.scatter_reduce_(
                dim=1,
                index=passage_idx,
                src=sim_q_chunk,
                reduce="amax",
                include_self=True,
            )
        else:
            scores = scores.scatter_reduce(
                dim=1,
                index=passage_idx,
                src=sim_q_chunk,
                reduce="amax",
                include_self=True,
            )

        passages_per_query = getattr(getattr(self.args, "data_args", None), "train_group_size", None)
        if not passages_per_query:
            if num_passages % B != 0:
                raise ValueError("Number of passages is not divisible by number of queries.")
            passages_per_query = num_passages // B

        labels = torch.arange(B, device=device) * passages_per_query

        loss = F.cross_entropy(scores / model.temperature, labels)
        if model.is_ddp:
            loss = loss * model.world_size

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
