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
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


class DistilTevatronTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage, reranker_labels = inputs
        # print(f"DistilTevatronTrainer: query={query}, passage={passage}, reranker_labels={reranker_labels}")
        # print the shapes of the inputs for debugging
        scores = model(query=query, passage=passage).scores
        # print(f"Scores shape: {scores}")
        # # print(f"Shapes - query: {query.size()}, passage: {passage.size()}, reranker_labels: {}, scores: {scores.size()}")
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        
        if model.is_ddp:
            # reranker_scores are gathered across all processes
            reranker_labels = model._dist_gather_tensor(reranker_labels)
        
        # normalize the reranker_labels to probabilities
        reranker_labels = torch.softmax(reranker_labels.view(scores.size(0), -1), dim=1, dtype=scores.dtype)
        
        num_queries, num_hn_passages = scores.size(0), reranker_labels.size(1)

        # Create a tensor of indices to gather the correct scores
        indices = torch.arange(0, num_queries * num_hn_passages, num_hn_passages, device=scores.device)
        indices = indices.view(-1, 1) + torch.arange(num_hn_passages, device=scores.device)
        scores_matrix = torch.gather(scores, 1, indices)

        # Normalize scores_matrix using softmax
        scores_matrix = torch.softmax(scores_matrix, dim=1, dtype=scores.dtype)

        loss = torch.nn.functional.kl_div(
            torch.log(scores_matrix + 1e-8),  # log(Q) - add small epsilon for numerical stability
            reranker_labels,                  # P (target distribution)
            reduction='batchmean'             # Average over batch dimension
        )
        return loss

    def training_step(self, *args):
        return super(DistilTevatronTrainer, self).training_step(*args) / self._dist_loss_scale