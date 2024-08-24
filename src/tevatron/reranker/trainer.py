import logging
from typing import Dict, Union, Any

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from grad_cache import GradCache

from grad_cache.functional import cached, cat_input_tensor
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


@cached
@autocast()
def get_model_rep(model, inputs):
    outputs = model(**inputs)
    return outputs.scores


@cat_input_tensor
@autocast()
def contrastive_loss(scores):
    batch_size = scores.size(0) // 2
    labels = torch.arange(batch_size, device=scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


def split_inputs(model_input, chunk_size):
    logger.debug(f"Splitting inputs with chunk size: {chunk_size}")
    keys = list(model_input.keys())
    chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]


class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initializing RerankerTrainer with GradCache")
        self.args: TrainingArguments

        # Add these lines to include the necessary parameters
        self.gc_chunk_size = getattr(self.args, 'gc_chunk_size', 4)  # default to 4 if not provided

        self.gc = GradCache(
            models=[self.model],
            chunk_sizes=self.gc_chunk_size,
            loss_fn=contrastive_loss,
            split_input_fn=split_inputs,
            get_rep_fn=lambda x: x.scores,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )
        logger.info(f"GradCache initialized with chunk size: {self.gc_chunk_size}")

    def compute_loss(self, model, inputs, return_outputs=False):
        logger.debug(f"Computing loss with inputs: {inputs.keys()}")
        outputs = model(**inputs)
        scores = outputs.scores
        loss = contrastive_loss(scores)
        logger.debug(f"Computed loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        logger.debug("Entering training step")
        model.train()
        inputs = self._prepare_inputs(inputs)
        _distributed = self.args.local_rank > -1
        loss = self.gc(inputs, no_sync_except_last=_distributed)
        logger.debug(f"Training step loss: {loss.item()}")
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: bool = None,
    ) -> PredictionOutput:
        logger.debug("Entering prediction step")
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.scores
        loss = contrastive_loss(scores)
        logger.debug(f"Prediction step loss: {loss.item() if loss is not None else 'N/A'}")
        return PredictionOutput(predictions=scores, label_ids=None, metrics=None)
