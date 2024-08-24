import logging
from typing import Dict, Union, Any

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput

from grad_cache import GradCache

from tevatron.reranker.arguments import TevatronTrainingArguments

logger = logging.getLogger(__name__)

def split_inputs(model_input, chunk_size):
    logger.debug(f"Splitting inputs with chunk size: {chunk_size}")
    keys = list(model_input.keys())
    chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

def get_rep(model_output):
    logger.debug(f"Getting representation from model output: {type(model_output)}")
    return model_output.scores

class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initializing RerankerTrainer")
        self.args: TevatronTrainingArguments

        def loss_fn(scores, labels):
            grouped_scores = scores.view(self.args.train_group_size, -1)
            labels = torch.zeros(self.args.train_group_size, dtype=torch.long, device=scores.device)
            return nn.CrossEntropyLoss()(grouped_scores, labels)

        self.gc = GradCache(
            models=[self.model],
            chunk_sizes=[self.args.gc_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_inputs,
            get_rep_fn=get_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )
        logger.info(f"GradCache initialized with chunk size: {self.args.gc_chunk_size}")

    def compute_loss(self, model, inputs, return_outputs=False):
        logger.debug(f"Computing loss with inputs: {inputs.keys()}")
        outputs = model(**inputs)
        loss = outputs.loss
        logger.debug(f"Computed loss: {loss.item()}")
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        logger.debug("Entering training step")
        model.train()
        inputs = self._prepare_inputs(inputs)
        _distributed = self.args.local_rank > -1
        self.gc.models = [model]
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
        loss = outputs.loss
        logits = outputs.scores
        logger.debug(f"Prediction step loss: {loss.item() if loss is not None else 'N/A'}")
        return PredictionOutput(predictions=logits, label_ids=inputs.get("labels"), metrics=None)
