import os
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.trainer import Trainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

import logging

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


def split_inputs(model_input: dict, chunk_size: int):
    keys = list(model_input.keys())
    chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]


def get_rep(x):
    return x.logits


class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RerankerTrainer, self).__init__(*args, **kwargs)

        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')

        self.gc = GradCache(
            models=[self.model],
            chunk_sizes=[self.args.gc_chunk_size],
            loss_fn=self.compute_loss,
            split_input_fn=split_inputs,
            get_rep_fn=get_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'hf_model.'
            assert all(
                k.startswith(prefix) or k == "target_label"
                for k in state_dict.keys()
            ), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            lora_state_dict = get_peft_model_state_dict(self.model.hf_model, state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Save adapter model at {output_dir}")

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        _distributed = self.args.local_rank > -1
        self.gc.models = [model]
        loss = self.gc(inputs, no_sync_except_last=_distributed)
        return loss
