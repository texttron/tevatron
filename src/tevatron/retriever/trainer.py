import os
from typing import Optional

import torch

from transformers.trainer import Trainer
import torch.distributed as dist
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict, PeftModel


import logging
logger = logging.getLogger(__name__)


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            if isinstance(self.model.encoder, PeftModel):
                lora_state_dict = get_peft_model_state_dict(self.model.encoder, state_dict)
                if self.args.process_index <= 0:
                    torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                    print(f"Save adapter model at {output_dir}")
            else:
                if self.args.process_index <= 0:
                    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    print(f"Save model at {output_dir}")

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

