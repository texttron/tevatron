import os
from typing import Optional

import torch

from transformers.trainer import Trainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

import logging
logger = logging.getLogger(__name__)


class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RerankerTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

        if is_deepspeed_zero3_enabled():
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'hf_model.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            lora_state_dict = get_peft_model_state_dict(self.model.hf_model, state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Save adapter model at {output_dir}")

    def compute_loss(self, model, inputs):
        return model(inputs).loss
