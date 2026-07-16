import os
from typing import Optional

import torch

from transformers.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


def _is_deepspeed_zero3_enabled():
    try:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
        return is_deepspeed_zero3_enabled()
    except (ImportError, ModuleNotFoundError):
        return False


def _strip_hf_model_prefix(state_dict):
    """Drop the 'hf_model.' wrapper prefix so keys match the bare HF model.

    Backbone rerankers additionally carry a sibling 'score.' head (not part of
    hf_model) — those keys are passed through unstripped so callers can split
    them out for RerankerModel._save_reranker_extras.
    """
    prefix = 'hf_model.'
    assert all(
        k.startswith(prefix) or k.startswith('score.') or k == "target_label"
        for k in state_dict.keys()
    ), list(state_dict.keys())
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
        if k.startswith(prefix) or k.startswith('score.')
    }


class RerankerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RerankerTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # Save the tokenizer into every checkpoint dir (rank 0 only -- otherwise
        # all ranks race-write the same files). Without this, intermediate
        # checkpoint-N/ dirs have no tokenizer and eval can't resolve the yes/no
        # probe; the trainer's top-level save_pretrained doesn't reach them.
        if self.args.process_index <= 0 and self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        is_lora = (_is_deepspeed_zero3_enabled() or self.is_fsdp_enabled) \
            and hasattr(self.model.hf_model, "peft_config")

        if is_lora:
            # Adapter-only save: de-prefix the (gathered) state_dict and let PEFT
            # extract the adapter weights. Backbone rerankers also carry a
            # 'score.' head sibling to hf_model, which PEFT doesn't know about —
            # split it out and save it (+ reranker_config.json) separately.
            from peft import get_peft_model_state_dict
            if state_dict is None:
                state_dict = self.model.state_dict()
            state_dict = _strip_hf_model_prefix(state_dict)
            score_state_dict = {k: v for k, v in state_dict.items() if k.startswith('score.')}
            hf_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('score.')}
            lora_state_dict = get_peft_model_state_dict(self.model.hf_model, hf_state_dict)
            if self.args.process_index <= 0:
                torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                print(f"Save adapter model at {output_dir}")
                if getattr(self.model, "reranker_model_type", None) == "backbone":
                    self.model._save_reranker_extras(output_dir, state_dict=score_state_dict)
            return

        # Full-FT save. Under FSDP full_shard the Trainer hands in a gathered
        # full-rank state_dict; pass it through (de-prefixed) so save_pretrained
        # writes real weights, not this rank's shard. Plain DDP -> state_dict is
        # None and save_pretrained pulls from the (replicated) model directly.
        if state_dict is not None:
            state_dict = _strip_hf_model_prefix(state_dict)
        self.model.save(output_dir, state_dict=state_dict)


    def compute_loss(self, model, inputs, **kwargs):
        return model(inputs).loss
