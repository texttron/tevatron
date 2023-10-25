from dataclasses import dataclass
from typing import Dict, Optional

from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.file_utils import ModelOutput
from peft import PeftConfig, PeftModel


import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForSequenceClassification

    def __init__(self, hf_model: PreTrainedModel):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pair: Dict[str, Tensor] = None):
        ranker_logits = self.hf_model(**pair, return_dict=True).logits
        return RerankerOutput(
            loss = None,
            scores = ranker_logits
        )

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        config = PeftConfig.from_pretrained(model_name_or_path)
        base_model = cls.TRANSFORMER_CLS.from_pretrained(config.base_model_name_or_path, num_labels=1, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = base_model.config.eos_token_id
        hf_model = PeftModel.from_pretrained(base_model, model_name_or_path)
        hf_model = hf_model.merge_and_unload()
        model = cls(hf_model=hf_model)
        return model
