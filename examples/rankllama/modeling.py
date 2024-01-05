from dataclasses import dataclass
from typing import Dict, Optional
import torch
from tevatron.arguments import ModelArguments, TrainingArguments

from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.file_utils import ModelOutput
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType


import logging


logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class RerankerModel(nn.Module):

    def __init__(self, hf_model: PeftModel, train_batch_size: int=None):
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_model.config
        self.train_batch_size = train_batch_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, pair: Dict[str, Tensor] = None):
        ranker_logits = self.hf_model(**pair, return_dict=True).logits
        if self.train_batch_size:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            target = torch.zeros(self.train_batch_size, dtype=torch.long)
            loss = self.cross_entropy(grouped_logits, target.to(grouped_logits.device))
            return RerankerOutput(
                loss = loss,
                scores = ranker_logits
            )

        return RerankerOutput(
            loss = None,
            scores = ranker_logits
        )

    def gradient_checkpointing_enable(self):
        self.hf_model.base_model.model.gradient_checkpointing_enable()

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        config = PeftConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        hf_model = PeftModel.from_pretrained(base_model, model_name_or_path)
        hf_model = hf_model.merge_and_unload()
        model = cls(hf_model=hf_model)
        return model


    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=1, **hf_kwargs)
        if train_args.gradient_checkpointing:
            base_model.enable_input_require_grads()
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            
        peft_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False
            )
        
        peft_model = get_peft_model(base_model, peft_config)
        for name, param in peft_model.named_parameters():
            # for some reason, ds zero 3 left some weights empty
            if 'modules_to_save' in name and param.numel() == 0:
                logger.warning(f'parameter {name}, shape {param.shape} is empty')
                param.data = nn.Linear(4096, 1).weight.data
                logger.warning('{} data: {}'.format(name, param.data.cpu().numpy()))
        
        
        model = cls(hf_model=peft_model, train_batch_size=train_args.per_device_train_batch_size)
        return model
    
    def save(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)