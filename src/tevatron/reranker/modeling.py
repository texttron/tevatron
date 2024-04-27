import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


from tevatron.reranker.arguments import ModelArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForSequenceClassification

    def __init__(self, hf_model: PreTrainedModel, train_batch_size: int = None):
        super().__init__()
        self.config = hf_model.config
        self.hf_model = hf_model
        self.train_batch_size = train_batch_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_batch_size:
            self.register_buffer(
                'target_label',
                torch.zeros(self.train_batch_size, dtype=torch.long, device=self.hf_model.device)
            )
        for name, param in self.hf_model.named_parameters():
            # for some reason, ds zero 3 left some weights empty
            if 'modules_to_save' in name and param.numel() == 0:
                logger.warning(f'parameter {name}, shape {param.shape} is empty')
                param.data = nn.Linear(self.hf_model.config.hidden_size, 1).weight.data
                logger.warning('{} data: {}'.format(name, param.data.cpu().numpy()))

    def forward(self, pair: Dict[str, Tensor] = None):
        ranker_logits = self.hf_model(**pair, return_dict=True).logits
        if self.train_batch_size:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            loss = self.cross_entropy(grouped_logits, self.target_label)
            return RerankerOutput(
                loss = loss,
                scores = ranker_logits
            )

        return RerankerOutput(
            loss = None,
            scores = ranker_logits
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.hf_model.base_model.model.gradient_checkpointing_enable(**kwargs)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            **hf_kwargs,
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.SEQ_CLS,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                hf_model=lora_model,
                train_batch_size=train_args.per_device_train_batch_size,
            )
        else:
            model = cls(
                hf_model=base_model,
                train_batch_size=train_args.per_device_train_batch_size,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, num_labels=1, **hf_kwargs, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                hf_model=lora_model,
            )
        else:
            model = cls(
                hf_model=base_model,
            )
        return model

    def save(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
