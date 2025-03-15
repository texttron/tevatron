import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from tevatron.reranker.arguments import ModelArguments

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForSequenceClassification

    def __init__(self, hf_model: PreTrainedModel):
        super().__init__()
        logger.info("Initializing RerankerModel")
        self.config = hf_model.config
        self.hf_model = hf_model
        logger.info(f"RerankerModel initialized with config: {self.config}")

    def forward(self, input_ids: Tensor = None, attention_mask: Tensor = None, **kwargs):
        logger.debug(f"Forward pass with input shape: {input_ids.shape if input_ids is not None else 'None'}")
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        return RerankerOutput(
            scores=outputs.logits
        )

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        logger.info(f"Building RerankerModel with args: {model_args}")
        base_model = cls.TRANSFORMER_CLS.from_pretrained(
            model_args.model_name_or_path,
            **hf_kwargs,
        )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            logger.info("Set pad_token_id to 0")

        if model_args.lora or model_args.lora_name_or_path:
            logger.info("Applying LoRA")
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                logger.info(f"Loading LoRA from {model_args.lora_name_or_path}")
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path,
                                                       torch_dtype=torch.bfloat16,
                                                       attn_implementation="flash_attention_2")
            else:
                logger.info("Initializing new LoRA")
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.SEQ_CLS,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(hf_model=lora_model)
        else:
            logger.info("Building model without LoRA")
            model = cls(hf_model=base_model)
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             lora_name_or_path: str = None,
             **hf_kwargs):
        logger.info(f"Loading RerankerModel from {model_name_or_path}")
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, num_labels=1, **hf_kwargs,
                                                         torch_dtype=torch.bfloat16,
                                                         attn_implementation="flash_attention_2")
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            logger.info("Set pad_token_id to 0")
        if lora_name_or_path:
            logger.info(f"Loading LoRA from {lora_name_or_path}")
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(hf_model=lora_model)
        else:
            logger.info("Loading model without LoRA")
            model = cls(hf_model=base_model)
        return model

    def save(self, output_dir: str):
        logger.info(f"Saving model to {output_dir}")
        self.hf_model.save_pretrained(output_dir)