import os
import json
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoModelForSequenceClassification, PreTrainedModel
from transformers.file_utils import ModelOutput
from transformers import TrainingArguments
from tevatron.reranker.arguments import ModelArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForSequenceClassification
    BACKBONE_CLS = AutoModel
    RERANKER_CONFIG = "reranker_config.json"
    RERANKER_HEAD = "reranker_head.pt"

    def __init__(self, hf_model: PreTrainedModel, train_batch_size: int = None,
                 reranker_model_type: str = "sequence_classification",
                 reranker_pooling: str = "last"):
        super().__init__()
        self.config = hf_model.config
        self.hf_model = hf_model
        self.train_batch_size = train_batch_size
        self.reranker_model_type = reranker_model_type
        self.reranker_pooling = reranker_pooling
        if self.reranker_model_type == "backbone":
            self.score = nn.Linear(self.hf_model.config.hidden_size, 1, bias=False)
            self.score.to(dtype=next(self.hf_model.parameters()).dtype)
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
        if self.reranker_model_type == "backbone":
            outputs = self.hf_model(**pair, return_dict=True)
            ranker_logits = self.score(self._pool_sequence(outputs.last_hidden_state, pair))
        else:
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

    def _pool_sequence(self, last_hidden_state: Tensor, pair: Dict[str, Tensor]):
        if self.reranker_pooling == "cls":
            return last_hidden_state[:, 0]
        if self.reranker_pooling != "last":
            raise ValueError(f"Unsupported reranker_pooling: {self.reranker_pooling}")

        if "attention_mask" not in pair:
            return last_hidden_state[:, -1]
        sequence_lengths = pair["attention_mask"].long().sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_idx, sequence_lengths]

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.hf_model, "gradient_checkpointing_enable"):
            self.hf_model.gradient_checkpointing_enable(**kwargs)
        else:
            self.hf_model.base_model.model.gradient_checkpointing_enable(**kwargs)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        if model_args.attn_implementation:
            hf_kwargs["attn_implementation"] = model_args.attn_implementation

        if model_args.reranker_model_type == "backbone":
            base_model = cls.BACKBONE_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        else:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path,
                num_labels=1,
                **hf_kwargs,
            )
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION if model_args.reranker_model_type == "backbone" else TaskType.SEQ_CLS,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False,
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                hf_model=lora_model,
                train_batch_size=train_args.per_device_train_batch_size,
                reranker_model_type=model_args.reranker_model_type,
                reranker_pooling=model_args.reranker_pooling,
            )
        else:
            model = cls(
                hf_model=base_model,
                train_batch_size=train_args.per_device_train_batch_size,
                reranker_model_type=model_args.reranker_model_type,
                reranker_pooling=model_args.reranker_pooling,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             lora_name_or_path: str = None,
             **hf_kwargs):
        reranker_config = cls._load_reranker_config(model_name_or_path)
        reranker_model_type = reranker_config.get("reranker_model_type", "sequence_classification")
        reranker_pooling = reranker_config.get("reranker_pooling", "last")
        if reranker_config.get("attn_implementation"):
            hf_kwargs["attn_implementation"] = reranker_config["attn_implementation"]

        if reranker_model_type == "backbone":
            from peft import PeftConfig, PeftModel
            adapter_path = lora_name_or_path
            base_model_name_or_path = model_name_or_path
            if adapter_path is None and os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
                adapter_path = model_name_or_path
                base_model_name_or_path = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
            base_model = cls.BACKBONE_CLS.from_pretrained(base_model_name_or_path, **hf_kwargs)
        else:
            # setdefault: hf_kwargs may already carry attn_implementation, either
            # from the caller or from the checkpoint's reranker_config.json above;
            # passing the keyword again alongside **hf_kwargs would raise
            # "got multiple values for keyword argument".
            hf_kwargs.setdefault("torch_dtype", torch.bfloat16)
            hf_kwargs.setdefault("attn_implementation", "flash_attention_2")
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, num_labels=1, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if reranker_model_type == "backbone" and adapter_path:
            lora_model = PeftModel.from_pretrained(base_model, adapter_path)
            model = cls(
                hf_model=lora_model,
                reranker_model_type=reranker_model_type,
                reranker_pooling=reranker_pooling,
            )
        elif reranker_model_type != "backbone" and lora_name_or_path:
            from peft import LoraConfig, PeftModel
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                hf_model=lora_model,
            )
        else:
            model = cls(
                hf_model=base_model,
                reranker_model_type=reranker_model_type,
                reranker_pooling=reranker_pooling,
            )
        if reranker_model_type == "backbone":
            model._load_score_head(model_name_or_path)
        return model

    def save(self, output_dir: str, state_dict: Dict[str, Tensor] = None):
        # Under FSDP full_shard, the Trainer gathers a consolidated full-rank
        # state_dict and hands it in; pass it through so save_pretrained writes
        # the real weights instead of this rank's shard. Plain DDP -> state_dict
        # is None and save_pretrained pulls from the (replicated) model directly.
        # The trainer keeps any "score." keys out of what it strips/passes here
        # as hf_model weights, so they're carved out separately below.
        if state_dict is not None:
            hf_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("score.")}
        else:
            hf_state_dict = None
        self.hf_model.save_pretrained(output_dir, state_dict=hf_state_dict)
        self._save_reranker_extras(output_dir, state_dict=state_dict)

    def _save_reranker_extras(self, output_dir: str, state_dict: Dict[str, Tensor] = None):
        """Writes reranker_config.json and (for backbone rerankers) the score head.

        Shared by the full-FT save path above and the trainer's LoRA-adapter save
        path, which saves only the adapter via `get_peft_model_state_dict` and
        needs this called separately for the score head to be persisted too.
        """
        with open(os.path.join(output_dir, self.RERANKER_CONFIG), "w") as f:
            json.dump({
                "reranker_model_type": self.reranker_model_type,
                "reranker_pooling": self.reranker_pooling,
                "attn_implementation": getattr(self.hf_model.config, "_attn_implementation", None),
            }, f, indent=2)
        if self.reranker_model_type == "backbone":
            if state_dict is not None:
                score_state_dict = {
                    k[len("score."):]: v.cpu()
                    for k, v in state_dict.items()
                    if k.startswith("score.")
                }
            else:
                score_state_dict = {k: v.cpu() for k, v in self.score.state_dict().items()}
            torch.save(score_state_dict, os.path.join(output_dir, self.RERANKER_HEAD))

    @classmethod
    def _load_reranker_config(cls, model_name_or_path: str):
        config_path = os.path.join(model_name_or_path, cls.RERANKER_CONFIG)
        if not os.path.exists(config_path):
            return {}
        with open(config_path) as f:
            return json.load(f)

    def _load_score_head(self, model_name_or_path: str):
        head_path = os.path.join(model_name_or_path, self.RERANKER_HEAD)
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"Backbone reranker checkpoint is missing {self.RERANKER_HEAD}: {model_name_or_path}")
        self.score.load_state_dict(torch.load(head_path, map_location="cpu"))
