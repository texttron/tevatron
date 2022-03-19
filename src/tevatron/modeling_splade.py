# coding=utf-8
"""
    PyTorch Splade [1] model.

    References
    ----------
    .. [1] Formal, Thibault, et al. "SPLADE v2: 
       Sparse lexical and expansion model for information retrieval."
       arXiv preprint arXiv:2109.10086 (2021).
"""


import os
import copy
import logging

import torch
from torch import nn
import torch.distributed as dist

from transformers import PreTrainedModel
from transformers import AutoModelForMaskedLM

from tevatron.modeling import DenseModel
from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments




logger = logging.getLogger(__name__)


class SpladePoolingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = torch.relu
        
    def forward(
        self,
        attention_mask=None,
        encoder_hidden_states=None,
    ):
        hidden_states = self.pool_fn(
            torch.log(1 + self.act_fn(encoder_hidden_states) * attention_mask.unsqueeze(-1)),
            dim=1
        ).values
        
        return hidden_states


class SpladeModel(DenseModel):
    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        pooler: nn.Module = None,
        model_args: ModelArguments = None,
        data_args: DataArguments = None,
        train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True)
        p_token_embs = psg_out.logits
        p_reps = self.pooler(p_token_embs)

        return None, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None

        qry_out = self.lm_q(**qry, return_dict=True)
        q_token_embs = qry_out.logits
        q_reps = self.pooler(q_token_embs)

        return None, q_reps
    
    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = AutoModelForMaskedLM.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = AutoModelForMaskedLM.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        pooler = SpladePoolingHead()

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model
