# coding=utf-8
"""
    PyTorch Splade [1] model.

    References
    ----------
    .. [1] Formal, Thibault, et al. "SPLADE v2: 
       Sparse lexical and expansion model for information retrieval."
       arXiv preprint arXiv:2109.10086 (2021).
"""


import logging

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.activations import get_activation, ACT2FN
from transformers.modeling_outputs import ModelOutput




logger = logging.getLogger(__name__)

ACT2FN.upate(
    {'max': torch.max,
     'sum': torch.sum,
     'mean': torch.sum
    }
)


def dist_gather_tensor(
    tensor: torch.Tensor,
    ):
    """
    A workaround to make tensors collected by dist.all_gather have gradients
    """
    if tensor is None:
        return None
    tensor = tensor.contiguous()
    all_tensors = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_tensors, tensor)

    # gathered tensors have no gradient
    # so we overwrite the gathered tensor which do have gradients.
    all_tensors[torch.distributed.get_rank()] = tensor
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors


@dataclass
class SpladeOutputWithPooling(ModelOutput):
    representation: torch.FloatTensor = None


@dataclass
class SpladeForRetrievalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: torch.FloatTensor = None
    qry_reps: torch.FloatTensor = None
    key_reps: torch.FloatTensor = None


class SpladePoolingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        if hasattr(self.config, 'Splade_act_fn'):
            self.act_fn = get_activation(self.config.Splade_act_fn)
        else:
            self.act_fn = get_activation('relu')
            self.config.Splade_act_fn = 'relu'

        if hasattr(self.config, 'Splade_pool_fn'):
            self.pool_fn = get_activation(self.config.Splade_pool_fn)
        else:
            self.pool_fn = get_activation('max')
            self.config.Splade_pool_fn = 'max'
        
    def forward(
        self,
        attention_mask=None,
        encoder_hidden_states=None,
    ):
        if self.config.Splade_pool_fn == 'max':
            hidden_states = self.pool_fn(
                torch.log(1 + self.act_fn(encoder_hidden_states) * attention_mask.unsqueeze(-1)),
                dim=1
            ).values
        else:
            hidden_states = self.pool_fn(
                torch.log(1 + self.act_fn(encoder_hidden_states) * attention_mask.unsqueeze(-1)),
                dim=1
            )
            if self.config.Splade_pool_fn == 'mean':
                seq_len = attention_mask.sum(dim=1)
                seq_len = torch.clamp(seq_len, min=1e-9)
                hidden_states = hidden_states / seq_len
        return hidden_states


class SpladeModel(PreTrainedModel):
    def __init__(self, pretrained_model_name_or_path):
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        super().__init__(self.config)
        self.encoder = AutoModelForMaskedLM.from_config(self.config)
        self.pooler = SpladePoolingHead(self.config)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = self.pooler(
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_output
        )

        return SpladeOutputWithPooling(
            representation=pooled_output
        )


class SpladeModelForRetrieval(PreTrainedModel):

    def __init__(
        self,
        config: PretrainedConfig     = None,
        qry_encoder: PreTrainedModel = None,
        key_encoder: PreTrainedModel = None,
    ):
        super().__init__(config)

        if qry_encoder is None:
            qry_encoder = SpladeModel(config.qry_encoder)
        
        if key_encoder is None:
            key_encoder = SpladeModel(config.key_encoder)
        
        self.qry_encoder = qry_encoder
        self.key_encoder = key_encoder

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.qry_encoder.config = self.config.qry_encoder
        self.key_encoder.config = self.config.key_encoder

        self.maybe_tie_weights()

        self.criteria = nn.CrossEntropyLoss()

        # hacky way to incorporate in-batch negatives
        if self.config.negatives_x_device:
            if not torch.distributed.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.world_size = torch.distributed.get_world_size()
    
    def maybe_tie_weights(self):
        if self.config.tie_weights:
            key_encoder_base_model_prefix = self.key_encoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.qry_encoder,
                self.key_encoder._modules[key_encoder_base_model_prefix],
                self.key_encoder.base_model_prefix
            )
    
    def forward(
        self,
        qry: Dict[str, torch.Tensor] = None,
        key: Dict[str, torch.Tensor] = None,
    ):
        # compute representations
        qry_reps = None
        key_reps = None

        if qry is not None:
            qry_encoder_output = self.qry_encoder(
                input_ids=qry.input_ids,
                attention_mask=qry.attention_mask
            )
            qry_reps = qry_encoder_output.representation

        if key is not None:
            key_encoder_output = self.key_encoder(
                input_ids=key.input_ids,
                attention_mask=key.attention_mask
            )
            key_reps = key_encoder_output.representation

        # compute additional outputs
        loss = None
        scores = None

        if self.training:
            if self.config.negatives_x_device:
                qry_reps = dist_gather_tensor(qry_reps)
                key_reps = dist_gather_tensor(key_reps)

            scores = torch.einsum('ik,jk->ij', qry_reps, key_reps)
            labels = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long,
            )
            labels = labels * self.config.train_n_passages
            loss = self.criteria(scores, labels)

            if self.config.negatives_x_device:
                loss = loss * self.world_size # hacky way to counter average weight reduction
        else:
            if qry and key:
                scores = torch.einsum('ik,jk->ij', qry_reps, key_reps)
        
        return SpladeForRetrievalOutput(
            loss=loss,
            scores=scores,
            qry_reps=qry_reps,
            key_reps=key_reps
        )

            





