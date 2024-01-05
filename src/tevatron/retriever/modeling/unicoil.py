from typing import Optional
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel

from .encoder import EncoderModel

import logging
logger = logging.getLogger(__name__)
        

class UniCoilEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.tok_proj = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return self._weights_to_vec(input_ids, tok_weights)
    
    def _weights_to_vec(self, input_ids, tok_weights):
        input_shape = input_ids.size()
        tok_weights = torch.relu(tok_weights)
        tok_emb = torch.zeros(input_shape[0], input_shape[1], self.lm_p.config.vocab_size, dtype=tok_weights.dtype,
                              device=input_ids.device)
        tok_emb = torch.scatter(tok_emb, dim=-1, index=input_ids.unsqueeze(-1), src=tok_weights)
        disabled_token_ids = [0, 101, 102, 103]  # hard code for bert for now, can pass in a tokenizer in the future
        tok_emb = torch.max(tok_emb, dim=1).values
        tok_emb[:, disabled_token_ids] *= 0
        return tok_emb


class UniCoilModel(EncoderModel):
    TRANSFORMER_CLS = UniCoilEncoder

    def encode_query(self, qry):
        return self.encoder(**qry)

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)