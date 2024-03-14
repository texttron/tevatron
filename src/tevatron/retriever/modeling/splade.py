import torch
import logging
from transformers import AutoModelForMaskedLM
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class SpladeModel(EncoderModel):
    TRANSFORMER_CLS = AutoModelForMaskedLM

    def encode_query(self, qry):
        qry_out = self.encoder(**qry, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1), dim=1)
        return aggregated_psg_out
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
