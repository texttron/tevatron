import torch
import logging
from transformers import AutoModelForMaskedLM
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class SpladeModel(EncoderModel):
    TRANSFORMER_CLS = AutoModelForMaskedLM

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(psg_out)) * psg['attention_mask'].unsqueeze(-1), dim=1)
        return aggregated_psg_out

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1), dim=1)
        return aggregated_psg_out
