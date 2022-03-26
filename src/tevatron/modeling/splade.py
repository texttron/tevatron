import torch
import logging

from .biencoder import BiEncoderModel, BiEncoderModelForInference

logger = logging.getLogger(__name__)


class SpladeModel(BiEncoderModel):
    def __init__(self, flops_q, flops_p, **kwargs):
        super().__init__(**kwargs)

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg_out = self.lm_p(**psg, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(psg_out)) * psg['attention_mask'].unsqueeze(-1), dim=1)
        return psg_out, aggregated_psg_out

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True).logits
        aggregated_psg_out, _ = torch.max(torch.log(1 + torch.relu(qry_out)) * qry['attention_mask'].unsqueeze(-1), dim=1)
        return qry_out, aggregated_psg_out

    def compute_similarity(self, q_reps, p_reps, query, passage):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target, q_reps, p_reps):
        return self.cross_entropy(scores, target)

class SpladeForInference(BiEncoderModelForInference, SpladeModel):

    @torch.no_grad()
    def encode_passage(self, psg):
        return SpladeModel.encode_passage(self, psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return SpladeModel.encode_query(self, qry)
