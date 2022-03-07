import torch
import torch.nn as nn
from torch import Tensor
import logging
from .biencoder import BiEncoderPooler, BiEncoderModel, BiEncoderModelForInference

logger = logging.getLogger(__name__)


class LinearPooler(BiEncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(LinearPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError


class DenseModel(BiEncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    def compute_similarity(self, q_reps, p_reps, query, passage):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler


class DenseModelForInference(BiEncoderModelForInference, DenseModel):
    POOLER_CLS = LinearPooler

    @torch.no_grad()
    def encode_passage(self, psg):
        return DenseModel.encode_passage(self, psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return DenseModel.encode_query(self, qry)
