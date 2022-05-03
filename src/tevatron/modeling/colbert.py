import torch
import torch.nn as nn
from torch import Tensor
import logging
from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)


class ColbertPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, tied=True):
        super(ColbertPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class ColbertModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.pooler(p=p_hidden)
        p_reps *= psg['attention_mask'][:, :, None].float()
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.pooler(q=q_hidden)
        q_reps *= qry['attention_mask'][:, :, None].float()
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1)
        return scores

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ColbertPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = ColbertPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
