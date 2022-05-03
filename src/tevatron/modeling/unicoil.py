import torch
from torch import Tensor, nn
import logging

from .encoder import EncoderPooler, EncoderModel

logger = logging.getLogger(__name__)


class UniCoilPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, tied=True):
        super(UniCoilPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, 1)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, 1)
        self._config = {'input_dim': input_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class UniCoilModel(EncoderModel):
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.pooler(p=p_hidden)
        return self._weights_to_vec(psg['input_ids'], p_reps)

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.pooler(q=q_hidden)
        return self._weights_to_vec(qry['input_ids'], q_reps)

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

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

    @staticmethod
    def build_pooler(model_args):
        pooler = UniCoilPooler(
            model_args.projection_in_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = UniCoilPooler(**config)
        pooler.load(model_weights_file)
        return pooler
