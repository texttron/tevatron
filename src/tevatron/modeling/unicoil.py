import torch
from torch import Tensor, nn
import logging

from .biencoder import BiEncoderPooler, BiEncoderModel, BiEncoderModelForInference
from ..arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


class UniCoilPooler(BiEncoderPooler):
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


class UniCoilModel(BiEncoderModel):

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.pooler(p=p_hidden)
        return p_hidden, self._weights_to_vec(psg['input_ids'], p_reps)

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.pooler(q=q_hidden)
        return q_hidden, self._weights_to_vec(qry['input_ids'], q_reps)

    def compute_similarity(self, q_reps, p_reps, query, passage):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def _weights_to_vec(self, input_ids, tok_weights):
        input_shape = input_ids.size()
        tok_weights = torch.relu(tok_weights)
        tok_emb = torch.zeros(input_shape[0], input_shape[1], self.lm_p.config.vocab_size, dtype=tok_weights.dtype,
                              device=input_ids.device)
        disabled_token_ids = [0, 101, 102, 103]  # hard code for bert for now, can pass in a tokenizer in the future
        tok_emb[:, :, disabled_token_ids] *= 0
        tok_emb = torch.scatter(tok_emb, dim=-1, index=input_ids.unsqueeze(-1), src=tok_weights)
        tok_emb = torch.max(tok_emb, dim=1).values
        return tok_emb

    @staticmethod
    def build_pooler(model_args):
        pooler = UniCoilPooler(
            model_args.projection_in_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        model_args.add_pooler = True
        return super().build(model_args, train_args, **hf_kwargs)


class UniCoilForInference(BiEncoderModelForInference, UniCoilModel):
    POOLER_CLS = UniCoilPooler

    @torch.no_grad()
    def encode_passage(self, psg):
        return UniCoilModel.encode_passage(self, psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return UniCoilModel.encode_query(self, qry)
