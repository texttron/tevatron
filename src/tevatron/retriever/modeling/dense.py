import torch
import logging
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):
    def encode_passage(self, psg):
        passage_hidden_states = self.lm_p(**psg, return_dict=True)
        passage_hidden_states = passage_hidden_states.last_hidden_state
        return self._pooling(passage_hidden_states, psg['attention_mask'])

    def encode_query(self, qry):
        query_hidden_states = self.lm_q(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def _pooling(self, last_hidden_state, attention_mask):
        if self.model_args.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.model_args.pooling in ['mean', 'avg', 'average']:
            reps = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.model_args.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.model_args.pooling}')
        if self.model_args.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
