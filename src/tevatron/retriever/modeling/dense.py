import torch
import torch.nn.functional as F
import logging
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from .encoder import EncoderModel

logger = logging.getLogger(__name__)
EOS_TOKEN_ID = 151643

class DenseModel(EncoderModel):

    def __init__(self, encoder, pooling='cls', normalize=False, temperature=1.0):
        super().__init__(encoder, pooling, normalize, temperature)
        self.passage_chunk_size = 0
        self.eos_positions = None

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg, eos_positions=None):
        logger.info(f"[DenseModel.encode_passage] passage_chunk_size: {self.passage_chunk_size}")
        logger.info(f"[DenseModel.encode_passage] eos_positions: "
                   f"{'None' if eos_positions is None else f'list[{len(eos_positions)}] with lens {[len(ep) for ep in eos_positions]}'}")
        
        hidden_states = self.encoder(**psg, return_dict=True).last_hidden_state
        logger.info(f"[DenseModel.encode_passage] hidden_states shape: {hidden_states.shape}")
        
        if self.passage_chunk_size > 0 and eos_positions:
            # Verify EOS tokens are at the right positions
            for i, ep in enumerate(eos_positions):
                for eos_pos in ep:
                    assert psg['input_ids'][i][eos_pos] == EOS_TOKEN_ID, \
                        f"Expected EOS token {EOS_TOKEN_ID} at position {eos_pos} in passage {i}, got {psg['input_ids'][i][eos_pos]}"

            logger.info(f"[DenseModel.encode_passage] Calling _pooling_chunked")
            result = self._pooling_chunked(hidden_states, eos_positions)
            logger.info(f"[DenseModel.encode_passage] Returning tuple: reps {result[0].shape}, mask {result[1].shape}")
            return result
        
        logger.info(f"[DenseModel.encode_passage] Calling regular _pooling (no chunking)")
        return self._pooling(hidden_states, psg['attention_mask'])

    def _pooling_chunked(self, last_hidden_state, eos_positions):
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        logger.info(f"[_pooling_chunked] Input: last_hidden_state shape {last_hidden_state.shape}")
        logger.info(f"[_pooling_chunked] eos_positions: {eos_positions}")
        
        if not eos_positions:
            # No chunks, return empty
            logger.warning(f"[_pooling_chunked] eos_positions is empty! Returning empty tensors")
            return torch.zeros(batch_size, 0, hidden_size, device=last_hidden_state.device, dtype=last_hidden_state.dtype), \
                   torch.zeros(batch_size, 0, device=last_hidden_state.device)
        
        # Find max number of chunks across all passages
        chunk_counts = [len(pos_list) for pos_list in eos_positions]
        logger.info(f"[_pooling_chunked] Chunk counts per passage: {chunk_counts}")
        max_chunks = max(chunk_counts)
        
        chunk_reps = torch.zeros(batch_size, max_chunks, hidden_size, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        chunk_mask = torch.zeros(batch_size, max_chunks, device=last_hidden_state.device, dtype=torch.float)
        
        # Extract embeddings at eos_positions (this is the pooling operation for chunked passages)
        valid_chunks_count = 0
        for i, positions in enumerate(eos_positions):
            for j, pos in enumerate(positions):
                if 0 <= pos < seq_len:
                    # i is the batch index, j is the chunk index, pos is the eos position
                    chunk_reps[i, j] = last_hidden_state[i, pos]
                    # chunk_mask is 1.0 for valid chunks, 0.0 for padding chunks
                    chunk_mask[i, j] = 1.0
                    valid_chunks_count += 1
                else:
                    logger.warning(f"Position {pos} out of bounds for sequence length {seq_len} in batch {i}, chunk {j}")
        
        if self.normalize:
            chunk_reps = F.normalize(chunk_reps, p=2, dim=-1)
        
        logger.info(f"[_pooling_chunked] Created chunk_reps {chunk_reps.shape}, chunk_mask {chunk_mask.shape}")
        logger.info(f"[_pooling_chunked] Valid chunks: {valid_chunks_count}/{batch_size * max_chunks}, mask sum: {chunk_mask.sum().item()}")
        
        return chunk_reps, chunk_mask
        

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps


class MultiModalDenseModel(DenseModel):
    TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration

    def __init__(self, encoder, pooling='eos', normalize=True, temperature=0.02):
        super().__init__(encoder, pooling, normalize, temperature)
        # freeze visual encoder
        self.encoder = encoder
        for param in self.encoder.visual.parameters():
            param.requires_grad = False
        # freeze audio_tower
        for param in self.encoder.audio_tower.parameters():
            param.requires_grad = False
        self.config.hidden_size = 3584

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def encode_query(self, qry):
        cache_position = torch.arange(0, qry['input_ids'].shape[1], device=qry['input_ids'].device)
        qry = self.encoder.prepare_inputs_for_generation(**qry, use_cache=True, cache_position=cache_position)
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        # query_hidden_states = query_hidden_states.hidden_states[1][-1]
        query_hidden_states = query_hidden_states.hidden_states[-1]

        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)
