import torch
import torch.nn.functional as F
import logging
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from .encoder import EncoderModel

logger = logging.getLogger(__name__)


class DenseModel(EncoderModel):

    def __init__(self, encoder, pooling='cls', normalize=False, temperature=1.0):
        super().__init__(encoder, pooling, normalize, temperature)
        self.passage_chunk_size = 0
        self.eos_positions = None
        self.eos_token_id = None  # Will be set by driver/trainer

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        return self._pooling(query_hidden_states, qry['attention_mask'])

    def encode_passage(self, psg, eos_positions=None):
        hidden_states = self.encoder(**psg, return_dict=True).last_hidden_state

        if self.passage_chunk_size > 0 and eos_positions:
            # Verify EOS tokens are at the right positions (optional, for debugging)
            if self.eos_token_id is not None:
                for i, ep in enumerate(eos_positions):
                    for eos_pos in ep:
                        actual_token = psg['input_ids'][i][eos_pos].item()
                        if actual_token != self.eos_token_id:
                            logger.warning(
                                f"Expected EOS token {self.eos_token_id} at position {eos_pos} "
                                f"in passage {i}, got {actual_token}"
                            )

            return self._pooling_chunked(hidden_states, eos_positions)

        return self._pooling(hidden_states, psg['attention_mask'])

    def _pooling_chunked(self, last_hidden_state, eos_positions):
        batch_size, seq_len, hidden_size = last_hidden_state.shape

        # Handle empty or all-empty eos_positions
        if not eos_positions or all(len(ep) == 0 for ep in eos_positions):
            max_chunks = 0
        else:
            # Find max number of chunks across all passages in this batch
            chunk_counts = [len(pos_list) for pos_list in eos_positions]
            max_chunks = max(chunk_counts)

        # CRITICAL: Synchronize max_chunks across all DDP ranks
        # This ensures all ranks produce tensors with the same shape for gathering
        if torch.distributed.is_initialized():
            local_max_chunks = max_chunks
            local_batch_size = batch_size

            # Sync max_chunks across ranks
            max_chunks_tensor = torch.tensor([max_chunks], dtype=torch.long, device=last_hidden_state.device)
            torch.distributed.all_reduce(max_chunks_tensor, op=torch.distributed.ReduceOp.MAX)
            max_chunks = int(max_chunks_tensor.item())

            # Also sync batch_size to verify consistency (for debugging)
            batch_size_tensor = torch.tensor([batch_size], dtype=torch.long, device=last_hidden_state.device)
            batch_sizes_list = [torch.zeros_like(batch_size_tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(batch_sizes_list, batch_size_tensor)
            batch_sizes = [int(bs.item()) for bs in batch_sizes_list]

            # Log detailed info for debugging DDP issues
            rank = torch.distributed.get_rank()
            if local_max_chunks != max_chunks:
                logger.info(
                    f"[Rank {rank}] max_chunks synced: local={local_max_chunks} → global={max_chunks}, "
                    f"batch_size={local_batch_size}, all_batch_sizes={batch_sizes}"
                )

            # Warn if batch sizes differ (this could cause gathering issues)
            if len(set(batch_sizes)) > 1:
                logger.warning(
                    f"[Rank {rank}] Batch sizes differ across ranks: {batch_sizes}. "
                    "This may cause DDP gathering errors!"
                )

        # If no chunks at all, return empty tensors
        if max_chunks == 0:
            return torch.zeros(batch_size, 0, hidden_size, device=last_hidden_state.device, dtype=last_hidden_state.dtype), \
                   torch.zeros(batch_size, 0, device=last_hidden_state.device, dtype=torch.float)

        chunk_reps = torch.zeros(batch_size, max_chunks, hidden_size, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        chunk_mask = torch.zeros(batch_size, max_chunks, device=last_hidden_state.device, dtype=torch.float)

        # Extract embeddings at eos_positions (this is the pooling operation for chunked passages)
        for i, positions in enumerate(eos_positions):
            for j, pos in enumerate(positions):
                if 0 <= pos < seq_len:
                    # i is the batch index, j is the chunk index, pos is the eos position
                    chunk_reps[i, j] = last_hidden_state[i, pos]
                    # chunk_mask is 1.0 for valid chunks, 0.0 for padding chunks
                    chunk_mask[i, j] = 1.0

        if self.normalize:
            chunk_reps = F.normalize(chunk_reps, p=2, dim=-1)

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
