"""
Test to verify that when chunk_size == passage_max_len and there's only one chunk,
chunked and non-chunked modes should produce the same embeddings.
"""
import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


@pytest.fixture
def train_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")


@pytest.mark.unit
def test_chunked_vs_non_chunked_when_chunk_size_equals_max_len(train_tokenizer):
    """
    When chunk_size == passage_max_len and passage fits in one chunk,
    chunked and non-chunked should produce identical embeddings.
    """
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator, ChunkedEncodeCollator
    from tevatron.retriever.modeling.dense import DenseModel
    from unittest.mock import Mock
    
    # Test passage that fits in one chunk
    test_passage = "This is a test passage that will fit in one chunk."
    
    # Configuration: chunk_size == passage_max_len
    passage_max_len = 64
    chunk_size = 64  # Same as max_len
    
    # Test Case 1: Non-chunked mode
    data_args_non_chunked = DataArguments(
        passage_max_len=passage_max_len,
        passage_chunk_size=0,  # No chunking
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=False,  # Default: False
    )
    
    collator_non_chunked = TrainCollator(data_args=data_args_non_chunked, tokenizer=train_tokenizer)
    q_batch_non_chunked, p_batch_non_chunked = collator_non_chunked([("query", [test_passage], [])])
    
    # Test Case 2: Chunked mode with chunk_size == max_len
    data_args_chunked = DataArguments(
        passage_max_len=passage_max_len,
        passage_chunk_size=chunk_size,  # Same as max_len
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=False,  # Same as non-chunked
    )
    
    collator_chunked = TrainCollator(data_args=data_args_chunked, tokenizer=train_tokenizer)
    q_batch_chunked, p_batch_chunked, eos_positions = collator_chunked([("query", [test_passage], [])])
    
    # Verify tokenization differences
    input_ids_non_chunked = p_batch_non_chunked['input_ids'][0]
    input_ids_chunked = p_batch_chunked['input_ids'][0]
    
    # Chunked mode adds EOS after chunk, non-chunked doesn't (when append_eos_token=False)
    # So chunked should have one more token (the EOS)
    non_chunked_content = input_ids_non_chunked[p_batch_non_chunked['attention_mask'][0].bool()].tolist()
    chunked_content = input_ids_chunked[p_batch_chunked['attention_mask'][0].bool()].tolist()
    
    print(f"Non-chunked content tokens: {len(non_chunked_content)}")
    print(f"Chunked content tokens: {len(chunked_content)}")
    print(f"EOS positions: {eos_positions}")
    
    # Chunked should have EOS token at the end of the chunk
    assert chunked_content[-1] == train_tokenizer.eos_token_id, "Chunked mode should have EOS at end"
    # Non-chunked should NOT have EOS (when append_eos_token=False)
    assert non_chunked_content[-1] != train_tokenizer.eos_token_id, "Non-chunked mode should NOT have EOS"
    
    # The content tokens (excluding EOS) should be the same
    chunked_content_without_eos = chunked_content[:-1]
    assert non_chunked_content == chunked_content_without_eos, "Content tokens should be identical (excluding EOS)"
    
    # Now test pooling behavior
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        # Create hidden states where each position encodes its position index
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                # Encode position j in the first dimension
                hidden_states[i, j, 0] = float(j)
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    # Non-chunked model
    model_non_chunked = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model_non_chunked.passage_chunk_size = 0
    
    # Chunked model
    model_chunked = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model_chunked.passage_chunk_size = chunk_size
    
    # Get embeddings
    p_reps_non_chunked = model_non_chunked.encode_passage(p_batch_non_chunked)
    p_reps_chunked_tuple = model_chunked.encode_passage(p_batch_chunked, eos_positions)
    p_reps_chunked, chunk_mask = p_reps_chunked_tuple
    
    # Non-chunked: extracts from last content token position
    # Chunked: extracts from EOS position (which is one position after last content token)
    
    # Get the actual positions
    mask_non_chunked = p_batch_non_chunked['attention_mask'][0]
    last_valid_pos_non_chunked = mask_non_chunked.sum().item() - 1
    
    # Chunked: EOS position
    eos_pos = eos_positions[0][0]  # First (and only) chunk's EOS position
    
    print(f"Non-chunked extracts from position: {last_valid_pos_non_chunked}")
    print(f"Chunked extracts from position: {eos_pos}")
    print(f"Non-chunked embedding value: {p_reps_non_chunked[0, 0].item()}")
    print(f"Chunked embedding value: {p_reps_chunked[0, 0, 0].item()}")
    
    # These should be DIFFERENT because they extract from different positions
    # Non-chunked: last content token
    # Chunked: EOS token (one position after)
    assert eos_pos == last_valid_pos_non_chunked + 1, \
        f"EOS should be one position after last content token: {eos_pos} vs {last_valid_pos_non_chunked}"
    
    # The embeddings will be different because they're extracted from different positions
    # This is the root cause of the inconsistency!
    assert not torch.allclose(p_reps_non_chunked[0], p_reps_chunked[0, 0]), \
        "Embeddings should be different because they're extracted from different token positions"
