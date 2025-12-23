"""
Test to verify that when chunk_size == passage_max_len and there's only one chunk,
chunked and non-chunked modes extract embeddings from different positions.
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
    """When chunk_size == passage_max_len, chunked mode adds EOS and extracts from EOS position."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator
    from tevatron.retriever.modeling.dense import DenseModel
    from unittest.mock import Mock
    
    test_passage = "This is a test passage that will fit in one chunk."
    passage_max_len = chunk_size = 64
    
    # Non-chunked mode
    data_args_non_chunked = DataArguments(
        passage_max_len=passage_max_len,
        passage_chunk_size=0,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator_non_chunked = TrainCollator(data_args=data_args_non_chunked, tokenizer=train_tokenizer)
    _, p_batch_non_chunked = collator_non_chunked([("query", [test_passage], [])])
    
    # Chunked mode
    data_args_chunked = DataArguments(
        passage_max_len=passage_max_len,
        passage_chunk_size=chunk_size,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator_chunked = TrainCollator(data_args=data_args_chunked, tokenizer=train_tokenizer)
    _, p_batch_chunked, eos_positions = collator_chunked([("query", [test_passage], [])])
    
    # Verify tokenization: chunked adds EOS, non-chunked doesn't
    non_chunked_content = p_batch_non_chunked['input_ids'][0][p_batch_non_chunked['attention_mask'][0].bool()].tolist()
    chunked_content = p_batch_chunked['input_ids'][0][p_batch_chunked['attention_mask'][0].bool()].tolist()
    
    assert chunked_content[-1] == train_tokenizer.eos_token_id
    assert non_chunked_content[-1] != train_tokenizer.eos_token_id
    assert non_chunked_content == chunked_content[:-1]
    
    # Test pooling: different positions yield different embeddings
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                hidden_states[i, j, 0] = float(j)
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    model_non_chunked = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model_non_chunked.passage_chunk_size = 0
    model_chunked = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model_chunked.passage_chunk_size = chunk_size
    
    p_reps_non_chunked = model_non_chunked.encode_passage(p_batch_non_chunked)
    p_reps_chunked, _ = model_chunked.encode_passage(p_batch_chunked, eos_positions)
    
    last_valid_pos = p_batch_non_chunked['attention_mask'][0].sum().item() - 1
    eos_pos = eos_positions[0][0]
    
    assert eos_pos == last_valid_pos + 1
    assert not torch.allclose(p_reps_non_chunked[0], p_reps_chunked[0, 0])
