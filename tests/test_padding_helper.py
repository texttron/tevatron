"""
Unit tests for _pad_and_adjust_eos_positions helper function.
"""
import sys
from pathlib import Path
import pytest
import torch


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


@pytest.fixture(scope="session")
def train_tokenizer():
    """Use the Qwen 0.6B tokenizer."""
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_right_padding(train_tokenizer):
    """Test padding with right padding (no EOS position adjustment needed)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, 2, 3, eos_id],  # Passage 0: 4 tokens, EOS at position 3
        [4, 5, eos_id],      # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[3], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, 3, eos_id],      # Passage 0: padded to 4 (no padding needed)
        [4, 5, eos_id, pad_id], # Passage 1: padded to 4 (1 padding token)
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1],   # Passage 0: all tokens valid
        [1, 1, 1, 0],   # Passage 1: last token is padding
    ])
    expected_eos_positions = [[3], [2]]  # EOS positions unchanged for right padding
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_left_padding(train_tokenizer):
    """Test padding with left padding (EOS positions should be shifted)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, 2, 3, eos_id],  # Passage 0: 4 tokens, EOS at position 3
        [4, 5, eos_id],      # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[3], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='left',
        pad_to_multiple_of=4,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, 3, eos_id],      # Passage 0: padded to 4 (no padding needed)
        [pad_id, 4, 5, eos_id], # Passage 1: padded to 4 (1 padding token on left)
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1],   # Passage 0: all tokens valid
        [0, 1, 1, 1],   # Passage 1: first token is padding
    ])
    # Passage 0: original length 4, padded length 4, padding_length=0, EOS stays at 3
    # Passage 1: original length 3, padded length 4, padding_length=1, EOS shifts from 2 to 3
    expected_eos_positions = [[3], [3]]
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_multiple_eos(train_tokenizer):
    """Test padding with multiple EOS positions per passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, 2, eos_id, 3, 4, eos_id],  # Passage 0: 6 tokens, EOS at positions 2, 5
        [5, eos_id],                    # Passage 1: 2 tokens, EOS at position 1
    ]
    all_eos_positions = [[2, 5], [1]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='left',
        pad_to_multiple_of=8,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [pad_id, pad_id, 1, 2, eos_id, 3, 4, eos_id],  # Passage 0: padded to 8 (2 padding tokens on left)
        [pad_id, pad_id, pad_id, pad_id, pad_id, pad_id, 5, eos_id],  # Passage 1: padded to 8 (6 padding tokens on left)
    ])
    expected_attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 1],   # Passage 0: first 2 tokens are padding
        [0, 0, 0, 0, 0, 0, 1, 1],   # Passage 1: first 6 tokens are padding
    ])
    # Passage 0: original length 6, padded length 8, padding_length=2, EOS shift from [2,5] to [4,7]
    # Passage 1: original length 2, padded length 8, padding_length=6, EOS shift from 1 to 7
    expected_eos_positions = [[4, 7], [7]]
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_no_padding_needed(train_tokenizer):
    """Test when sequences are already the same length."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, 2, eos_id],
        [3, 4, eos_id],
    ]
    all_eos_positions = [[2], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, eos_id, pad_id],  # Padded to 4 (1 padding token)
        [3, 4, eos_id, pad_id],   # Padded to 4 (1 padding token)
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1, 0],   # Last token is padding
        [1, 1, 1, 0],   # Last token is padding
    ])
    expected_eos_positions = [[2], [2]]  # EOS positions unchanged for right padding
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_empty_input(train_tokenizer):
    """Test with empty input."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = []
    all_eos_positions = []
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # Hardcoded golden output for empty input
    expected_eos_positions = []
    
    assert adjusted_eos_positions == expected_eos_positions
    # When input is empty, tokenizer.pad may return list or tensor depending on implementation
    if isinstance(padded_dict['input_ids'], torch.Tensor):
        assert padded_dict['input_ids'].shape[0] == 0
        assert padded_dict['attention_mask'].shape[0] == 0
    else:
        assert len(padded_dict['input_ids']) == 0
        assert len(padded_dict['attention_mask']) == 0


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_single_passage(train_tokenizer):
    """Test with single passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    
    all_input_ids = [[1, 2, 3, eos_id]]
    all_eos_positions = [[3]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, 3, eos_id],  # Already length 4, no padding needed
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1, 1],   # All tokens valid
    ])
    expected_eos_positions = [[3]]
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_pad_to_multiple_of_one(train_tokenizer):
    """Test with pad_to_multiple_of=1 (no rounding)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, 2, eos_id],
        [3, eos_id],
    ]
    all_eos_positions = [[2], [1]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=1,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, eos_id],        # Padded to max_len=3 (no rounding needed with pad_to_multiple_of=1)
        [3, eos_id, pad_id],    # Padded to max_len=3 (1 padding token)
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1],   # All tokens valid
        [1, 1, 0],   # Last token is padding
    ])
    expected_eos_positions = [[2], [1]]  # EOS positions unchanged for right padding
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_left_padding_multiple_chunks(train_tokenizer):
    """Test left padding with multiple chunks per passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    
    all_input_ids = [
        [1, eos_id, 2, 3, eos_id],  # Passage 0: 5 tokens, EOS at positions 1, 4
        [4, 5, eos_id],              # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[1, 4], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='left',
        pad_to_multiple_of=8,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [pad_id, pad_id, pad_id, 1, eos_id, 2, 3, eos_id],  # Passage 0: padded to 8 (3 padding tokens on left)
        [pad_id, pad_id, pad_id, pad_id, pad_id, 4, 5, eos_id],  # Passage 1: padded to 8 (5 padding tokens on left)
    ])
    expected_attention_mask = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1],   # Passage 0: first 3 tokens are padding
        [0, 0, 0, 0, 0, 1, 1, 1],   # Passage 1: first 5 tokens are padding
    ])
    # Passage 0: original length 5, padded length 8, padding_length=3, EOS shift from [1,4] to [4,7]
    # Passage 1: original length 3, padded length 8, padding_length=5, EOS shift from 2 to 7
    expected_eos_positions = [[4, 7], [7]]
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_tokenizer_padding_side_set(train_tokenizer):
    """Test that tokenizer.padding_side is set correctly."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    eos_id = train_tokenizer.eos_token_id
    
    all_input_ids = [[1, 2, eos_id]]
    all_eos_positions = [[2]]
    
    # Test right padding
    train_tokenizer.padding_side = 'right'
    _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    assert train_tokenizer.padding_side == 'right'
    
    # Test left padding
    train_tokenizer.padding_side = 'left'
    _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='left',
        pad_to_multiple_of=4,
    )
    assert train_tokenizer.padding_side == 'left'

