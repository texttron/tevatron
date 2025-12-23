"""
Unit tests for _pad_and_adjust_eos_positions helper function.
"""
import sys
from pathlib import Path
import pytest
import torch
from unittest.mock import Mock, MagicMock


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 99
    
    def pad_fn(encodings, padding=True, pad_to_multiple_of=None, return_attention_mask=True, return_tensors=None):
        """Mock pad function that simulates tokenizer.pad behavior."""
        input_ids = encodings['input_ids']
        max_len = max(len(ids) for ids in input_ids) if input_ids else 0
        
        # Round up to multiple of pad_to_multiple_of
        if pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        
        padded_ids = []
        attention_masks = []
        
        for ids in input_ids:
            if tokenizer.padding_side == 'right':
                pad_length = max_len - len(ids)
                padded = ids + [tokenizer.pad_token_id] * pad_length
                mask = [1] * len(ids) + [0] * pad_length
            else:  # left padding
                pad_length = max_len - len(ids)
                padded = [tokenizer.pad_token_id] * pad_length + ids
                mask = [0] * pad_length + [1] * len(ids)
            
            padded_ids.append(padded)
            attention_masks.append(mask)
        
        result = {'input_ids': padded_ids, 'attention_mask': attention_masks}
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result
    
    tokenizer.pad = MagicMock(side_effect=pad_fn)
    tokenizer.padding_side = 'right'
    return tokenizer


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_right_padding(mock_tokenizer):
    """Test padding with right padding (no EOS position adjustment needed)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 2, 3, 99],  # Passage 0: 4 tokens, EOS at position 3
        [4, 5, 99],     # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[3], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # With pad_to_multiple_of=4, max_len=4 -> padded to 4
    assert padded_dict['input_ids'].shape == (2, 4)
    assert padded_dict['attention_mask'].shape == (2, 4)
    
    # EOS positions should not change for right padding
    assert adjusted_eos_positions == [[3], [2]]
    
    # Verify EOS tokens are at correct positions
    assert padded_dict['input_ids'][0][3].item() == 99
    assert padded_dict['input_ids'][1][2].item() == 99


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_left_padding(mock_tokenizer):
    """Test padding with left padding (EOS positions should be shifted)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 2, 3, 99],  # Passage 0: 4 tokens, EOS at position 3
        [4, 5, 99],     # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[3], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='left',
        pad_to_multiple_of=4,
    )
    
    # With pad_to_multiple_of=4, max_len=4 -> padded to 4
    assert padded_dict['input_ids'].shape == (2, 4)
    
    # Passage 0: original length 4, padded length 4, padding_length=0, EOS stays at 3
    # Passage 1: original length 3, padded length 4, padding_length=1, EOS shifts from 2 to 3
    assert adjusted_eos_positions == [[3], [3]]
    
    # Verify EOS tokens are at correct positions after padding
    assert padded_dict['input_ids'][0][3].item() == 99
    assert padded_dict['input_ids'][1][3].item() == 99


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_multiple_eos(mock_tokenizer):
    """Test padding with multiple EOS positions per passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 2, 99, 3, 4, 99],  # Passage 0: 6 tokens, EOS at positions 2, 5
        [5, 99],                # Passage 1: 2 tokens, EOS at position 1
    ]
    all_eos_positions = [[2, 5], [1]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='left',
        pad_to_multiple_of=8,
    )
    
    # With pad_to_multiple_of=8, max_len=6 -> padded to 8
    assert padded_dict['input_ids'].shape == (2, 8)
    
    # Passage 0: original length 6, padded length 8, padding_length=2, EOS shift from [2,5] to [4,7]
    # Passage 1: original length 2, padded length 8, padding_length=6, EOS shift from 1 to 7
    assert adjusted_eos_positions == [[4, 7], [7]]
    
    # Verify EOS tokens are at correct positions
    assert padded_dict['input_ids'][0][4].item() == 99
    assert padded_dict['input_ids'][0][7].item() == 99
    assert padded_dict['input_ids'][1][7].item() == 99


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_no_padding_needed(mock_tokenizer):
    """Test when sequences are already the same length."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 2, 99],
        [3, 4, 99],
    ]
    all_eos_positions = [[2], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    # With pad_to_multiple_of=4, max_len=3 -> padded to 4
    assert padded_dict['input_ids'].shape == (2, 4)
    
    # EOS positions unchanged for right padding
    assert adjusted_eos_positions == [[2], [2]]


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_empty_input(mock_tokenizer):
    """Test with empty input."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = []
    all_eos_positions = []
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    assert len(adjusted_eos_positions) == 0
    assert padded_dict['input_ids'].shape[0] == 0


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_single_passage(mock_tokenizer):
    """Test with single passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [[1, 2, 3, 99]]
    all_eos_positions = [[3]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    
    assert padded_dict['input_ids'].shape == (1, 4)
    assert adjusted_eos_positions == [[3]]


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_pad_to_multiple_of_one(mock_tokenizer):
    """Test with pad_to_multiple_of=1 (no rounding)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 2, 99],
        [3, 99],
    ]
    all_eos_positions = [[2], [1]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=1,
    )
    
    # Should pad to max_len=3 (no rounding needed)
    assert padded_dict['input_ids'].shape == (2, 3)
    assert adjusted_eos_positions == [[2], [1]]


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_left_padding_multiple_chunks(mock_tokenizer):
    """Test left padding with multiple chunks per passage."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [
        [1, 99, 2, 3, 99],  # Passage 0: 5 tokens, EOS at positions 1, 4
        [4, 5, 99],          # Passage 1: 3 tokens, EOS at position 2
    ]
    all_eos_positions = [[1, 4], [2]]
    
    padded_dict, adjusted_eos_positions = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='left',
        pad_to_multiple_of=8,
    )
    
    # With pad_to_multiple_of=8, max_len=5 -> padded to 8
    assert padded_dict['input_ids'].shape == (2, 8)
    
    # Passage 0: original length 5, padded length 8, padding_length=3, EOS shift from [1,4] to [4,7]
    # Passage 1: original length 3, padded length 8, padding_length=5, EOS shift from 2 to 7
    assert adjusted_eos_positions == [[4, 7], [7]]


@pytest.mark.unit
def test_pad_and_adjust_eos_positions_tokenizer_padding_side_set(mock_tokenizer):
    """Test that tokenizer.padding_side is set correctly."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _pad_and_adjust_eos_positions
    
    all_input_ids = [[1, 2, 99]]
    all_eos_positions = [[2]]
    
    # Test right padding
    mock_tokenizer.padding_side = 'right'
    _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='right',
        pad_to_multiple_of=4,
    )
    assert mock_tokenizer.padding_side == 'right'
    
    # Test left padding
    mock_tokenizer.padding_side = 'left'
    _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=mock_tokenizer,
        padding_side='left',
        pad_to_multiple_of=4,
    )
    assert mock_tokenizer.padding_side == 'left'

