import sys
from pathlib import Path

import pytest
import torch


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    # tevatron/tests/test_chunking.py -> tevatron/ -> tevatron/src
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


def _strictly_increasing(xs):
    return all(xs[i] > xs[i - 1] for i in range(1, len(xs)))

REAL_TEXT = (
    "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical "
    "development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging "
    "(MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to "
    "calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in "
    "preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter "
    "development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white "
    "matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to "
    "1.2 microm2/ms. In the posterior limb of the internal capsule, the mean apparent diffusion coefficients at both "
    "times were similar (1.2 versus 1.1 microm2/ms). Relative anisotropy was higher the closer birth was to term with "
    "greater absolute values in the internal capsule than in the central white matter. Preterm infants at term showed "
    "higher mean diffusion coefficients in the central white matter (1.4 +/- 0.24 versus 1.15 +/- 0.09 microm2/ms, "
    "p = 0.016) and lower relative anisotropy in both areas compared with full-term infants (white matter, 10.9 +/- "
    "0.6 versus 22.9 +/- 3.0%, p = 0.001; internal capsule, 24.0 +/- 4.44 versus 33.1 +/- 0.6% p = 0.006). "
    "Nonmyelinated fibers in the corpus callosum were visible by diffusion tensor MRI as early as 28 wk; full-term and "
    "preterm infants at term showed marked differences in white matter fiber organization. The data indicate that "
    "quantitative assessment of water diffusion by diffusion tensor MRI provides insight into microstructural "
    "development in cerebral white matter in living infants"
)
EOS_TOKEN_ID = 151643
PADDING_TOKEN_ID = 151643

@pytest.fixture(scope="session")
def train_tokenizer():
    """
    Use the Qwen 0.6B tokenizer.
    """
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.eos_token_id = tok.pad_token_id
    tok.padding_side = "right"  # finetune_with_chunk.sh uses --padding_side right
    return tok


# ============================================================================
# Unit tests for _chunk_tokens helper function
# ============================================================================

@pytest.mark.unit
def test_chunk_tokens_basic():
    """Test basic chunking functionality."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    eos_id = 99
    chunk_size = 4
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    # chunk_size=4 means chunk_len=3, so chunks are:
    # [0,1,2,99], [3,4,5,99], [6,7,8,99], [9,99]
    expected_ids = [0, 1, 2, 99, 3, 4, 5, 99, 6, 7, 8, 99, 9, 99]
    expected_eos_pos = [3, 7, 11, 13]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_with_max_length():
    """Test chunking with max_length constraint."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(20))
    eos_id = 99
    chunk_size = 5
    max_length = 12
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    
    # Hardcoded golden output: chunk_size=5 means chunk_len=4
    # First chunk: [0,1,2,3,99] = 5 tokens
    # Second chunk: [4,5,6,7,99] = 5 tokens
    # Third chunk: [8,99] = 2 tokens (partial, fits in remaining 2 tokens)
    # Total: 12 tokens
    expected_ids = [0, 1, 2, 3, 99, 4, 5, 6, 7, 99, 8, 99]
    expected_eos_pos = [4, 9, 11]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_max_length_exact_fit():
    """Test chunking when max_length exactly fits chunks."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(10))
    eos_id = 99
    chunk_size = 4
    max_length = 14  # Exactly fits 3 chunks: 3*4 + 2 = 14
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)

    expected_ids = [0, 1, 2, 99, 3, 4, 5, 99, 6, 7, 8, 99, 9, 99]
    expected_eos_pos = [3, 7, 11, 13]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_max_length_too_small():
    """Test chunking when max_length is too small for even one chunk."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(10))
    eos_id = 99
    chunk_size = 4
    max_length = 1  # Too small for even one chunk (need at least 2: 1 token + EOS)
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    
    # Should return empty since we can't fit even one chunk
    assert ids == []
    assert eos_pos == []


@pytest.mark.unit
def test_chunk_tokens_empty_input():
    """Test chunking with empty token list."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = []
    eos_id = 99
    chunk_size = 4
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    assert ids == []
    assert eos_pos == []

@pytest.mark.unit
def test_chunk_tokens_same_length_as_chunk_size():
    """Test chunking when tokens are the same length as chunk_size."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens

    tokens = list(range(20))
    eos_id = 99
    chunk_size = 16
    max_length = 16
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)

    expected_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 99]
    expected_eos_pos = [15]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_single_token():
    """Test chunking with single token."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = [42]
    eos_id = 99
    chunk_size = 4
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    assert ids == [42, 99]
    assert eos_pos == [1]


@pytest.mark.unit
def test_chunk_tokens_no_max_length():
    """Test chunking without max_length constraint."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(15))
    eos_id = 99
    chunk_size = 5
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length=None)
    
    # Hardcoded golden output: chunk_size=5 means chunk_len=4
    # Chunks: [0-3,99], [4-7,99], [8-11,99], [12-14,99]
    expected_ids = [0, 1, 2, 3, 99, 4, 5, 6, 7, 99, 8, 9, 10, 11, 99, 12, 13, 14, 99]
    expected_eos_pos = [4, 9, 14, 18]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_chunk_size_one():
    """Test chunking with chunk_size=1 (invalid, should return empty)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = [1, 2, 3]
    eos_id = 99
    chunk_size = 1
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    # chunk_size=1 is invalid (need at least 2: 1 token + 1 EOS)
    # Should return empty
    assert ids == []
    assert eos_pos == []


@pytest.mark.unit
def test_chunk_tokens_chunk_size_two():
    """Test chunking with chunk_size=2."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = [1, 2, 3, 4, 5]
    eos_id = 99
    chunk_size = 2
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    # chunk_size=2 means chunk_len=1
    # Chunks: [1,99], [2,99], [3,99], [4,99], [5,99]
    expected_ids = [1, 99, 2, 99, 3, 99, 4, 99, 5, 99]
    expected_eos_pos = [1, 3, 5, 7, 9]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_max_length_stops_at_boundary():
    """Test that max_length stops chunking at chunk boundary."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(20))
    eos_id = 99
    chunk_size = 5
    max_length = 10  # Exactly 2 chunks: 2*5 = 10
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    expected_ids = [0, 1, 2, 3, 99, 4, 5, 6, 7, 99]
    expected_eos_pos = [4, 9]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_chunk_size_greater_than_max_length():
    """Test chunking when chunk_size > max_length (only one partial chunk fits)."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(20))
    eos_id = 99
    chunk_size = 10  # chunk_size > max_length
    max_length = 5   # max_length < chunk_size
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    
    # Hardcoded golden output: chunk_size=10 means chunk_len=9, but max_length=5
    # Can only fit: 4 tokens + 1 EOS = 5 tokens (exactly max_length)
    expected_ids = [0, 1, 2, 3, 99]
    expected_eos_pos = [4]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos


@pytest.mark.unit
def test_chunk_tokens_truncation_takes_from_front():
    """Test that truncation when tokens exceed max_length takes from the front (beginning) of the list."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    # Create tokens with distinct values at front and back to verify truncation direction
    tokens = list(range(20))  # [0, 1, 2, ..., 19]
    eos_id = 99
    chunk_size = 5  # chunk_len = 4
    max_length = 8  # Can fit: 1 full chunk (4 tokens + 1 EOS = 5) + 1 partial (2 tokens + 1 EOS = 3) = 8 total
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    
    # Hardcoded golden output: truncation takes from front, so we get [0,1,2,3,99,4,5,99]
    # If it took from back, we'd get [16,17,18,19,99,...] or similar
    expected_ids = [0, 1, 2, 3, 99, 4, 5, 99]
    expected_eos_pos = [4, 7]
    
    assert ids == expected_ids
    assert eos_pos == expected_eos_pos
    
    # Verify it's taking from the front: first token should be 0 (beginning of original list)
    assert ids[0] == 0
    # Verify it's NOT taking from the back: last content token should be 5, not 19
    assert ids[-2] == 5  # Last content token before final EOS
    assert ids[-2] != 19  # Confirms we're not taking from the end


@pytest.mark.unit
def test_chunk_tokens_truncation_then_padding_complex_case(train_tokenizer):
    """Test complex case: tokens exceed max_length (truncation from front), then padding is applied."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens, _pad_and_adjust_eos_positions
    
    # Create a long token sequence that will be truncated
    # Use distinct values to clearly see truncation direction
    tokens = list(range(100, 200))  # [100, 101, 102, ..., 199] - 100 tokens
    eos_id = train_tokenizer.eos_token_id
    pad_id = train_tokenizer.pad_token_id
    chunk_size = 10  # chunk_len = 9
    max_length = 20  # Can fit: 1 full chunk (9 tokens + 1 EOS = 10) + 1 partial (9 tokens + 1 EOS = 10) = 20 total
    
    # Step 1: Chunk with truncation (takes from front)
    chunked_ids, eos_positions = _chunk_tokens(tokens, chunk_size, eos_id, max_length)
    
    # Verify truncation takes from front: should start with 100, not 199
    assert chunked_ids[0] == 100  # First token from original list
    assert chunked_ids[-2] == 117  # Last content token (not 199) - second chunk ends at 117
    assert len(chunked_ids) == 20  # Exactly max_length
    
    # Hardcoded golden output: truncated from front
    # Original: 100 tokens [100-199]
    # After truncation (front): 18 tokens [100-117] + 2 EOS = 20 tokens
    expected_chunked_ids = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, eos_id,  # First chunk: 9 tokens + EOS
        109, 110, 111, 112, 113, 114, 115, 116, 117, eos_id   # Second chunk: 9 tokens + EOS
    ]
    expected_eos_positions = [9, 19]  # EOS positions before padding (list, not list of lists)
    
    assert chunked_ids == expected_chunked_ids
    assert eos_positions == expected_eos_positions
    
    # Step 2: Test left padding with truncation
    all_input_ids = [chunked_ids]
    all_eos_positions = [eos_positions]
    
    # Apply our padding function
    padded_dict_left, adjusted_eos_positions_left = _pad_and_adjust_eos_positions(
        all_input_ids=all_input_ids,
        all_eos_positions=all_eos_positions,
        tokenizer=train_tokenizer,
        padding_side='left',
        pad_to_multiple_of=8,
    )
    expected_padded_ids_left = [
        pad_id, pad_id, pad_id, pad_id,  # 4 padding tokens
        100, 101, 102, 103, 104, 105, 106, 107, 108, eos_id,  # First chunk: 9 tokens + EOS
        109, 110, 111, 112, 113, 114, 115, 116, 117, eos_id   # Second chunk: 9 tokens + EOS
    ]
    expected_attention_mask_left = [0, 0, 0, 0] + [1] * 20  # 4 padding + 20 content
    expected_adjusted_eos_positions_left = [[13, 23]]
    
    assert padded_dict_left['input_ids'][0].tolist() == expected_padded_ids_left
    assert padded_dict_left['attention_mask'][0].tolist() == expected_attention_mask_left
    assert adjusted_eos_positions_left == expected_adjusted_eos_positions_left

# ============================================================================
# Unit tests for _pad_and_adjust_eos_positions helper function
# ============================================================================

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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'right'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=4,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'left'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=4,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'left'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
        pad_to_multiple_of=3,
    )
    
    # Hardcoded golden output
    expected_input_ids = torch.tensor([
        [1, 2, eos_id],
        [3, 4, eos_id],
    ])
    expected_attention_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
    ])
    expected_eos_positions = [[2], [2]]  # EOS positions unchanged for right padding
    
    assert torch.equal(padded_dict['input_ids'], expected_input_ids)
    assert torch.equal(padded_dict['attention_mask'], expected_attention_mask)
    assert adjusted_eos_positions == expected_eos_positions
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'right'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=3,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'right'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=4,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'right'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=1,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])


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
    
    # Verify behavior matches tokenizer.pad directly
    train_tokenizer.padding_side = 'left'
    direct_padded = train_tokenizer.pad(
        {'input_ids': all_input_ids},
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors='pt',
    )
    assert torch.equal(padded_dict['input_ids'], direct_padded['input_ids'])
    assert torch.equal(padded_dict['attention_mask'], direct_padded['attention_mask'])




@pytest.mark.unit
def test_train_collator_chunked_passages(train_tokenizer):
    """Test chunking with passage_max_len=512, passage_chunk_size=256."""
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        passage_max_len=512,
        passage_chunk_size=256,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])

    got_ids = d_collated["input_ids"][0].tolist()
    got_mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 2 chunks (255 tokens + EOS, 174 tokens + EOS) = 431 tokens, padded to 432
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349,
        2326, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77, 284, 220, 16,
        22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239, 315, 6811,
        37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434, 320, 77, 284,
        220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622, 4158, 4925,
        279, 3076, 9981, 57330, 35606, 518, 220, 17, 23, 73760, 572, 1550, 11, 220, 16, 13, 23,
        19197, 441, 17, 58634, 11, 323, 24938, 8841, 4647, 311, 220, 16, 13, 17, 19197, 441, 17,
        58634, 13, 758, 279, 44900, 47594, 315, 279, 5306, 47639, 11, 279, 3076, 9981, 57330,
        36829, 518, 2176, 3039, 1033, 4428, 320, 16, 13, 17, 19041, 220, 16, 13, 16, 19197, 441,
        17, 58634, 568, 39402, 458, 285, 354, 17764, 572, 5080, 279, 12128, 7194, 572, 311, 4647,
        448, 7046, 10740, 2750, 304, 279, 5306, 47639, 1091, 304, 279, 8622, 4158, 4925, 13, 4968,
        4991, 41434, 518, 4647, 8542, 5080, 3076, 57330, 36829, 304, 279, 8622, 4158, 4925, 320,
        16, 13, 19, 51615, 220, 15, 13, 17, 19, 19041, 220, 16, 13, 16, EOS_TOKEN_ID, 20, 51615,
        220, 15, 13, 15, 24, 19197, 441, 17, 58634, 11, 281, 284, 220, 15, 13, 15, 16, 21, 8, 323,
        4722, 8674, 458, 285, 354, 17764, 304, 2176, 5671, 7707, 448, 2480, 9663, 41434, 320,
        5782, 4925, 11, 220, 16, 15, 13, 24, 51615, 220, 15, 13, 21, 19041, 220, 17, 17, 13, 24,
        51615, 220, 18, 13, 15, 13384, 281, 284, 220, 15, 13, 15, 15, 16, 26, 5306, 47639, 11,
        220, 17, 19, 13, 15, 51615, 220, 19, 13, 19, 19, 19041, 220, 18, 18, 13, 16, 51615, 220,
        15, 13, 21, 4, 281, 284, 220, 15, 13, 15, 15, 21, 568, 11581, 2408, 301, 15479, 48674,
        304, 279, 42094, 1620, 385, 1242, 1033, 9434, 553, 57330, 15626, 51360, 438, 4124, 438,
        220, 17, 23, 73760, 26, 2480, 9663, 323, 855, 4991, 41434, 518, 4647, 8542, 12864, 11799,
        304, 4158, 4925, 23788, 7321, 13, 576, 821, 13216, 429, 46516, 15449, 315, 3015, 57330,
        553, 57330, 15626, 51360, 5707, 20017, 1119, 8003, 95697, 4401, 304, 59645, 4158, 4925,
        304, 5382, 41434, EOS_TOKEN_ID, PADDING_TOKEN_ID
    ]
    expected_mask = [1] * 431 + [0]  # 431 ones + 1 zero
    expected_eos_positions = [[255, 430]]

    assert sum(got_mask) == 431
    assert len(got_ids) == 432  # Padded to multiple of 16
    assert eos_positions == expected_eos_positions
    assert got_ids == expected_ids
    assert got_mask == expected_mask
    assert got_ids[255] == train_tokenizer.eos_token_id
    assert got_ids[430] == train_tokenizer.eos_token_id
    assert got_mask[255] == 1
    assert got_mask[430] == 1


@pytest.mark.unit
def test_train_collator_chunked_passages_left_padding(train_tokenizer):
    """Test chunking with passage_max_len=512, passage_chunk_size=256, left padding."""
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        passage_max_len=512,
        passage_chunk_size=256,
        pad_to_multiple_of=16,
        padding_side="left",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])

    got_ids = d_collated["input_ids"][0].tolist()
    got_mask = d_collated["attention_mask"][0].tolist()
    expected_ids = [ PADDING_TOKEN_ID,
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349,
        2326, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77, 284, 220, 16,
        22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239, 315, 6811,
        37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434, 320, 77, 284,
        220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622, 4158, 4925,
        279, 3076, 9981, 57330, 35606, 518, 220, 17, 23, 73760, 572, 1550, 11, 220, 16, 13, 23,
        19197, 441, 17, 58634, 11, 323, 24938, 8841, 4647, 311, 220, 16, 13, 17, 19197, 441, 17,
        58634, 13, 758, 279, 44900, 47594, 315, 279, 5306, 47639, 11, 279, 3076, 9981, 57330,
        36829, 518, 2176, 3039, 1033, 4428, 320, 16, 13, 17, 19041, 220, 16, 13, 16, 19197, 441,
        17, 58634, 568, 39402, 458, 285, 354, 17764, 572, 5080, 279, 12128, 7194, 572, 311, 4647,
        448, 7046, 10740, 2750, 304, 279, 5306, 47639, 1091, 304, 279, 8622, 4158, 4925, 13, 4968,
        4991, 41434, 518, 4647, 8542, 5080, 3076, 57330, 36829, 304, 279, 8622, 4158, 4925, 320,
        16, 13, 19, 51615, 220, 15, 13, 17, 19, 19041, 220, 16, 13, 16, EOS_TOKEN_ID, 20, 51615,
        220, 15, 13, 15, 24, 19197, 441, 17, 58634, 11, 281, 284, 220, 15, 13, 15, 16, 21, 8, 323,
        4722, 8674, 458, 285, 354, 17764, 304, 2176, 5671, 7707, 448, 2480, 9663, 41434, 320,
        5782, 4925, 11, 220, 16, 15, 13, 24, 51615, 220, 15, 13, 21, 19041, 220, 17, 17, 13, 24,
        51615, 220, 18, 13, 15, 13384, 281, 284, 220, 15, 13, 15, 15, 16, 26, 5306, 47639, 11,
        220, 17, 19, 13, 15, 51615, 220, 19, 13, 19, 19, 19041, 220, 18, 18, 13, 16, 51615, 220,
        15, 13, 21, 4, 281, 284, 220, 15, 13, 15, 15, 21, 568, 11581, 2408, 301, 15479, 48674,
        304, 279, 42094, 1620, 385, 1242, 1033, 9434, 553, 57330, 15626, 51360, 438, 4124, 438,
        220, 17, 23, 73760, 26, 2480, 9663, 323, 855, 4991, 41434, 518, 4647, 8542, 12864, 11799,
        304, 4158, 4925, 23788, 7321, 13, 576, 821, 13216, 429, 46516, 15449, 315, 3015, 57330,
        553, 57330, 15626, 51360, 5707, 20017, 1119, 8003, 95697, 4401, 304, 59645, 4158, 4925,
        304, 5382, 41434, EOS_TOKEN_ID
    ]
    expected_mask = [0] + [1] * 431  # 1 padding + 431 content
    expected_eos_positions = [[256, 431]]

    assert got_ids == expected_ids
    assert got_mask == expected_mask
    assert eos_positions == expected_eos_positions


@pytest.mark.unit
def test_chunked_collator_with_multiple_passages(train_tokenizer):
    """Test TrainCollator with chunking enabled returns (q_batch, p_batch, eos_positions)."""
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        query_max_len=32,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
        train_group_size=2,
        passage_chunk_size=32,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    features = [
        (("q1", None, None, None), [(REAL_TEXT, None, None, None), (REAL_TEXT, None, None, None)]),
    ]
    
    q_batch, p_batch, eos_positions = collator(features)
    
    # Hardcoded golden output: both passages have 2 chunks (31 tokens + EOS, 31 tokens + EOS) = 64 tokens each
    expected_ids_0 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        EOS_TOKEN_ID, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311,
        6629, 279, 9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311,
        90684, 349, EOS_TOKEN_ID
    ]
    expected_mask_0 = [1] * 64
    expected_eos_0 = [31, 63]
    
    expected_ids_1 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        EOS_TOKEN_ID, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311,
        6629, 279, 9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311,
        90684, 349, EOS_TOKEN_ID
    ]
    expected_mask_1 = [1] * 64
    expected_eos_1 = [31, 63]
    
    assert p_batch["input_ids"].shape[0] == 2
    assert len(eos_positions) == 2
    
    got_ids_0 = p_batch["input_ids"][0].tolist()
    got_mask_0 = p_batch["attention_mask"][0].tolist()
    got_ids_1 = p_batch["input_ids"][1].tolist()
    got_mask_1 = p_batch["attention_mask"][1].tolist()
    
    assert got_ids_0 == expected_ids_0
    assert got_mask_0 == expected_mask_0
    assert eos_positions[0] == expected_eos_0
    assert got_ids_1 == expected_ids_1
    assert got_mask_1 == expected_mask_1
    assert eos_positions[1] == expected_eos_1
    
    for i in range(p_batch["input_ids"].shape[0]):
        got_ids = p_batch["input_ids"][i].tolist()
        got_mask = p_batch["attention_mask"][i].tolist()
        
        assert len(eos_positions[i]) > 0
        assert _strictly_increasing(eos_positions[i])
        for eos_pos in eos_positions[i]:
            assert got_ids[eos_pos] == train_tokenizer.eos_token_id
            assert got_mask[eos_pos] == 1
        assert len(got_ids) == 64


@pytest.mark.unit
def test_chunking_capped_to_maxlen_chunk_size_64(train_tokenizer):
    """When chunk_size >= max_len, chunking is capped to max_len with one EOS (chunk_size=64)."""
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    long_text = (REAL_TEXT + " ") * 20
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 63 tokens + 1 EOS = 64 tokens
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349,
        2326, EOS_TOKEN_ID
    ]
    expected_mask = [1] * 64
    expected_eos_positions = [[63]]

    assert sum(mask) == 64
    assert len(ids) == 64
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert mask == expected_mask
    assert ids[63] == EOS_TOKEN_ID
    assert EOS_TOKEN_ID not in ids[:63]
    assert _strictly_increasing(eos_positions[0])


@pytest.mark.unit
def test_chunking_capped_to_maxlen_chunk_size_128(train_tokenizer):
    """When chunk_size >= max_len, chunking is capped to max_len with one EOS (chunk_size=128)."""
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    long_text = (REAL_TEXT + " ") * 20
    data_args = DataArguments(
        passage_chunk_size=128,
        passage_max_len=64,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([long_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 63 tokens + 1 EOS = 64 tokens
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684, 349,
        2326, EOS_TOKEN_ID
    ]
    expected_mask = [1] * 64
    expected_eos_positions = [[63]]

    assert sum(mask) == 64
    assert len(ids) == 64
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert mask == expected_mask
    assert ids[63] == EOS_TOKEN_ID
    assert EOS_TOKEN_ID not in ids[:63]
    assert _strictly_increasing(eos_positions[0])


@pytest.mark.unit
def test_chunking_short_passage_shorter_than_chunk_size(train_tokenizer):
    """
    When passage is shorter than chunk_size, it should still get one chunk with EOS,
    and padding should be applied to pad_to_multiple_of.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    short_text = "Hello world"
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([short_text])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: "Hello world" -> 2 tokens + 1 EOS = 3 tokens, padded to 16
    expected_ids = [9707, 1879, EOS_TOKEN_ID] + [PADDING_TOKEN_ID] * 13  # 3 content + 13 padding
    expected_eos_positions = [[2]]
    expected_mask = [1, 1, 1] + [0] * 13  # 3 ones + 13 zeros

    assert sum(mask) == 3
    assert len(ids) == 16  # Padded to multiple of 16
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[2] == EOS_TOKEN_ID  # EOS at position 2
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


@pytest.mark.unit
def test_chunking_passage_needs_padding_unpadded_not_multiple_of_pad_to_multiple_of(train_tokenizer):
    """
    When unpadded length is not a multiple of pad_to_multiple_of, padding should be added.
    This tests: unpadded_len=50, pad_to_multiple_of=16 -> padded_len=64.
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    data_args = DataArguments(
        passage_chunk_size=32,
        passage_max_len=50,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages([REAL_TEXT])
    ids = d_collated["input_ids"][0].tolist()
    mask = d_collated["attention_mask"][0].tolist()

    # Hardcoded golden output: 50 unpadded tokens (2 chunks: 31+1 EOS, 18+1 EOS), padded to 64
    expected_ids = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        EOS_TOKEN_ID, 56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629,
        279, 9981, 57330, EOS_TOKEN_ID
    ] + [PADDING_TOKEN_ID] * 14  # 50 content + 14 padding
    expected_eos_positions = [[31, 49]]
    expected_mask = [1] * 50 + [0] * 14  # 50 ones + 14 zeros
    assert sum(mask) == 50
    assert len(ids) == 64  # Padded to multiple of 16
    assert eos_positions == expected_eos_positions
    assert ids == expected_ids
    assert ids[31] == EOS_TOKEN_ID  # First EOS
    assert ids[49] == EOS_TOKEN_ID  # Second EOS
    assert mask == expected_mask
    assert _strictly_increasing(eos_positions[0])


@pytest.mark.unit
def test_chunking_multiple_passages_different_lengths(train_tokenizer):
    """
    Test batch processing with multiple passages of different lengths:
    - Short passage (2 tokens)
    - Medium passage (18 tokens)
    - Long passage (128 tokens, multiple chunks)
    - Very long passage (158 tokens, multiple chunks)
    All should be padded to the same length (longest unpadded length rounded up to pad_to_multiple_of).
    """
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator

    # Create a passage that will result in ~158 tokens
    # REAL_TEXT is ~431 tokens, so we'll use a portion of it repeated or extended
    long_passage = REAL_TEXT + " " + REAL_TEXT[:200]
    
    texts = ["Short", REAL_TEXT[:100], REAL_TEXT, long_passage]
    data_args = DataArguments(
        passage_chunk_size=64,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages(texts)
    
    expected_ids_0 = [12472, EOS_TOKEN_ID] + [PADDING_TOKEN_ID] * 126
    expected_mask_0 = [1, 1] + [0] * 126
    expected_eos_0 = [1]
    
    # Passage 1: REAL_TEXT[:100] -> 17 tokens + 1 EOS = 18 tokens, padded to 160
    expected_ids_1 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        1062, EOS_TOKEN_ID
    ] + [PADDING_TOKEN_ID] * 110
    expected_mask_1 = [1] * 18 + [0] * 110
    expected_eos_1 = [17]
    
    # Passage 2: REAL_TEXT -> 2 chunks (63+1 EOS, 63+1 EOS) = 128 tokens, padded to 160
    expected_ids_2 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77,
        284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239,
        315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434,
        320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622,
        EOS_TOKEN_ID
    ]
    expected_mask_2 = [1] * 128
    expected_eos_2 = [63, 127]
    
    expected_ids_3 = [
        74290, 804, 315, 279, 17646, 315, 59645, 4158, 4925, 304, 279, 11220, 3738, 8109, 646, 7802,
        82519, 4401, 323, 1102, 304, 15629, 35701, 13, 362, 1555, 8569, 57330, 12635, 291, 23970,
        56981, 31658, 320, 78670, 8, 8500, 448, 57330, 15626, 6358, 572, 9251, 311, 6629, 279,
        9981, 57330, 35606, 11, 311, 11047, 8674, 458, 285, 354, 17764, 11, 323, 311, 90684,
        349, 2326, EOS_TOKEN_ID, 32420, 23788, 17646, 304, 59645, 4158, 4925, 304, 855, 4991, 320, 77,
        284, 220, 16, 22, 8, 323, 2480, 9663, 41434, 320, 77, 284, 220, 22, 568, 2014, 8552, 6239,
        315, 6811, 37854, 389, 59645, 4158, 4925, 4401, 11, 4124, 12743, 367, 855, 4991, 41434,
        320, 77, 284, 220, 16, 15, 8, 1033, 19476, 264, 2086, 882, 518, 4647, 13, 758, 279, 8622,
        EOS_TOKEN_ID
    ]
    expected_mask_3 = [1] * 128
    expected_eos_3 = [63, 127]

    ids_0 = d_collated["input_ids"][0].tolist()
    mask_0 = d_collated["attention_mask"][0].tolist()
    ids_1 = d_collated["input_ids"][1].tolist()
    mask_1 = d_collated["attention_mask"][1].tolist()
    ids_2 = d_collated["input_ids"][2].tolist()
    mask_2 = d_collated["attention_mask"][2].tolist()
    ids_3 = d_collated["input_ids"][3].tolist()
    mask_3 = d_collated["attention_mask"][3].tolist()

    # Passage 0 assertions
    assert sum(mask_0) == 2
    assert len(ids_0) == 128
    assert ids_0 == expected_ids_0
    assert mask_0 == expected_mask_0
    assert eos_positions[0] == expected_eos_0
    
    # Passage 1 assertions
    assert sum(mask_1) == 18
    assert len(ids_1) == 128
    assert ids_1 == expected_ids_1
    assert mask_1 == expected_mask_1
    assert eos_positions[1] == expected_eos_1
    
    # Passage 2 assertions
    assert sum(mask_2) == 128
    assert len(ids_2) == 128
    assert ids_2 == expected_ids_2
    assert mask_2 == expected_mask_2
    assert eos_positions[2] == expected_eos_2
    assert _strictly_increasing(eos_positions[2])
    
    # Passage 3 assertions
    assert sum(mask_3) == 128
    assert len(ids_3) == 128
    assert eos_positions[3] == expected_eos_3
    assert ids_3 == expected_ids_3
    assert mask_3 == expected_mask_3
