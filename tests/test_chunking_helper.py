"""
Unit tests for _chunk_tokens helper function.
"""
import sys
from pathlib import Path
import pytest


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


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
    
    # chunk_size=5 means chunk_len=4
    # First chunk: [0,1,2,3,99] = 5 tokens
    # Second chunk: [4,5,6,7,99] = 5 tokens
    # Total: 10 tokens, but max_length=12 allows one more EOS
    # Third chunk would need at least 1 token + 1 EOS = 2 tokens, but we only have 2 left
    # So we can fit: [8,99] = 2 tokens
    # Total: 12 tokens
    assert len(ids) == 12
    assert ids[-1] == eos_id  # Last token should be EOS
    assert len(eos_pos) == 3
    assert all(ids[pos] == eos_id for pos in eos_pos)


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
    
    # Should have 3 chunks: [0,1,2,99], [3,4,5,99], [6,7,8,99] = 12 tokens
    # Plus [9,99] = 2 tokens, total 14
    assert len(ids) == 14
    assert len(eos_pos) == 4


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
    
    # chunk_size=5 means chunk_len=4
    # Should have 4 chunks: [0-3,99], [4-7,99], [8-11,99], [12-14,99]
    assert len(ids) == 19  # 15 tokens + 4 EOS tokens
    assert len(eos_pos) == 4
    assert all(ids[pos] == eos_id for pos in eos_pos)


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
def test_chunk_tokens_eos_positions_are_correct():
    """Test that EOS positions correctly point to EOS tokens."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.collator import _chunk_tokens
    
    tokens = list(range(10))
    eos_id = 99
    chunk_size = 4
    
    ids, eos_pos = _chunk_tokens(tokens, chunk_size, eos_id)
    
    # Verify all EOS positions contain EOS token
    for pos in eos_pos:
        assert ids[pos] == eos_id
    
    # Verify EOS positions are strictly increasing
    assert all(eos_pos[i] < eos_pos[i + 1] for i in range(len(eos_pos) - 1))


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
    
    assert len(ids) == 10
    assert len(eos_pos) == 2
    # Should have exactly 2 chunks: [0,1,2,3,99], [4,5,6,7,99]
    assert ids == [0, 1, 2, 3, 99, 4, 5, 6, 7, 99]

