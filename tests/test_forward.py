import sys
from pathlib import Path

import pytest
import torch
from unittest.mock import Mock


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    # tevatron/tests/test_forward.py -> tevatron/ -> tevatron/src
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


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
    tok.eos_token_id = tok.pad_token_id  # Match training setup
    tok.padding_side = "right"
    return tok


@pytest.mark.unit
def test_compute_maxsim_similarity():
    """
    Test compute_maxsim_similarity function to verify MaxSim pooling logic.
    """
    _add_tevatron_src_to_path()
    from tevatron.retriever.modeling.encoder import EncoderModel
    
    # Create a concrete implementation for testing
    class TestEncoderModel(EncoderModel):
        def encode_query(self, qry):
            raise NotImplementedError
        def encode_passage(self, psg):
            raise NotImplementedError
    
    model = TestEncoderModel(encoder=Mock(), pooling='last', normalize=False)
    
    # Test Case 1: Basic MaxSim computation
    # Q=2 queries, P=3 passages, C=4 chunks per passage, H=8 hidden size
    Q, P, C, H = 2, 3, 4, 8
    
    q_reps = torch.randn(Q, H)
    p_reps = torch.randn(P, C, H)
    chunk_mask = torch.ones(P, C)  # All chunks valid
    
    scores = model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
    
    # Verify output shape
    assert scores.shape == (Q, P)
    
    # Verify scores are computed correctly
    # For each query-passage pair, score should be max of chunk similarities
    for q_idx in range(Q):
        for p_idx in range(P):
            # Compute chunk scores manually
            chunk_scores = torch.einsum('h,ch->c', q_reps[q_idx], p_reps[p_idx])
            expected_score = chunk_scores.max().item()
            assert torch.allclose(scores[q_idx, p_idx], torch.tensor(expected_score))
    
    # Test Case 2: With padding (some chunks are invalid)
    chunk_mask_padded = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],  # Passage 0: 3 valid chunks
        [1.0, 1.0, 0.0, 0.0],  # Passage 1: 2 valid chunks
        [1.0, 0.0, 0.0, 0.0],  # Passage 2: 1 valid chunk
    ])
    
    scores_padded = model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask_padded)
    
    # Verify shape
    assert scores_padded.shape == (Q, P)
    
    # Verify that padding chunks don't affect the max
    for q_idx in range(Q):
        for p_idx in range(P):
            # Compute chunk scores manually, masking out invalid chunks
            chunk_scores = torch.einsum('h,ch->c', q_reps[q_idx], p_reps[p_idx])
            # Mask invalid chunks with -inf
            valid_mask = chunk_mask_padded[p_idx].bool()
            chunk_scores_masked = chunk_scores.clone()
            chunk_scores_masked[~valid_mask] = float('-inf')
            expected_score = chunk_scores_masked.max().item()
            assert torch.allclose(scores_padded[q_idx, p_idx], torch.tensor(expected_score))
    
    # Test Case 3: Single chunk per passage
    P_single, C_single = 2, 1
    p_reps_single = torch.randn(P_single, C_single, H)
    chunk_mask_single = torch.ones(P_single, C_single)
    
    scores_single = model.compute_maxsim_similarity(q_reps, p_reps_single, chunk_mask_single)
    assert scores_single.shape == (Q, P_single)
    
    # With single chunk, MaxSim should equal the single chunk similarity
    for q_idx in range(Q):
        for p_idx in range(P_single):
            expected_score = torch.dot(q_reps[q_idx], p_reps_single[p_idx, 0]).item()
            assert torch.allclose(scores_single[q_idx, p_idx], torch.tensor(expected_score))
    
    # Test Case 4: Different number of chunks per passage
    # This tests that max_chunks is handled correctly
    p_reps_uneven = torch.randn(P, C, H)
    # Passage 0: all 4 chunks valid
    # Passage 1: first 2 chunks valid
    # Passage 2: first 1 chunk valid
    chunk_mask_uneven = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    
    scores_uneven = model.compute_maxsim_similarity(q_reps, p_reps_uneven, chunk_mask_uneven)
    assert scores_uneven.shape == (Q, P)
    
    # Verify that only valid chunks are considered
    for q_idx in range(Q):
        for p_idx in range(P):
            chunk_scores = torch.einsum('h,ch->c', q_reps[q_idx], p_reps_uneven[p_idx])
            valid_mask = chunk_mask_uneven[p_idx].bool()
            chunk_scores_masked = chunk_scores.clone()
            chunk_scores_masked[~valid_mask] = float('-inf')
            expected_score = chunk_scores_masked.max().item()
            assert torch.allclose(scores_uneven[q_idx, p_idx], torch.tensor(expected_score))


