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


@pytest.mark.unit
def test_forward_with_chunking(train_tokenizer):
    """Test model forward with chunked passages: encode_query, encode_passage, compute_maxsim_similarity."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import TrainCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    REAL_TEXT = (
        "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical "
        "development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging "
        "(MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient."
    )
    
    data_args = DataArguments(
        passage_chunk_size=32,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = TrainCollator(data_args=data_args, tokenizer=train_tokenizer)
    
    queries = ["What is cerebral white matter?", "What is MRI?"]
    passages = [REAL_TEXT, "MRI stands for Magnetic Resonance Imaging."]
    q_batch, p_batch, eos_positions = collator([(q, [p], []) for q, p in zip(queries, passages)])
    
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        return MockEncoderOutput(last_hidden_state=torch.randn(batch_size, seq_len, hidden_size))
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = data_args.passage_chunk_size
    model.eos_positions = eos_positions
    model.training = True
    
    output = model(query=q_batch, passage=p_batch)
    
    assert hasattr(output, 'q_reps')
    assert hasattr(output, 'p_reps')
    assert hasattr(output, 'scores')
    assert hasattr(output, 'loss')
    assert output.q_reps.shape == (len(queries), hidden_size)
    
    chunk_reps, chunk_mask = output.p_reps
    assert chunk_reps.shape[0] == len(passages)
    assert chunk_reps.shape[2] == hidden_size
    assert output.scores.shape == (len(queries), len(passages))
    assert output.loss.item() >= 0
    
    # Test MaxSim with known embeddings
    model.eval()
    with torch.no_grad():
        q_reps_test = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        p_reps_test = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ], dtype=torch.float32)
        chunk_mask_test = torch.ones(2, 2)
        
        scores_test = model.compute_maxsim_similarity(q_reps_test, p_reps_test, chunk_mask_test)
        assert torch.allclose(scores_test, torch.ones(2, 2))
    
    # Test padding chunks are ignored
    p_reps_padded = torch.randn(2, 3, hidden_size)
    chunk_mask_padded = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    scores_padded = model.compute_maxsim_similarity(output.q_reps, p_reps_padded, chunk_mask_padded)
    
    for q_idx in range(len(queries)):
        for p_idx in range(len(passages)):
            chunk_scores = torch.einsum('h,ch->c', output.q_reps[q_idx], p_reps_padded[p_idx])
            valid_mask = chunk_mask_padded[p_idx].bool()
            chunk_scores_masked = chunk_scores.clone()
            chunk_scores_masked[~valid_mask] = float('-inf')
            expected_score = chunk_scores_masked.max().item()
            assert torch.allclose(scores_padded[q_idx, p_idx], torch.tensor(expected_score))
