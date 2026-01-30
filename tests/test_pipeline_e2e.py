"""
End-to-end chunking pipeline tests with hardcoded examples.

These tests trace the full data flow through the chunking pipeline using
small, human-readable values so you can follow each transformation:

  _chunk_tokens → _pad_and_adjust_eos_positions → _pooling_chunked
  → compute_maxsim_similarity → search aggregation

No model downloads needed — all tests use hardcoded tensors and mocks.
"""
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_tevatron_src_to_path()
from tevatron.retriever.collator import _chunk_tokens, _pad_and_adjust_eos_positions
from tevatron.retriever.modeling.encoder import EncoderModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TestEncoderModel(EncoderModel):
    """Concrete subclass so we can call compute_maxsim_similarity."""
    def encode_query(self, qry):
        raise NotImplementedError
    def encode_passage(self, psg):
        raise NotImplementedError


def _make_model():
    return _TestEncoderModel(encoder=Mock(), pooling='last', normalize=False)


# ---------------------------------------------------------------------------
# Test 1: _chunk_tokens with hardcoded tokens
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunk_tokens_hardcoded():
    """
    tokens:     [10, 20, 30, 40, 50, 60, 70, 80]
    chunk_size: 3  (2 content tokens + 1 EOS)
    eos_token:  99

    Expected chunking:
      chunk 0: [10, 20, 99]       eos at index 2
      chunk 1: [30, 40, 99]       eos at index 5
      chunk 2: [50, 60, 99]       eos at index 8
      chunk 3: [70, 80, 99]       eos at index 11

    Result: [10, 20, 99, 30, 40, 99, 50, 60, 99, 70, 80, 99]
    """
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    ids, eos_pos = _chunk_tokens(tokens, chunk_size=3, eos_token_id=99)

    assert ids == [10, 20, 99, 30, 40, 99, 50, 60, 99, 70, 80, 99]
    assert eos_pos == [2, 5, 8, 11]

    # Verify every EOS position actually holds the EOS token
    for pos in eos_pos:
        assert ids[pos] == 99, f"Expected EOS (99) at position {pos}, got {ids[pos]}"


# ---------------------------------------------------------------------------
# Test 2: _chunk_tokens with max_length truncation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunk_tokens_with_max_length_hardcoded():
    """
    Same tokens, but max_length=9 truncates after 3 full chunks.

    chunk 0: [10, 20, 99]       total_length=3
    chunk 1: [30, 40, 99]       total_length=6
    chunk 2: [50, 60, 99]       total_length=9  → hits max_length, stop

    Result: [10, 20, 99, 30, 40, 99, 50, 60, 99]
    """
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    ids, eos_pos = _chunk_tokens(tokens, chunk_size=3, eos_token_id=99, max_length=9)

    assert ids == [10, 20, 99, 30, 40, 99, 50, 60, 99]
    assert eos_pos == [2, 5, 8]


@pytest.mark.unit
def test_chunk_tokens_max_length_partial_chunk():
    """
    max_length=7: fits 2 full chunks (6 tokens) + 1 partial.

    chunk 0: [10, 20, 99]       total_length=3
    chunk 1: [30, 40, 99]       total_length=6
    chunk 2: next chunk_size=3, total_length+3=9 > 7
             remaining = 7 - 6 - 1 = 0  → no room for content, break

    Result: [10, 20, 99, 30, 40, 99]
    """
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    ids, eos_pos = _chunk_tokens(tokens, chunk_size=3, eos_token_id=99, max_length=7)

    # Only 2 full chunks fit; remaining=0 means no partial chunk
    assert ids == [10, 20, 99, 30, 40, 99]
    assert eos_pos == [2, 5]


@pytest.mark.unit
def test_chunk_tokens_max_length_partial_with_room():
    """
    max_length=8: fits 2 full chunks (6 tokens) + partial with 1 content + EOS.

    chunk 0: [10, 20, 99]       total_length=3
    chunk 1: [30, 40, 99]       total_length=6
    chunk 2: next chunk_size=3, total_length+3=9 > 8
             remaining = 8 - 6 - 1 = 1  → take 1 token + EOS

    Result: [10, 20, 99, 30, 40, 99, 50, 99]
    """
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    ids, eos_pos = _chunk_tokens(tokens, chunk_size=3, eos_token_id=99, max_length=8)

    assert ids == [10, 20, 99, 30, 40, 99, 50, 99]
    assert eos_pos == [2, 5, 7]


# ---------------------------------------------------------------------------
# Test 3: _pad_and_adjust_eos_positions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_tokenizer():
    """Minimal tokenizer mock that supports .pad()."""
    _add_tevatron_src_to_path()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.eos_token_id = tok.pad_token_id
    return tok


@pytest.mark.unit
def test_pad_and_adjust_eos_right_padding(mock_tokenizer):
    """
    Right padding: EOS positions stay the same.

    seq 0: [10, 20, 99, 30, 40, 99]  length=6, eos_pos=[2, 5]
    seq 1: [10, 99, 30, 99]          length=4, eos_pos=[1, 3]

    After right-pad to length 8 (multiple of 16 → actually padded to 16):
    seq 0: [10, 20, 99, 30, 40, 99, PAD, PAD, ...]  eos still [2, 5]
    seq 1: [10, 99, 30, 99, PAD, PAD, PAD, PAD, ...] eos still [1, 3]
    """
    all_ids = [[10, 20, 99, 30, 40, 99], [10, 99, 30, 99]]
    all_eos = [[2, 5], [1, 3]]

    d_collated, adjusted_eos = _pad_and_adjust_eos_positions(
        all_ids, all_eos, mock_tokenizer, padding_side='right', pad_to_multiple_of=16
    )

    # Right padding: positions unchanged
    assert adjusted_eos[0] == [2, 5]
    assert adjusted_eos[1] == [1, 3]

    # Verify the actual tokens at EOS positions
    for seq_idx, eos_positions in enumerate(adjusted_eos):
        for pos in eos_positions:
            assert d_collated['input_ids'][seq_idx][pos].item() == 99


@pytest.mark.unit
def test_pad_and_adjust_eos_left_padding(mock_tokenizer):
    """
    Left padding: EOS positions shift right by padding amount.

    seq 0: [10, 20, 99, 30, 40, 99]  length=6, eos_pos=[2, 5]
    seq 1: [10, 99, 30, 99]          length=4, eos_pos=[1, 3]

    Padded to 16 (pad_to_multiple_of=16):
    seq 0: padding=10, eos → [2+10, 5+10] = [12, 15]
    seq 1: padding=12, eos → [1+12, 3+12] = [13, 15]
    """
    all_ids = [[10, 20, 99, 30, 40, 99], [10, 99, 30, 99]]
    all_eos = [[2, 5], [1, 3]]

    d_collated, adjusted_eos = _pad_and_adjust_eos_positions(
        all_ids, all_eos, mock_tokenizer, padding_side='left', pad_to_multiple_of=16
    )

    padded_len = d_collated['input_ids'].shape[1]
    # seq 0: padding = padded_len - 6
    pad0 = padded_len - 6
    assert adjusted_eos[0] == [2 + pad0, 5 + pad0]
    # seq 1: padding = padded_len - 4
    pad1 = padded_len - 4
    assert adjusted_eos[1] == [1 + pad1, 3 + pad1]

    # Verify the actual tokens at adjusted EOS positions
    for seq_idx, eos_positions in enumerate(adjusted_eos):
        for pos in eos_positions:
            assert d_collated['input_ids'][seq_idx][pos].item() == 99


# ---------------------------------------------------------------------------
# Test 4: _pooling_chunked with hardcoded hidden states
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pooling_chunked_hardcoded():
    """
    Simulate _pooling_chunked with known hidden states.

    Setup:
      B=2 passages, seq_len=12, H=4
      eos_positions = [[2, 5, 8], [3, 7]]  (3 chunks, 2 chunks)
      max_chunks = 3

    We place known vectors at EOS positions:
      hidden[0, 2] = [1, 0, 0, 0]   (passage 0, chunk 0)
      hidden[0, 5] = [0, 1, 0, 0]   (passage 0, chunk 1)
      hidden[0, 8] = [0, 0, 1, 0]   (passage 0, chunk 2)
      hidden[1, 3] = [0, 0, 0, 1]   (passage 1, chunk 0)
      hidden[1, 7] = [1, 1, 0, 0]   (passage 1, chunk 1)

    Expected chunk_reps:
      [0]: [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
      [1]: [[0,0,0,1], [1,1,0,0], [0,0,0,0]]  (chunk 2 is padding)

    Expected chunk_mask:
      [[1, 1, 1],
       [1, 1, 0]]
    """
    from tevatron.retriever.modeling.dense import DenseModel

    B, seq_len, H = 2, 12, 4
    hidden = torch.zeros(B, seq_len, H)

    # Place known vectors at EOS positions
    hidden[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    hidden[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    hidden[0, 8] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    hidden[1, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    hidden[1, 7] = torch.tensor([1.0, 1.0, 0.0, 0.0])

    eos_positions = [[2, 5, 8], [3, 7]]

    # Create DenseModel with normalize=False so we get raw vectors
    model = DenseModel(encoder=Mock(), pooling='last', normalize=False)
    model.passage_chunk_size = 1  # Enable chunked mode

    chunk_reps, chunk_mask = model._pooling_chunked(hidden, eos_positions)

    # Shape checks
    assert chunk_reps.shape == (2, 3, 4), f"Expected (2,3,4), got {chunk_reps.shape}"
    assert chunk_mask.shape == (2, 3), f"Expected (2,3), got {chunk_mask.shape}"

    # Passage 0: all 3 chunks valid
    assert torch.allclose(chunk_reps[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(chunk_reps[0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(chunk_reps[0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # Passage 1: 2 chunks valid, chunk 2 is zero-padded
    assert torch.allclose(chunk_reps[1, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(chunk_reps[1, 1], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(chunk_reps[1, 2], torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # Mask
    expected_mask = torch.tensor([[1.0, 1.0, 1.0],
                                   [1.0, 1.0, 0.0]])
    assert torch.allclose(chunk_mask, expected_mask)


# ---------------------------------------------------------------------------
# Test 5: compute_maxsim_similarity with hardcoded values
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_maxsim_hardcoded():
    """
    Hardcoded MaxSim with manually computed expected values.

    q_reps: Q=1, H=4
      q0 = [1, 0, 0, 0]

    p_reps: P=2, C=3, H=4
      p0_c0 = [1, 0, 0, 0]   → dot(q0, p0_c0) = 1.0
      p0_c1 = [0, 1, 0, 0]   → dot(q0, p0_c1) = 0.0
      p0_c2 = [0.5, 0.5, 0, 0] → dot(q0, p0_c2) = 0.5
      → MaxSim(q0, p0) = max(1.0, 0.0, 0.5) = 1.0

      p1_c0 = [0, 0, 0, 1]   → dot(q0, p1_c0) = 0.0
      p1_c1 = [0.3, 0, 0, 0] → dot(q0, p1_c1) = 0.3
      p1_c2 = MASKED (padding)
      → MaxSim(q0, p1) = max(0.0, 0.3) = 0.3

    chunk_mask:
      [[1, 1, 1],
       [1, 1, 0]]
    """
    model = _make_model()

    q_reps = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # [1, 4]

    p_reps = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],     # p0_c0
         [0.0, 1.0, 0.0, 0.0],     # p0_c1
         [0.5, 0.5, 0.0, 0.0]],    # p0_c2
        [[0.0, 0.0, 0.0, 1.0],     # p1_c0
         [0.3, 0.0, 0.0, 0.0],     # p1_c1
         [9.9, 9.9, 9.9, 9.9]],    # p1_c2 (should be masked, high values to catch bugs)
    ])  # [2, 3, 4]

    chunk_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])  # [2, 3]

    scores = model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)

    assert scores.shape == (1, 2)
    assert torch.allclose(scores[0, 0], torch.tensor(1.0))
    assert torch.allclose(scores[0, 1], torch.tensor(0.3))

    # If masking were broken, p1 score would be 9.9*1 + 9.9*0 + ... = 9.9
    # This confirms masking works correctly
    assert scores[0, 1].item() < 1.0, "Masked chunk leaked into MaxSim!"


# ---------------------------------------------------------------------------
# Test 6: Full pipeline end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_full_pipeline_e2e():
    """
    Trace the full pipeline with hardcoded values:

    Step 1: _chunk_tokens
      tokens = [10, 20, 30, 40, 50, 60]
      chunk_size=3, eos=99
      → ids = [10, 20, 99, 30, 40, 99, 50, 60, 99]
      → eos_positions = [2, 5, 8]

    Step 2: _pooling_chunked
      Fake hidden_state [1, 9, 4] with known vectors at EOS positions:
        hidden[0, 2] = [1, 0, 0, 0]  (chunk 0 embedding)
        hidden[0, 5] = [0, 1, 0, 0]  (chunk 1 embedding)
        hidden[0, 8] = [0, 0, 1, 0]  (chunk 2 embedding)
      → chunk_reps = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]  shape [1, 3, 4]
      → chunk_mask = [[1, 1, 1]]

    Step 3: compute_maxsim_similarity
      query = [0, 1, 0, 0]   (aligns with chunk 1)
      → dot products: [0.0, 1.0, 0.0]
      → MaxSim = 1.0 (chunk 1 wins)

    Step 4: Search aggregation (simulated)
      Same chunks stored as separate FAISS entries for "doc_A"
      Query finds chunks with scores [0.0, 1.0, 0.0]
      → max per doc = 1.0
    """
    from tevatron.retriever.modeling.dense import DenseModel

    # --- Step 1: Chunk tokens ---
    tokens = [10, 20, 30, 40, 50, 60]
    ids, eos_pos = _chunk_tokens(tokens, chunk_size=3, eos_token_id=99)
    assert ids == [10, 20, 99, 30, 40, 99, 50, 60, 99]
    assert eos_pos == [2, 5, 8]

    # --- Step 2: Pooling at EOS positions ---
    seq_len = len(ids)  # 9
    H = 4
    hidden = torch.zeros(1, seq_len, H)
    # Plant known embeddings at EOS positions
    hidden[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # chunk 0
    hidden[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # chunk 1
    hidden[0, 8] = torch.tensor([0.0, 0.0, 1.0, 0.0])  # chunk 2

    model = DenseModel(encoder=Mock(), pooling='last', normalize=False)
    model.passage_chunk_size = 1
    chunk_reps, chunk_mask = model._pooling_chunked(hidden, [eos_pos])

    assert chunk_reps.shape == (1, 3, 4)
    assert torch.allclose(chunk_mask, torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(chunk_reps[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(chunk_reps[0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(chunk_reps[0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # --- Step 3: MaxSim scoring ---
    # Query that aligns perfectly with chunk 1
    q_reps = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # [1, 4]

    scores = model.compute_maxsim_similarity(q_reps, chunk_reps, chunk_mask)
    assert scores.shape == (1, 1)
    assert torch.allclose(scores[0, 0], torch.tensor(1.0))

    # --- Step 4: Simulate search aggregation ---
    # In real search, each chunk is a separate FAISS entry
    p_lookup = [("doc_A", 0), ("doc_A", 1), ("doc_A", 2)]
    # Simulate FAISS scores (dot products with query)
    chunk_scores = [0.0, 1.0, 0.0]  # chunk 1 wins

    doc_max_scores = defaultdict(lambda: float('-inf'))
    for (doc_id, _), score in zip(p_lookup, chunk_scores):
        doc_max_scores[doc_id] = max(doc_max_scores[doc_id], score)

    assert doc_max_scores["doc_A"] == 1.0


# ---------------------------------------------------------------------------
# Test 7: Search MaxSim aggregation with multiple documents
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_search_maxsim_aggregation_hardcoded():
    """
    Simulate search_queries_chunked aggregation logic.

    p_lookup (chunks in FAISS index):
      idx 0: ("doc1", 0)  score=0.8
      idx 1: ("doc1", 1)  score=0.3
      idx 2: ("doc1", 2)  score=0.9   ← doc1 best
      idx 3: ("doc2", 0)  score=0.7
      idx 4: ("doc2", 1)  score=0.85  ← doc2 best

    Expected MaxSim per document:
      doc1: max(0.8, 0.3, 0.9) = 0.9
      doc2: max(0.7, 0.85) = 0.85

    Ranking: doc1 (0.9) > doc2 (0.85)
    """
    p_lookup = [
        ("doc1", 0), ("doc1", 1), ("doc1", 2),
        ("doc2", 0), ("doc2", 1),
    ]

    # Simulated FAISS results for one query
    indices = [2, 0, 4, 3, 1]  # FAISS returns indices into p_lookup
    scores  = [0.9, 0.8, 0.85, 0.7, 0.3]  # corresponding scores

    # Replicate the aggregation logic from search_queries_chunked
    doc_max_scores = defaultdict(lambda: float('-inf'))
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        doc_id, chunk_idx = p_lookup[idx]
        doc_max_scores[doc_id] = max(doc_max_scores[doc_id], score)

    sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)

    assert sorted_docs[0] == ("doc1", 0.9)
    assert sorted_docs[1] == ("doc2", 0.85)

    # Verify MaxSim picked the correct best chunk for each doc
    assert doc_max_scores["doc1"] == 0.9   # chunk 2 won
    assert doc_max_scores["doc2"] == 0.85  # chunk 1 won


@pytest.mark.unit
def test_search_aggregation_with_faiss_negative_indices():
    """
    FAISS returns -1 for insufficient results. These must be skipped.
    """
    p_lookup = [("doc1", 0), ("doc1", 1), ("doc2", 0)]

    indices = [0, 1, -1, 2, -1]
    scores  = [0.8, 0.3, 0.0, 0.7, 0.0]

    doc_max_scores = defaultdict(lambda: float('-inf'))
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        doc_id, chunk_idx = p_lookup[idx]
        doc_max_scores[doc_id] = max(doc_max_scores[doc_id], score)

    assert doc_max_scores["doc1"] == 0.8
    assert doc_max_scores["doc2"] == 0.7
    assert -1 not in doc_max_scores  # -1 indices must be skipped


# ---------------------------------------------------------------------------
# Test 8: Training MaxSim == Search MaxSim equivalence
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_training_vs_search_maxsim_equivalence():
    """
    Prove that training MaxSim (einsum→max) and search MaxSim
    (per-chunk dot product → max per doc) give identical results.

    Setup:
      q = [1, 2, 0, 0]
      p0 = doc with 3 chunks:
        c0 = [1, 0, 0, 0]   dot = 1*1 + 2*0 = 1.0
        c1 = [0, 1, 0, 0]   dot = 1*0 + 2*1 = 2.0  ← best
        c2 = [0, 0, 1, 0]   dot = 0.0
      p1 = doc with 2 chunks (padded to 3):
        c0 = [0.5, 0.5, 0, 0]  dot = 0.5 + 1.0 = 1.5  ← best
        c1 = [0, 0, 0, 1]      dot = 0.0
        c2 = PADDING

    Training MaxSim:
      maxsim(q, p0) = max(1.0, 2.0, 0.0) = 2.0
      maxsim(q, p1) = max(1.5, 0.0) = 1.5  (c2 masked)

    Search MaxSim:
      FAISS finds all 5 chunks, returns dot products
      Aggregate: doc0 → max(1.0, 2.0, 0.0) = 2.0
                 doc1 → max(1.5, 0.0) = 1.5
    """
    model = _make_model()

    q_reps = torch.tensor([[1.0, 2.0, 0.0, 0.0]])  # [1, 4]

    p_reps = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],      # p0_c0
         [0.0, 1.0, 0.0, 0.0],      # p0_c1
         [0.0, 0.0, 1.0, 0.0]],     # p0_c2
        [[0.5, 0.5, 0.0, 0.0],      # p1_c0
         [0.0, 0.0, 0.0, 1.0],      # p1_c1
         [0.0, 0.0, 0.0, 0.0]],     # p1_c2 (padding)
    ])  # [2, 3, 4]

    chunk_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    # --- Training MaxSim ---
    training_scores = model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
    assert training_scores.shape == (1, 2)

    # --- Search MaxSim (simulate per-chunk FAISS + aggregation) ---
    # Flatten chunks as they would be stored in FAISS (skip padding)
    p_lookup = []
    chunk_embeddings = []
    for doc_idx in range(2):
        for chunk_idx in range(3):
            if chunk_mask[doc_idx, chunk_idx] > 0:
                p_lookup.append((f"doc{doc_idx}", chunk_idx))
                chunk_embeddings.append(p_reps[doc_idx, chunk_idx])

    # Compute per-chunk dot products (simulates FAISS inner product)
    search_doc_scores = defaultdict(lambda: float('-inf'))
    for (doc_id, _), chunk_emb in zip(p_lookup, chunk_embeddings):
        dot = torch.dot(q_reps[0], chunk_emb).item()
        search_doc_scores[doc_id] = max(search_doc_scores[doc_id], dot)

    # Compare
    assert torch.allclose(training_scores[0, 0], torch.tensor(search_doc_scores["doc0"]))
    assert torch.allclose(training_scores[0, 1], torch.tensor(search_doc_scores["doc1"]))

    # Verify actual values
    assert torch.allclose(training_scores[0, 0], torch.tensor(2.0))  # p0_c1 wins
    assert torch.allclose(training_scores[0, 1], torch.tensor(1.5))  # p1_c0 wins


# ---------------------------------------------------------------------------
# Test 9: Variable chunk sizes (deterministic hash)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_variable_chunk_sizes_deterministic():
    """
    Verify that _chunk_tokens with chunk_size_range produces deterministic
    results for the same passage_seed, and different results for different seeds.
    """
    tokens = list(range(100))  # 100 tokens

    # Same seed → same chunking
    ids1, eos1 = _chunk_tokens(tokens, chunk_size=0, eos_token_id=99,
                                chunk_size_range=(4, 16), passage_seed=42)
    ids2, eos2 = _chunk_tokens(tokens, chunk_size=0, eos_token_id=99,
                                chunk_size_range=(4, 16), passage_seed=42)
    assert ids1 == ids2
    assert eos1 == eos2

    # Different seed → (very likely) different chunking
    ids3, eos3 = _chunk_tokens(tokens, chunk_size=0, eos_token_id=99,
                                chunk_size_range=(4, 16), passage_seed=123)
    # With 100 tokens and range 4-16, the probability of identical chunking
    # with different seeds is negligible
    assert eos1 != eos3, "Different seeds should produce different chunk boundaries"

    # Verify all EOS tokens are actually present
    for pos in eos1:
        assert ids1[pos] == 99


# ---------------------------------------------------------------------------
# Test 10: Normalization in _pooling_chunked
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pooling_chunked_with_normalization():
    """
    Verify that normalize=True produces unit-length chunk embeddings,
    and that zero-padded chunks remain zero (no NaN from F.normalize).
    """
    from tevatron.retriever.modeling.dense import DenseModel

    hidden = torch.zeros(1, 8, 4)
    hidden[0, 2] = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm = 5.0
    hidden[0, 5] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # zero vector (edge case)

    eos_positions = [[2, 5]]

    model = DenseModel(encoder=Mock(), pooling='last', normalize=True)
    model.passage_chunk_size = 1
    chunk_reps, chunk_mask = model._pooling_chunked(hidden, eos_positions)

    # Chunk 0: [3,4,0,0] normalized → [0.6, 0.8, 0, 0]
    assert torch.allclose(chunk_reps[0, 0], torch.tensor([0.6, 0.8, 0.0, 0.0]), atol=1e-6)
    # Verify unit length
    assert torch.allclose(chunk_reps[0, 0].norm(), torch.tensor(1.0), atol=1e-6)

    # Chunk 1: zero vector → stays zero (F.normalize with eps handles this)
    assert torch.allclose(chunk_reps[0, 1], torch.tensor([0.0, 0.0, 0.0, 0.0]), atol=1e-6)
    # No NaN
    assert not torch.isnan(chunk_reps).any()


# ---------------------------------------------------------------------------
# Test 11: Training MaxSim — trace which chunk is selected per query-passage
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_maxsim_chunk_selection_training():
    """
    Trace exactly which chunk index MaxSim selects for each (query, passage) pair.

    This replicates the internal logic of compute_maxsim_similarity:
        chunk_scores = einsum('qh,pch->qpc')   → [Q, P, C]
        masked_fill(padding, -inf)
        max(dim=-1)                             → max_vals [Q,P], max_idx [Q,P]

    Scenario (3 passages, each about a different topic):
      Passage 0 ("sports"): 3 chunks
        c0 = [1,  0,  0]   "football"
        c1 = [0,  1,  0]   "basketball"
        c2 = [0.8, 0.6, 0] "general sports"

      Passage 1 ("science"): 2 chunks + 1 padding
        c0 = [0, 0, 1]     "physics"
        c1 = [0, 0.3, 0.9] "chemistry"
        c2 = PADDING

      Passage 2 ("mixed"): 1 chunk + 2 padding
        c0 = [0.5, 0.5, 0.5]  "general knowledge"
        c1 = PADDING
        c2 = PADDING

    Queries:
      q0 = [1, 0, 0]  → "football fan"
        vs p0: dot = [1.0, 0.0, 0.8]  → max = 1.0 at c0 ✓
        vs p1: dot = [0.0, 0.0, -inf]  → max = 0.0 at c0
        vs p2: dot = [0.5, -inf, -inf] → max = 0.5 at c0

      q1 = [0, 0, 1]  → "science lover"
        vs p0: dot = [0.0, 0.0, 0.0]   → max = 0.0 at c0 (tie, picks first)
        vs p1: dot = [1.0, 0.9, -inf]  → max = 1.0 at c0 ✓
        vs p2: dot = [0.5, -inf, -inf] → max = 0.5 at c0

      q2 = [0, 1, 0]  → "basketball fan"
        vs p0: dot = [0.0, 1.0, 0.6]   → max = 1.0 at c1 ✓
        vs p1: dot = [0.0, 0.3, -inf]  → max = 0.3 at c1
        vs p2: dot = [0.5, -inf, -inf] → max = 0.5 at c0
    """
    model = _make_model()

    H = 3
    q_reps = torch.tensor([
        [1.0, 0.0, 0.0],   # q0: football
        [0.0, 0.0, 1.0],   # q1: science
        [0.0, 1.0, 0.0],   # q2: basketball
    ])  # [3, 3]

    p_reps = torch.tensor([
        [[1.0, 0.0, 0.0],      # p0_c0: football
         [0.0, 1.0, 0.0],      # p0_c1: basketball
         [0.8, 0.6, 0.0]],     # p0_c2: general sports
        [[0.0, 0.0, 1.0],      # p1_c0: physics
         [0.0, 0.3, 0.9],      # p1_c1: chemistry
         [0.0, 0.0, 0.0]],     # p1_c2: PADDING (will be masked)
        [[0.5, 0.5, 0.5],      # p2_c0: general knowledge
         [0.0, 0.0, 0.0],      # p2_c1: PADDING
         [0.0, 0.0, 0.0]],     # p2_c2: PADDING
    ])  # [3, 3, 3]

    chunk_mask = torch.tensor([
        [1.0, 1.0, 1.0],   # p0: all 3 valid
        [1.0, 1.0, 0.0],   # p1: 2 valid
        [1.0, 0.0, 0.0],   # p2: 1 valid
    ])  # [3, 3]

    # --- Run MaxSim ---
    scores = model.compute_maxsim_similarity(q_reps, p_reps, chunk_mask)
    assert scores.shape == (3, 3)  # [Q=3, P=3]

    # --- Manually replicate einsum + mask + max to trace chunk selection ---
    chunk_scores = torch.einsum('qh,pch->qpc', q_reps, p_reps)  # [3, 3, 3]
    padding_mask = ~chunk_mask.unsqueeze(0).bool()  # [1, 3, 3]
    chunk_scores_masked = chunk_scores.masked_fill(padding_mask, float('-inf'))
    max_vals, max_idx = chunk_scores_masked.max(dim=-1)  # [3, 3]

    # --- Verify scores match ---
    assert torch.allclose(scores, max_vals)

    # --- Trace which chunk was selected for each (query, passage) pair ---

    # q0 ("football") vs p0 ("sports"): chunk 0 wins (football=1.0 > basketball=0.0 > general=0.8)
    assert max_idx[0, 0].item() == 0
    assert torch.allclose(scores[0, 0], torch.tensor(1.0))

    # q0 vs p1 ("science"): chunk 0 wins (physics dot [1,0,0] = 0.0, chemistry dot [1,0,0] = 0.0, tie → c0)
    assert max_idx[0, 1].item() == 0
    assert torch.allclose(scores[0, 1], torch.tensor(0.0))

    # q0 vs p2 ("mixed"): chunk 0 wins (only valid chunk, dot = 0.5)
    assert max_idx[0, 2].item() == 0
    assert torch.allclose(scores[0, 2], torch.tensor(0.5))

    # q1 ("science") vs p1 ("science"): chunk 0 wins (physics=1.0 > chemistry=0.9)
    assert max_idx[1, 1].item() == 0
    assert torch.allclose(scores[1, 1], torch.tensor(1.0))

    # q2 ("basketball") vs p0 ("sports"): chunk 1 wins (basketball=1.0 > general=0.6 > football=0.0)
    assert max_idx[2, 0].item() == 1
    assert torch.allclose(scores[2, 0], torch.tensor(1.0))

    # q2 vs p1: chunk 1 wins (chemistry has 0.3 basketball component > physics has 0.0)
    assert max_idx[2, 1].item() == 1
    assert torch.allclose(scores[2, 1], torch.tensor(0.3))

    # --- Verify ranking: which passage does each query prefer? ---
    # q0 (football): p0=1.0 > p2=0.5 > p1=0.0
    q0_ranking = scores[0].argsort(descending=True).tolist()
    assert q0_ranking == [0, 2, 1]

    # q1 (science): p1=1.0 > p2=0.5 > p0=0.0
    q1_ranking = scores[1].argsort(descending=True).tolist()
    assert q1_ranking == [1, 2, 0]

    # q2 (basketball): p0=1.0 > p2=0.5 > p1=0.3
    q2_ranking = scores[2].argsort(descending=True).tolist()
    assert q2_ranking == [0, 2, 1]


# ---------------------------------------------------------------------------
# Test 12: Search MaxSim — trace which chunk is selected per document
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_maxsim_chunk_selection_search():
    """
    Trace exactly which chunk is selected during search aggregation,
    mirroring the training test above.

    Same 3 passages stored as individual chunk embeddings in FAISS:
      idx 0: ("p0", 0) = [1, 0, 0]     football
      idx 1: ("p0", 1) = [0, 1, 0]     basketball
      idx 2: ("p0", 2) = [0.8, 0.6, 0] general sports
      idx 3: ("p1", 0) = [0, 0, 1]     physics
      idx 4: ("p1", 1) = [0, 0.3, 0.9] chemistry
      idx 5: ("p2", 0) = [0.5, 0.5, 0.5] general knowledge

    Query q0 = [1, 0, 0] ("football"):
      FAISS returns dot products for all 6 chunks:
        idx 0 → 1.0, idx 1 → 0.0, idx 2 → 0.8, idx 3 → 0.0, idx 4 → 0.0, idx 5 → 0.5
      Aggregation (max per doc):
        p0: max(1.0, 0.0, 0.8) = 1.0   (chunk 0 "football" selected)
        p1: max(0.0, 0.0) = 0.0
        p2: max(0.5) = 0.5
      Ranking: p0 (1.0) > p2 (0.5) > p1 (0.0)
    """
    import numpy as np
    from tevatron.retriever.searcher import FaissFlatSearcher

    p_lookup = [
        ("p0", 0), ("p0", 1), ("p0", 2),  # sports doc: 3 chunks
        ("p1", 0), ("p1", 1),              # science doc: 2 chunks
        ("p2", 0),                          # mixed doc: 1 chunk
    ]

    # Build FAISS index with chunk embeddings (must be float32 numpy)
    chunk_embeddings = np.array([
        [1.0, 0.0, 0.0],       # p0_c0: football
        [0.0, 1.0, 0.0],       # p0_c1: basketball
        [0.8, 0.6, 0.0],       # p0_c2: general sports
        [0.0, 0.0, 1.0],       # p1_c0: physics
        [0.0, 0.3, 0.9],       # p1_c1: chemistry
        [0.5, 0.5, 0.5],       # p2_c0: general knowledge
    ], dtype=np.float32)

    retriever = FaissFlatSearcher(chunk_embeddings)
    retriever.add(chunk_embeddings)

    # 3 queries
    q_reps = np.array([
        [1.0, 0.0, 0.0],   # q0: football
        [0.0, 0.0, 1.0],   # q1: science
        [0.0, 1.0, 0.0],   # q2: basketball
    ], dtype=np.float32)

    # Search all 6 chunks (depth=3, multiplier=2 → search 6 chunks)
    search_depth = 6
    all_scores, all_indices = retriever.batch_search(q_reps, search_depth, batch_size=3)

    # --- Aggregate per document, tracking which chunk won ---
    # expected_winners only checks chunks where one chunk has a strictly higher score
    for q_idx, (q_label, expected_ranking, expected_scores, expected_winners) in enumerate([
        # (query_name, doc ranking, {doc: expected_score}, {doc: winning_chunk_idx})
        ("football", ["p0", "p2", "p1"],
         {"p0": 1.0, "p2": 0.5, "p1": 0.0},
         {"p0": 0}),  # p0_c0 "football" clearly wins (1.0 > 0.8 > 0.0)
        ("science", ["p1", "p2", "p0"],
         {"p1": 1.0, "p2": 0.5, "p0": 0.0},
         {"p1": 0}),  # p1_c0 "physics" clearly wins (1.0 > 0.9)
        ("basketball", ["p0", "p2", "p1"],
         {"p0": 1.0, "p2": 0.5, "p1": 0.3},
         {"p0": 1, "p1": 1}),  # p0_c1 "basketball" (1.0), p1_c1 "chemistry" (0.3)
    ]):
        scores = all_scores[q_idx]
        indices = all_indices[q_idx]

        doc_max_scores = defaultdict(lambda: float('-inf'))
        doc_best_chunk = {}  # Track which chunk index was selected

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(p_lookup):
                continue
            doc_id, chunk_idx = p_lookup[int(idx)]
            score = float(score)
            if score > doc_max_scores[doc_id]:
                doc_max_scores[doc_id] = score
                doc_best_chunk[doc_id] = chunk_idx

        sorted_docs = sorted(doc_max_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = [doc_id for doc_id, _ in sorted_docs]

        # Verify ranking
        assert ranking == expected_ranking, (
            f"q{q_idx} ({q_label}): expected ranking {expected_ranking}, got {ranking}"
        )

        # Verify MaxSim scores per document
        for doc_id, exp_score in expected_scores.items():
            assert abs(doc_max_scores[doc_id] - exp_score) < 1e-5, (
                f"q{q_idx} ({q_label}), {doc_id}: expected score {exp_score}, "
                f"got {doc_max_scores[doc_id]}"
            )

        # Verify which chunk was selected (only for unambiguous cases)
        for doc_id, expected_chunk in expected_winners.items():
            assert doc_best_chunk[doc_id] == expected_chunk, (
                f"q{q_idx} ({q_label}), {doc_id}: expected chunk {expected_chunk} "
                f"selected, got chunk {doc_best_chunk[doc_id]}"
            )

    # --- Verify training and search agree ---
    # Compute training MaxSim for comparison
    model = _make_model()
    q_torch = torch.tensor(q_reps)
    p_torch = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.3, 0.9], [0.0, 0.0, 0.0]],  # pad c2
        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # pad c1,c2
    ])
    mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    training_scores = model.compute_maxsim_similarity(q_torch, p_torch, mask)

    # Compare: for each query, the ranking from training MaxSim must match search MaxSim
    for q_idx in range(3):
        training_ranking = training_scores[q_idx].argsort(descending=True).tolist()
        # Map passage indices to doc names
        doc_names = ["p0", "p1", "p2"]
        training_ranking_names = [doc_names[i] for i in training_ranking]

        sorted_docs = sorted(
            {doc: float('-inf') for doc in doc_names}.items(),
            key=lambda x: x[1]
        )
        # Recompute search ranking for this query
        search_scores_q = defaultdict(lambda: float('-inf'))
        for score, idx in zip(all_scores[q_idx], all_indices[q_idx]):
            if idx < 0 or idx >= len(p_lookup):
                continue
            doc_id, _ = p_lookup[idx]
            search_scores_q[doc_id] = max(search_scores_q[doc_id], score)
        search_ranking_names = [d for d, _ in sorted(search_scores_q.items(), key=lambda x: x[1], reverse=True)]

        assert training_ranking_names == search_ranking_names, (
            f"q{q_idx}: training ranking {training_ranking_names} != "
            f"search ranking {search_ranking_names}"
        )
