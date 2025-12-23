import sys
from pathlib import Path

import pytest


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    # tevatron/tests/test_pooling.py -> tevatron/ -> tevatron/src
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


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
EOS_TOKEN_ID = 151645
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
    tok.padding_side = "right"  # finetune_with_chunk.sh uses --padding_side right
    return tok


@pytest.mark.unit
def test_encode_with_chunking(train_tokenizer, tmp_path):
    """
    Test the full encode functionality with chunking enabled.
    This tests the integration of:
    - EncodeDataset loading JSONL data
    - ChunkedEncodeCollator creating batches with eos_positions
    - DenseModel.encode_passage with chunking
    - Output shape and lookup_indices creation
    """
    import json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from unittest.mock import Mock
    
    from tevatron.retriever.arguments import DataArguments, TevatronTrainingArguments as TrainingArguments
    from tevatron.retriever.dataset import EncodeDataset
    from tevatron.retriever.collator import ChunkedEncodeCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    # Create temporary JSONL file with test passages
    test_passages = [
        {"docid": "doc1", "text": REAL_TEXT},  # Long passage that will be chunked
        {"docid": "doc2", "text": "Short passage."},  # Short passage
    ]
    
    jsonl_file = tmp_path / "test_corpus.jsonl"
    with open(jsonl_file, 'w') as f:
        for passage in test_passages:
            f.write(json.dumps(passage) + '\n')
    
    # Setup data arguments for chunked encoding
    data_args = DataArguments(
        dataset_name='json',
        dataset_path=str(jsonl_file),
        dataset_split='train',
        passage_chunk_size=32,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        encode_is_query=False,
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(tmp_path / "output"),
        per_device_eval_batch_size=2,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
    )
    
    # Create dataset
    encode_dataset = EncodeDataset(data_args=data_args)
    assert len(encode_dataset) == 2
    
    # Create chunked collator
    encode_collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
    
    # Create data loader
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    
    # Create a mock encoder model
    hidden_size = 64
    
    # Create a proper mock that returns an object with last_hidden_state
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    # Mock the encoder forward pass to return hidden states
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        # Create dummy hidden states with positional encoding for testing
        hidden_states = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float32)
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)
        # Add some variation based on input_ids for testing
        hidden_states = hidden_states + input_ids.unsqueeze(-1).float() * 0.01
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    # Create DenseModel with mock encoder
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = data_args.passage_chunk_size
    model.eval()
    
    # Simulate the encode loop
    encoded = []
    lookup_indices = []
    
    for batch in encode_loader:
        doc_ids, batch_inputs, eos_positions = batch
        
        # Verify batch structure
        assert isinstance(doc_ids, list)
        # batch_inputs is a BatchEncoding (from tokenizer.pad), which behaves like a dict
        assert hasattr(batch_inputs, '__getitem__')  # Check if it's dict-like
        assert 'input_ids' in batch_inputs
        assert 'attention_mask' in batch_inputs
        assert isinstance(eos_positions, list)
        assert len(eos_positions) == len(doc_ids)
        
        # Verify eos_positions structure
        for i, eos_pos_list in enumerate(eos_positions):
            assert isinstance(eos_pos_list, list)
            assert len(eos_pos_list) > 0  # Should have at least one chunk
            # Verify eos_positions are within sequence length
            seq_len = batch_inputs['input_ids'].shape[1]
            for pos in eos_pos_list:
                assert 0 <= pos < seq_len
        
        # Encode with chunking
        with torch.no_grad():
            chunk_embs, chunk_mask = model.encode_passage(batch_inputs, eos_positions)
            
            # Verify output shapes
            batch_size, max_chunks, hidden_size_out = chunk_embs.shape
            assert batch_size == len(doc_ids)
            assert hidden_size_out == hidden_size
            assert chunk_mask.shape == (batch_size, max_chunks)
            
            # Verify chunk_mask values (should be 0 or 1)
            assert torch.all((chunk_mask == 0) | (chunk_mask == 1))
            
            # Process chunks and create lookup indices
            for i, doc_id in enumerate(doc_ids):
                for chunk_idx in range(max_chunks):
                    if chunk_mask[i, chunk_idx] > 0:  # Valid chunk
                        encoded.append(chunk_embs[i, chunk_idx].cpu().detach().numpy())
                        lookup_indices.append((doc_id, chunk_idx))
    
    # Verify results
    assert len(encoded) > 0
    assert len(lookup_indices) == len(encoded)
    
    # Stack encoded embeddings
    encoded_array = np.stack(encoded)
    assert encoded_array.shape[0] == len(encoded)
    assert encoded_array.shape[1] == hidden_size
    
    # Verify lookup_indices structure
    unique_docs = set(doc_id for doc_id, _ in lookup_indices)
    assert len(unique_docs) == 2  # Should have both doc1 and doc2
    
    # Verify doc1 has multiple chunks (it's a long passage)
    doc1_chunks = [chunk_idx for doc_id, chunk_idx in lookup_indices if doc_id == "doc1"]
    assert len(doc1_chunks) > 1  # Should have multiple chunks
    
    # Verify doc2 has at least one chunk
    doc2_chunks = [chunk_idx for doc_id, chunk_idx in lookup_indices if doc_id == "doc2"]
    assert len(doc2_chunks) >= 1
    
    # Verify chunk indices are sequential starting from 0
    for doc_id in unique_docs:
        doc_chunks = sorted([chunk_idx for d, chunk_idx in lookup_indices if d == doc_id])
        assert doc_chunks == list(range(len(doc_chunks)))  # Should be 0, 1, 2, ...
    
    # Verify embeddings are not all zeros (they should have been computed)
    assert not np.allclose(encoded_array, 0)
    
    # Verify embeddings have reasonable values (not NaN or Inf)
    assert np.all(np.isfinite(encoded_array))


@pytest.mark.unit
def test_pooling_chunked_eos_positions_alignment():
    """
    Test _pooling_chunked to verify that eos_positions correctly align with hidden states.
    This test uses known hidden states and eos_positions to verify exact alignment.
    """
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.modeling.dense import DenseModel
    
    # Create a mock encoder
    mock_encoder = Mock()
    mock_encoder.config.hidden_size = 8
    
    # Create DenseModel
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = 32
    
    # Test Case 1: Simple case with known positions
    # Batch size=2, seq_len=10, hidden_size=8
    # Passage 0: eos at positions [2, 5, 8] (3 chunks)
    # Passage 1: eos at positions [3, 7] (2 chunks)
    batch_size = 2
    seq_len = 10
    hidden_size = 8
    
    # Create hidden states with known values - each position has a unique pattern
    # We'll use position index as part of the embedding to make verification easy
    hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
    for i in range(batch_size):
        for j in range(seq_len):
            # Set embedding at position j to have value j*100 + i*10 in first dimension
            # This makes it easy to verify we're extracting the right positions
            hidden_states[i, j, 0] = j * 100 + i * 10
            # Fill other dimensions with position-dependent values
            for k in range(1, hidden_size):
                hidden_states[i, j, k] = j * 10 + k
    
    eos_positions = [[2, 5, 8], [3, 7]]
    
    # Call _pooling_chunked
    chunk_reps, chunk_mask = model._pooling_chunked(hidden_states, eos_positions)
    
    # Verify output shapes
    assert chunk_reps.shape == (batch_size, 3, hidden_size)  # max_chunks = 3
    assert chunk_mask.shape == (batch_size, 3)
    
    # Verify Passage 0: should extract positions [2, 5, 8]
    # Position 2: should have 2*100 + 0*10 = 200 in first dim
    assert torch.allclose(chunk_reps[0, 0, 0], torch.tensor(200.0))
    assert torch.allclose(chunk_reps[0, 0, 1], torch.tensor(21.0))  # 2*10 + 1
    
    # Position 5: should have 5*100 + 0*10 = 500 in first dim
    assert torch.allclose(chunk_reps[0, 1, 0], torch.tensor(500.0))
    assert torch.allclose(chunk_reps[0, 1, 1], torch.tensor(51.0))  # 5*10 + 1
    
    # Position 8: should have 8*100 + 0*10 = 800 in first dim
    assert torch.allclose(chunk_reps[0, 2, 0], torch.tensor(800.0))
    assert torch.allclose(chunk_reps[0, 2, 1], torch.tensor(81.0))  # 8*10 + 1
    
    # Verify Passage 1: should extract positions [3, 7]
    # Position 3: should have 3*100 + 1*10 = 310 in first dim
    assert torch.allclose(chunk_reps[1, 0, 0], torch.tensor(310.0))
    assert torch.allclose(chunk_reps[1, 0, 1], torch.tensor(31.0))  # 3*10 + 1
    
    # Position 7: should have 7*100 + 1*10 = 710 in first dim
    assert torch.allclose(chunk_reps[1, 1, 0], torch.tensor(710.0))
    assert torch.allclose(chunk_reps[1, 1, 1], torch.tensor(71.0))  # 7*10 + 1
    
    # Verify chunk_mask
    assert chunk_mask[0, 0] == 1.0  # Passage 0, chunk 0 (pos 2)
    assert chunk_mask[0, 1] == 1.0  # Passage 0, chunk 1 (pos 5)
    assert chunk_mask[0, 2] == 1.0  # Passage 0, chunk 2 (pos 8)
    assert chunk_mask[1, 0] == 1.0  # Passage 1, chunk 0 (pos 3)
    assert chunk_mask[1, 1] == 1.0  # Passage 1, chunk 1 (pos 7)
    assert chunk_mask[1, 2] == 0.0  # Passage 1, chunk 2 (no chunk, should be 0)
    
    # Test Case 2: Verify exact tensor equality (not just close)
    # Create hidden states where each position has a unique embedding
    hidden_states_2 = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float32)
    hidden_states_2 = hidden_states_2.reshape(batch_size, seq_len, hidden_size)
    
    # Extract embeddings manually for comparison
    expected_chunk_0_0 = hidden_states_2[0, 2]  # Passage 0, position 2
    expected_chunk_0_1 = hidden_states_2[0, 5]  # Passage 0, position 5
    expected_chunk_0_2 = hidden_states_2[0, 8]  # Passage 0, position 8
    expected_chunk_1_0 = hidden_states_2[1, 3]  # Passage 1, position 3
    expected_chunk_1_1 = hidden_states_2[1, 7]  # Passage 1, position 7
    
    chunk_reps_2, chunk_mask_2 = model._pooling_chunked(hidden_states_2, eos_positions)
    
    # Verify exact equality
    assert torch.equal(chunk_reps_2[0, 0], expected_chunk_0_0)
    assert torch.equal(chunk_reps_2[0, 1], expected_chunk_0_1)
    assert torch.equal(chunk_reps_2[0, 2], expected_chunk_0_2)
    assert torch.equal(chunk_reps_2[1, 0], expected_chunk_1_0)
    assert torch.equal(chunk_reps_2[1, 1], expected_chunk_1_1)
    
    # Test Case 3: Edge case - empty eos_positions
    chunk_reps_empty, chunk_mask_empty = model._pooling_chunked(hidden_states, [])
    assert chunk_reps_empty.shape == (batch_size, 0, hidden_size)
    assert chunk_mask_empty.shape == (batch_size, 0)
    
    # Test Case 4: Edge case - out of bounds position (should be handled gracefully)
    eos_positions_oob = [[2, 5, 15], [3, 7]]  # 15 is out of bounds for seq_len=10
    chunk_reps_oob, chunk_mask_oob = model._pooling_chunked(hidden_states, eos_positions_oob)
    
    # Should still extract valid positions
    assert chunk_reps_oob.shape == (batch_size, 3, hidden_size)
    assert torch.allclose(chunk_reps_oob[0, 0], hidden_states[0, 2])  # Valid
    assert torch.allclose(chunk_reps_oob[0, 1], hidden_states[0, 5])  # Valid
    # Position 15 is out of bounds, so chunk_reps[0, 2] should be zeros
    assert torch.allclose(chunk_reps_oob[0, 2], torch.zeros(hidden_size))
    assert chunk_mask_oob[0, 2] == 0.0  # Should be masked out
    
    # Test Case 5: Normalize=True
    model.normalize = True
    chunk_reps_norm, chunk_mask_norm = model._pooling_chunked(hidden_states_2, eos_positions)
    
    # Verify normalization (L2 norm should be 1 for non-zero chunks)
    for i in range(batch_size):
        for j in range(len(eos_positions[i])):
            norm = torch.norm(chunk_reps_norm[i, j])
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
    
    # Verify the normalized embeddings are proportional to original
    model.normalize = False
    chunk_reps_no_norm, _ = model._pooling_chunked(hidden_states_2, eos_positions)
    for i in range(batch_size):
        for j in range(len(eos_positions[i])):
            # Normalized version should be original / norm
            expected_norm = torch.norm(chunk_reps_no_norm[i, j])
            normalized_manual = chunk_reps_no_norm[i, j] / expected_norm
            assert torch.allclose(chunk_reps_norm[i, j], normalized_manual, atol=1e-6)
    
    # Test Case 6: Single chunk per passage
    eos_positions_single = [[4], [6]]
    chunk_reps_single, chunk_mask_single = model._pooling_chunked(hidden_states_2, eos_positions_single)
    
    assert chunk_reps_single.shape == (batch_size, 1, hidden_size)
    assert torch.equal(chunk_reps_single[0, 0], hidden_states_2[0, 4])
    assert torch.equal(chunk_reps_single[1, 0], hidden_states_2[1, 6])
    assert chunk_mask_single[0, 0] == 1.0
    assert chunk_mask_single[1, 0] == 1.0
    
    # Test Case 7: Verify positions are extracted in correct order
    # Use sequential positions to verify order
    eos_positions_ordered = [[1, 3, 5], [2, 4]]
    chunk_reps_ordered, _ = model._pooling_chunked(hidden_states_2, eos_positions_ordered)
    
    # Passage 0: should be in order [1, 3, 5]
    assert torch.equal(chunk_reps_ordered[0, 0], hidden_states_2[0, 1])
    assert torch.equal(chunk_reps_ordered[0, 1], hidden_states_2[0, 3])
    assert torch.equal(chunk_reps_ordered[0, 2], hidden_states_2[0, 5])
    
    # Passage 1: should be in order [2, 4]
    assert torch.equal(chunk_reps_ordered[1, 0], hidden_states_2[1, 2])
    assert torch.equal(chunk_reps_ordered[1, 1], hidden_states_2[1, 4])


@pytest.mark.unit
def test_pooling_chunked_real_tokenizer_alignment(train_tokenizer):
    """
    Integration test: Verify that eos_positions from ChunkedEncodeCollator
    correctly align with hidden states when using _pooling_chunked.
    This uses real tokenizer to ensure end-to-end correctness.
    """
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import ChunkedEncodeCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    # Setup data arguments
    data_args = DataArguments(
        passage_chunk_size=32,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        passage_prefix="",
        append_eos_token=False,
    )
    
    # Create collator
    collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
    
    # Test passages
    passages = [
        REAL_TEXT,  # Long passage that will be chunked
        "Short passage for testing.",  # Short passage
    ]
    
    # Get tokenized and chunked data
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages(passages)
    
    # Verify eos_positions are valid
    input_ids = d_collated['input_ids']
    seq_len = input_ids.shape[1]
    
    for i, eos_pos_list in enumerate(eos_positions):
        assert len(eos_pos_list) > 0, f"Passage {i} should have at least one chunk"
        for pos in eos_pos_list:
            assert 0 <= pos < seq_len, f"EOS position {pos} out of bounds for seq_len {seq_len}"
            # Verify that the position actually contains EOS token
            assert input_ids[i, pos] == train_tokenizer.eos_token_id, \
                f"Position {pos} should contain EOS token {train_tokenizer.eos_token_id}, got {input_ids[i, pos]}"
    
    # Create mock encoder that returns hidden states based on input_ids
    # This allows us to verify exact alignment
    hidden_size = 64
    
    class MockEncoderOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state
    
    def mock_encoder_forward(**kwargs):
        input_ids = kwargs['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # Create hidden states where each position's embedding encodes its position
        # This makes it easy to verify we're extracting the right positions
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(seq_len):
                # Encode position j in the embedding
                # Use input_ids[i, j] as seed to make it unique per token
                hidden_states[i, j, 0] = float(j)  # Position index
                hidden_states[i, j, 1] = float(input_ids[i, j])  # Token ID
                # Fill rest with position-dependent values
                for k in range(2, hidden_size):
                    hidden_states[i, j, k] = float(j * hidden_size + k)
        
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    # Create model
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = data_args.passage_chunk_size
    
    # Convert BatchEncoding to dict for model
    batch_inputs = {
        'input_ids': d_collated['input_ids'],
        'attention_mask': d_collated['attention_mask'],
    }
    
    # Encode with chunking
    chunk_reps, chunk_mask = model.encode_passage(batch_inputs, eos_positions)
    
    # Verify shapes
    batch_size = len(passages)
    max_chunks = max(len(pos_list) for pos_list in eos_positions)
    assert chunk_reps.shape == (batch_size, max_chunks, hidden_size)
    assert chunk_mask.shape == (batch_size, max_chunks)
    
    # Verify that extracted embeddings match the eos_positions
    # We need to get the hidden states that were generated
    # Since we can't easily access them, we'll verify by checking the mock was called correctly
    # and that the extracted positions match what we expect
    
    # Re-create hidden states with the same logic to verify
    hidden_states_expected = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            hidden_states_expected[i, j, 0] = float(j)
            hidden_states_expected[i, j, 1] = float(input_ids[i, j])
            for k in range(2, hidden_size):
                hidden_states_expected[i, j, k] = float(j * hidden_size + k)
    
    # Verify each extracted chunk embedding matches the expected position
    for i, eos_pos_list in enumerate(eos_positions):
        for j, pos in enumerate(eos_pos_list):
            # The extracted embedding should match the hidden state at position pos
            expected_embedding = hidden_states_expected[i, pos]
            extracted_embedding = chunk_reps[i, j]
            
            # Verify exact match (they should be identical)
            assert torch.equal(extracted_embedding, expected_embedding), \
                f"Passage {i}, chunk {j} (eos_pos={pos}): extracted embedding doesn't match hidden state at position {pos}"
            
            # Verify chunk mask is set correctly
            assert chunk_mask[i, j] == 1.0, f"Chunk mask should be 1.0 for valid chunk"
    
    # Verify that invalid chunks (beyond actual chunks) have mask=0
    for i in range(batch_size):
        num_chunks = len(eos_positions[i])
        for j in range(num_chunks, max_chunks):
            assert chunk_mask[i, j] == 0.0, f"Invalid chunk should have mask=0"
    
    # Verify that the first dimension of extracted embeddings contains position indices
    for i, eos_pos_list in enumerate(eos_positions):
        for j, pos in enumerate(eos_pos_list):
            # First dimension should equal the position
            assert torch.allclose(chunk_reps[i, j, 0], torch.tensor(float(pos))), \
                f"First dim should equal position {pos}, got {chunk_reps[i, j, 0]}"
            
            # Second dimension should equal the token ID at that position
            expected_token_id = float(input_ids[i, pos])
            assert torch.allclose(chunk_reps[i, j, 1], torch.tensor(expected_token_id)), \
                f"Second dim should equal token ID {expected_token_id}, got {chunk_reps[i, j, 1]}"
