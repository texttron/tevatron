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
    """Test _pooling_chunked extracts embeddings from correct EOS positions."""
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.modeling.dense import DenseModel
    
    mock_encoder = Mock()
    mock_encoder.config.hidden_size = 8
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = 32
    
    batch_size, seq_len, hidden_size = 2, 10, 8
    hidden_states = torch.zeros(batch_size, seq_len, hidden_size)
    for i in range(batch_size):
        for j in range(seq_len):
            hidden_states[i, j, 0] = j * 100 + i * 10
            for k in range(1, hidden_size):
                hidden_states[i, j, k] = j * 10 + k
    
    eos_positions = [[2, 5, 8], [3, 7]]
    chunk_reps, chunk_mask = model._pooling_chunked(hidden_states, eos_positions)
    
    assert chunk_reps.shape == (batch_size, 3, hidden_size)
    assert chunk_mask.shape == (batch_size, 3)
    
    # Verify correct positions extracted
    assert torch.allclose(chunk_reps[0, 0, 0], torch.tensor(200.0))  # pos 2
    assert torch.allclose(chunk_reps[0, 1, 0], torch.tensor(500.0))  # pos 5
    assert torch.allclose(chunk_reps[0, 2, 0], torch.tensor(800.0))  # pos 8
    assert torch.allclose(chunk_reps[1, 0, 0], torch.tensor(310.0))  # pos 3
    assert torch.allclose(chunk_reps[1, 1, 0], torch.tensor(710.0))  # pos 7
    
    # Verify chunk mask
    assert (chunk_mask[0, :3] == 1.0).all()
    assert (chunk_mask[1, :2] == 1.0).all()
    assert chunk_mask[1, 2] == 0.0
    
    # Test exact equality with sequential hidden states
    hidden_states_2 = torch.arange(batch_size * seq_len * hidden_size, dtype=torch.float32)
    hidden_states_2 = hidden_states_2.reshape(batch_size, seq_len, hidden_size)
    chunk_reps_2, _ = model._pooling_chunked(hidden_states_2, eos_positions)
    
    assert torch.equal(chunk_reps_2[0, 0], hidden_states_2[0, 2])
    assert torch.equal(chunk_reps_2[0, 1], hidden_states_2[0, 5])
    assert torch.equal(chunk_reps_2[0, 2], hidden_states_2[0, 8])
    assert torch.equal(chunk_reps_2[1, 0], hidden_states_2[1, 3])
    assert torch.equal(chunk_reps_2[1, 1], hidden_states_2[1, 7])
    
    # Test edge cases
    chunk_reps_empty, chunk_mask_empty = model._pooling_chunked(hidden_states, [])
    assert chunk_reps_empty.shape == (batch_size, 0, hidden_size)
    
    eos_positions_oob = [[2, 5, 15], [3, 7]]
    chunk_reps_oob, chunk_mask_oob = model._pooling_chunked(hidden_states, eos_positions_oob)
    assert torch.allclose(chunk_reps_oob[0, 2], torch.zeros(hidden_size))
    assert chunk_mask_oob[0, 2] == 0.0
    
    # Test normalization
    model.normalize = True
    chunk_reps_norm, _ = model._pooling_chunked(hidden_states_2, eos_positions)
    for i in range(batch_size):
        for j in range(len(eos_positions[i])):
            assert torch.allclose(torch.norm(chunk_reps_norm[i, j]), torch.tensor(1.0), atol=1e-6)


@pytest.mark.unit
def test_pooling_chunked_real_tokenizer_alignment(train_tokenizer):
    """Integration test: eos_positions from collator correctly align with hidden states."""
    import torch
    from unittest.mock import Mock
    from tevatron.retriever.arguments import DataArguments
    from tevatron.retriever.collator import ChunkedEncodeCollator
    from tevatron.retriever.modeling.dense import DenseModel
    
    data_args = DataArguments(
        passage_chunk_size=32,
        passage_max_len=128,
        pad_to_multiple_of=16,
        padding_side="right",
        append_eos_token=False,
    )
    collator = ChunkedEncodeCollator(data_args=data_args, tokenizer=train_tokenizer)
    passages = [REAL_TEXT, "Short passage for testing."]
    d_collated, eos_positions = collator._tokenize_and_pad_chunked_passages(passages)
    
    input_ids = d_collated['input_ids']
    seq_len = input_ids.shape[1]
    
    # Verify eos_positions are valid
    for i, eos_pos_list in enumerate(eos_positions):
        assert len(eos_pos_list) > 0
        for pos in eos_pos_list:
            assert 0 <= pos < seq_len
            assert input_ids[i, pos] == train_tokenizer.eos_token_id
    
    # Create mock encoder
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
                hidden_states[i, j, 1] = float(input_ids[i, j])
                for k in range(2, hidden_size):
                    hidden_states[i, j, k] = float(j * hidden_size + k)
        return MockEncoderOutput(last_hidden_state=hidden_states)
    
    mock_encoder = Mock(side_effect=mock_encoder_forward)
    mock_encoder.config = Mock()
    mock_encoder.config.hidden_size = hidden_size
    
    model = DenseModel(encoder=mock_encoder, pooling='last', normalize=False)
    model.passage_chunk_size = data_args.passage_chunk_size
    
    batch_inputs = {
        'input_ids': d_collated['input_ids'],
        'attention_mask': d_collated['attention_mask'],
    }
    chunk_reps, chunk_mask = model.encode_passage(batch_inputs, eos_positions)
    
    batch_size = len(passages)
    max_chunks = max(len(pos_list) for pos_list in eos_positions)
    assert chunk_reps.shape == (batch_size, max_chunks, hidden_size)
    
    # Re-create expected hidden states
    hidden_states_expected = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            hidden_states_expected[i, j, 0] = float(j)
            hidden_states_expected[i, j, 1] = float(input_ids[i, j])
            for k in range(2, hidden_size):
                hidden_states_expected[i, j, k] = float(j * hidden_size + k)
    
    # Verify extracted embeddings match expected positions
    for i, eos_pos_list in enumerate(eos_positions):
        for j, pos in enumerate(eos_pos_list):
            assert torch.equal(chunk_reps[i, j], hidden_states_expected[i, pos])
            assert chunk_mask[i, j] == 1.0
            assert torch.allclose(chunk_reps[i, j, 0], torch.tensor(float(pos)))
    
    # Verify invalid chunks are masked
    for i in range(batch_size):
        num_chunks = len(eos_positions[i])
        for j in range(num_chunks, max_chunks):
            assert chunk_mask[i, j] == 0.0
