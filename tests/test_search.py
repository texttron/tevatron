import sys
from pathlib import Path
import pickle
import numpy as np
import pytest
from collections import defaultdict


def _tevatron_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_tevatron_src_to_path():
    src = _tevatron_root() / "src"
    sys.path.insert(0, str(src))


@pytest.mark.unit
def test_search_chunked_vs_non_chunked():
    """
    Test search behavior differences between chunked and non-chunked modes.
    This verifies:
    1. Auto-detection of chunked format
    2. MaxSim aggregation logic
    3. Search depth handling
    """
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked, search_queries
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    # Create mock query and passage embeddings
    num_queries = 3
    num_docs = 10
    hidden_size = 64
    
    # Query embeddings
    q_reps = np.random.randn(num_queries, hidden_size).astype(np.float32)
    # Normalize for inner product search
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    # Test Case 1: Non-chunked format
    # Each document has one embedding
    p_reps_non_chunked = np.random.randn(num_docs, hidden_size).astype(np.float32)
    # Normalize for inner product search
    p_reps_non_chunked = p_reps_non_chunked / np.linalg.norm(p_reps_non_chunked, axis=1, keepdims=True)
    p_lookup_non_chunked = [f"doc_{i}" for i in range(num_docs)]
    
    retriever_non_chunked = FaissFlatSearcher(p_reps_non_chunked)
    # Need to add embeddings to index
    retriever_non_chunked.add(p_reps_non_chunked)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 10
    
    args = MockArgs()
    
    # Non-chunked search
    scores_non_chunked, indices_non_chunked = search_queries(
        retriever_non_chunked, q_reps, p_lookup_non_chunked, args
    )
    
    # Verify non-chunked results
    assert len(scores_non_chunked) == num_queries
    assert len(indices_non_chunked) == num_queries
    for q_idx in range(num_queries):
        assert len(scores_non_chunked[q_idx]) == args.depth
        assert len(indices_non_chunked[q_idx]) == args.depth
        # indices_non_chunked contains document IDs (strings), not indices
        assert all(isinstance(doc_id, (str, np.str_)) for doc_id in indices_non_chunked[q_idx][:5])
    
    # Test Case 2: Chunked format - single chunk per document
    # This simulates chunk_size == max_passage_size scenario
    # Each document has exactly one chunk
    p_reps_chunked_single = np.random.randn(num_docs, hidden_size).astype(np.float32)
    # Normalize for inner product search
    p_reps_chunked_single = p_reps_chunked_single / np.linalg.norm(p_reps_chunked_single, axis=1, keepdims=True)
    q_reps_normalized = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_lookup_chunked_single = [(f"doc_{i}", 0) for i in range(num_docs)]
    
    retriever_chunked_single = FaissFlatSearcher(p_reps_chunked_single)
    # Need to add embeddings to index
    retriever_chunked_single.add(p_reps_chunked_single)
    
    # Chunked search with single chunk per doc
    results_chunked_single = search_queries_chunked(
        retriever_chunked_single, q_reps_normalized, p_lookup_chunked_single, args
    )
    
    # Verify chunked results
    assert len(results_chunked_single) == num_queries
    for q_idx in range(num_queries):
        # Results might be less than depth if fewer documents exist
        assert len(results_chunked_single[q_idx]) <= args.depth
        assert len(results_chunked_single[q_idx]) > 0, "Should have at least some results"
        # Each result should be (doc_id, score) tuple
        for doc_id, score in results_chunked_single[q_idx]:
            assert isinstance(doc_id, str)
            assert isinstance(score, (int, float, np.floating))
    
    # Test Case 3: Chunked format - multiple chunks per document
    # Some documents have multiple chunks
    num_chunks_total = 20  # More chunks than documents
    p_reps_chunked_multi = np.random.randn(num_chunks_total, hidden_size).astype(np.float32)
    # Normalize for inner product search
    p_reps_chunked_multi = p_reps_chunked_multi / np.linalg.norm(p_reps_chunked_multi, axis=1, keepdims=True)
    p_lookup_chunked_multi = []
    # Document 0-4: 2 chunks each (10 chunks)
    # Document 5-9: 2 chunks each (10 chunks)
    for doc_idx in range(num_docs):
        for chunk_idx in range(2):
            p_lookup_chunked_multi.append((f"doc_{doc_idx}", chunk_idx))
    
    retriever_chunked_multi = FaissFlatSearcher(p_reps_chunked_multi)
    retriever_chunked_multi.add(p_reps_chunked_multi)
    
    # Chunked search with multiple chunks per doc
    results_chunked_multi = search_queries_chunked(
        retriever_chunked_multi, q_reps, p_lookup_chunked_multi, args
    )
    
    # Verify MaxSim aggregation
    assert len(results_chunked_multi) == num_queries
    for q_idx in range(num_queries):
        assert len(results_chunked_multi[q_idx]) == args.depth
        # Verify MaxSim: each document should appear at most once
        doc_ids = [doc_id for doc_id, _ in results_chunked_multi[q_idx]]
        assert len(doc_ids) == len(set(doc_ids)), "Each document should appear only once (MaxSim aggregation)"
        
        # Verify scores are in descending order
        scores = [score for _, score in results_chunked_multi[q_idx]]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    
    # Test Case 4: Verify MaxSim logic - same document with multiple chunks
    # Create a scenario where one document has the best chunks
    q_rep_test = np.random.randn(1, hidden_size).astype(np.float32)
    q_rep_test = q_rep_test / np.linalg.norm(q_rep_test, axis=1, keepdims=True)
    
    # Create embeddings where doc_0 chunks are most similar to query
    p_reps_test = np.random.randn(5, hidden_size).astype(np.float32)
    # Make doc_0 chunks (indices 0, 1) more similar to query
    p_reps_test[0] = q_rep_test[0] * 0.9 + np.random.randn(hidden_size) * 0.1
    p_reps_test[1] = q_rep_test[0] * 0.8 + np.random.randn(hidden_size) * 0.2
    # Other chunks less similar
    p_reps_test[2:] = q_rep_test[0] * 0.3 + np.random.randn(3, hidden_size) * 0.7
    # Normalize
    p_reps_test = p_reps_test / np.linalg.norm(p_reps_test, axis=1, keepdims=True)
    
    p_lookup_test = [
        ("doc_0", 0),  # Best chunk
        ("doc_0", 1),  # Second best chunk
        ("doc_1", 0),  # Less similar
        ("doc_2", 0),  # Less similar
        ("doc_3", 0),  # Less similar
    ]
    
    retriever_test = FaissFlatSearcher(p_reps_test)
    retriever_test.add(p_reps_test)
    results_test = search_queries_chunked(retriever_test, q_rep_test, p_lookup_test, args)
    
    # Verify MaxSim: doc_0 should be ranked first (max of its two chunks)
    assert len(results_test) == 1
    assert len(results_test[0]) > 0, "Should have results"
    top_doc = results_test[0][0][0]
    assert top_doc == "doc_0", "doc_0 should be ranked first due to MaxSim (max of its chunks)"
    
    # Test Case 5: Verify search depth multiplier
    args_large = MockArgs()
    args_large.depth = 5
    args_large.chunk_multiplier = 10
    args_large.batch_size = 0
    args_large.quiet = True
    
    # With chunk_multiplier=10, should search 5 * 10 = 50 chunks
    # But we only have 20 chunks, so should get all chunks
    results_depth_test = search_queries_chunked(
        retriever_chunked_multi, q_reps_normalized, p_lookup_chunked_multi, args_large
    )
    
    # Should return up to depth documents (after MaxSim aggregation)
    assert len(results_depth_test[0]) <= args_large.depth
    assert len(results_depth_test[0]) > 0, "Should have some results"
    
    # Test Case 6: Verify auto-detection logic
    # Test that tuple format is detected as chunked
    assert isinstance(p_lookup_chunked_single[0], tuple), "Chunked lookup should be tuple"
    assert not isinstance(p_lookup_non_chunked[0], tuple), "Non-chunked lookup should be string"
    
    # Test Case 7: Verify that single chunk per doc behaves correctly
    # When chunk_size == max_passage_size, each doc has one chunk
    # In this case, MaxSim should give same result as non-chunked (if embeddings are identical)
    # But search depth multiplier means we search more candidates
    p_reps_single_chunk = p_reps_chunked_single.copy()
    q_reps_single = q_reps_normalized.copy()
    
    # Search with same embeddings but different formats
    results_single_chunk = search_queries_chunked(
        retriever_chunked_single, q_reps_single, p_lookup_chunked_single, args
    )
    
    # Verify results structure
    assert len(results_single_chunk) == num_queries
    for q_idx in range(num_queries):
        assert len(results_single_chunk[q_idx]) > 0
        # Each result should be (doc_id, score)
        for doc_id, score in results_single_chunk[q_idx]:
            assert isinstance(doc_id, str)
            assert isinstance(score, (int, float, np.floating))


@pytest.mark.unit
def test_write_ranking():
    """Test write_ranking function for non-chunked search results."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import write_ranking
    import tempfile
    import os
    
    # Create mock data
    q_lookup = ["q1", "q2", "q3"]
    corpus_scores = [
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.95, 0.85, 0.75, 0.65, 0.55],
        [0.88, 0.78, 0.68, 0.58, 0.48]
    ]
    corpus_indices = [
        ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
        ["doc_10", "doc_20", "doc_30", "doc_40", "doc_50"],
        ["doc_100", "doc_200", "doc_300", "doc_400", "doc_500"]
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        write_ranking(corpus_indices, corpus_scores, q_lookup, temp_path)
        
        # Verify file contents
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 15  # 3 queries * 5 results
        
        # Check first query results (should be sorted by score descending)
        first_query_lines = lines[:5]
        scores = [float(line.strip().split('\t')[2]) for line in first_query_lines]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
        
        # Verify format: qid\tidx\tscore
        for line in lines:
            parts = line.strip().split('\t')
            assert len(parts) == 3, "Each line should have 3 parts: qid, idx, score"
            assert parts[0] in q_lookup, "Query ID should be in q_lookup"
            assert float(parts[2]) >= 0, "Score should be a number"
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
def test_write_ranking_chunked():
    """Test write_ranking_chunked function for chunked search results."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import write_ranking_chunked
    import tempfile
    import os
    
    # Create mock chunked results
    q_lookup = ["q1", "q2"]
    results = [
        [("doc_1", 0.95), ("doc_2", 0.85), ("doc_3", 0.75)],
        [("doc_10", 0.92), ("doc_20", 0.82), ("doc_30", 0.72)]
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        write_ranking_chunked(results, q_lookup, temp_path)
        
        # Verify file contents
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 6  # 2 queries * 3 results
        
        # Verify format: qid\tdoc_id\tscore
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            assert len(parts) == 3, "Each line should have 3 parts: qid, doc_id, score"
            
            # Check query ID
            if i < 3:
                assert parts[0] == "q1"
            else:
                assert parts[0] == "q2"
            
            # Check score is a number
            assert float(parts[2]) >= 0, "Score should be a number"
        
        # Verify scores are in descending order for each query
        q1_scores = [float(lines[i].strip().split('\t')[2]) for i in range(3)]
        q2_scores = [float(lines[i].strip().split('\t')[2]) for i in range(3, 6)]
        assert q1_scores == sorted(q1_scores, reverse=True)
        assert q2_scores == sorted(q2_scores, reverse=True)
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
def test_pickle_load_save():
    """Test pickle_load and pickle_save functions."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import pickle_load, pickle_save
    import tempfile
    import os
    
    # Create test data
    test_reps = np.random.randn(10, 64).astype(np.float32)
    test_lookup = [f"doc_{i}" for i in range(10)]
    test_data = (test_reps, test_lookup)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        # Save
        pickle_save(test_data, temp_path)
        assert os.path.exists(temp_path), "Pickle file should be created"
        
        # Load
        loaded_reps, loaded_lookup = pickle_load(temp_path)
        
        # Verify data integrity
        assert np.array_equal(loaded_reps, test_reps), "Embeddings should match"
        assert loaded_lookup == test_lookup, "Lookup should match"
        assert isinstance(loaded_reps, np.ndarray), "Loaded reps should be numpy array"
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.unit
def test_search_batch_size():
    """Test that batch_size parameter works correctly."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    num_queries = 10
    num_docs = 20
    hidden_size = 64
    
    q_reps = np.random.randn(num_queries, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_docs, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [f"doc_{i}" for i in range(num_docs)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        quiet = True
        chunk_multiplier = 10
    
    # Test with batch_size = 0 (no batching)
    args_no_batch = MockArgs()
    args_no_batch.batch_size = 0
    scores_no_batch, indices_no_batch = search_queries(retriever, q_reps, p_lookup, args_no_batch)
    
    # Test with batch_size > 0 (batching)
    args_batch = MockArgs()
    args_batch.batch_size = 3
    scores_batch, indices_batch = search_queries(retriever, q_reps, p_lookup, args_batch)
    
    # Results should be the same regardless of batching
    assert len(scores_no_batch) == len(scores_batch) == num_queries
    assert len(indices_no_batch) == len(indices_batch) == num_queries
    
    # Scores should match (allowing for small numerical differences)
    for q_idx in range(num_queries):
        assert len(scores_no_batch[q_idx]) == len(scores_batch[q_idx]) == args_no_batch.depth
        # Scores should be very similar (allowing for floating point precision)
        np.testing.assert_allclose(scores_no_batch[q_idx], scores_batch[q_idx], rtol=1e-5)


@pytest.mark.unit
def test_search_chunked_with_negative_indices():
    """Test chunked search handles FAISS -1 indices correctly."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    from unittest.mock import Mock, patch
    
    hidden_size = 64
    num_docs = 3
    num_chunks = 5
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_chunks, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [
        ("doc_0", 0),
        ("doc_0", 1),
        ("doc_1", 0),
        ("doc_2", 0),
        ("doc_2", 1),
    ]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 10  # Request more than available
        batch_size = 0
        quiet = True
        chunk_multiplier = 1
    
    args = MockArgs()
    
    # Mock search to return -1 for insufficient results
    original_search = retriever.search
    
    def mock_search(q_reps, k):
        scores, indices = original_search(q_reps, k)
        # Simulate FAISS returning -1 for insufficient results
        if k > num_chunks:
            # Pad with -1 indices
            padded_indices = np.full((scores.shape[0], k), -1, dtype=indices.dtype)
            padded_scores = np.full((scores.shape[0], k), -np.inf, dtype=scores.dtype)
            padded_indices[:, :indices.shape[1]] = indices
            padded_scores[:, :scores.shape[1]] = scores
            return padded_scores, padded_indices
        return scores, indices
    
    retriever.search = mock_search
    
    results = search_queries_chunked(retriever, q_reps, p_lookup, args)
    
    # Should handle -1 indices gracefully
    assert len(results) == 1
    assert len(results[0]) <= num_docs  # Should aggregate to unique documents
    # All results should be valid (doc_id, score) tuples
    for doc_id, score in results[0]:
        assert isinstance(doc_id, str)
        assert isinstance(score, (int, float, np.floating))
        assert not np.isinf(score), "Scores should not be infinite"


@pytest.mark.unit
def test_search_single_query():
    """Test search with a single query."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries, search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_docs = 10
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_docs, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [f"doc_{i}" for i in range(num_docs)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 10
    
    args = MockArgs()
    
    # Non-chunked search
    scores, indices = search_queries(retriever, q_reps, p_lookup, args)
    assert len(scores) == 1
    assert len(indices) == 1
    assert len(scores[0]) == args.depth
    assert len(indices[0]) == args.depth
    
    # Chunked search
    p_lookup_chunked = [(f"doc_{i}", 0) for i in range(num_docs)]
    results = search_queries_chunked(retriever, q_reps, p_lookup_chunked, args)
    assert len(results) == 1
    assert len(results[0]) <= args.depth
    assert len(results[0]) > 0


@pytest.mark.unit
def test_search_empty_results():
    """Test search behavior with edge cases."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    
    # Single query, no passages
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    # Empty passage index
    p_reps = np.random.randn(0, hidden_size).astype(np.float32)
    p_lookup = []
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 10
    
    args = MockArgs()
    
    # Should handle empty index gracefully
    results = search_queries_chunked(retriever, q_reps, p_lookup, args)
    assert len(results) == 1
    assert len(results[0]) == 0, "Should return empty results for empty index"


@pytest.mark.unit
def test_search_depth_larger_than_documents():
    """Test search when depth is larger than available documents."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries, search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_docs = 5
    
    q_reps = np.random.randn(2, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_docs, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [f"doc_{i}" for i in range(num_docs)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 20  # Larger than num_docs
        batch_size = 0
        quiet = True
        chunk_multiplier = 10
    
    args = MockArgs()
    
    # Non-chunked: should return depth results (with padding if needed)
    scores, indices = search_queries(retriever, q_reps, p_lookup, args)
    assert len(scores) == 2
    assert len(scores[0]) == args.depth  # FAISS will pad with -1 indices
    
    # Chunked: should return at most num_docs results
    p_lookup_chunked = [(f"doc_{i}", 0) for i in range(num_docs)]
    results = search_queries_chunked(retriever, q_reps, p_lookup_chunked, args)
    assert len(results) == 2
    for q_result in results:
        assert len(q_result) <= num_docs, "Should not return more documents than available"


@pytest.mark.unit
def test_search_chunked_multiplier_effect():
    """Test that chunk_multiplier affects search depth correctly."""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_docs = 10
    chunks_per_doc = 3
    num_chunks = num_docs * chunks_per_doc
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_chunks, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [(f"doc_{i}", j) for i in range(num_docs) for j in range(chunks_per_doc)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
    
    # Test with different multipliers
    for multiplier in [1, 5, 10]:
        args = MockArgs()
        args.chunk_multiplier = multiplier
        
        results = search_queries_chunked(retriever, q_reps, p_lookup, args)
        
        # Should search depth * multiplier chunks
        # After MaxSim aggregation, should return at most depth documents
        assert len(results) == 1
        assert len(results[0]) <= args.depth, f"With multiplier {multiplier}, should return at most {args.depth} docs"
        assert len(results[0]) > 0, "Should have some results"


@pytest.mark.unit
def test_index_boundary_check():
    """Verify index boundary check - ensure no out-of-bounds access to p_lookup"""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_chunks = 10
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_chunks, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [(f"doc_{i}", 0) for i in range(num_chunks)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 10  # Will search 5 * 10 = 50 chunks, but only 10 available
    
    args = MockArgs()
    
    # Should not raise IndexError, FAISS will return -1 or valid indices
    results = search_queries_chunked(retriever, q_reps, p_lookup, args)
    
    assert len(results) == 1
    # Should handle gracefully without out-of-bounds
    assert len(results[0]) <= num_chunks


@pytest.mark.unit
def test_p_lookup_format_validation():
    """Verify p_lookup format - must be (doc_id, chunk_idx) tuples"""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_chunks = 5
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_chunks, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    
    # Correct format: tuples
    p_lookup_correct = [(f"doc_{i}", i % 2) for i in range(num_chunks)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 1
    
    args = MockArgs()
    
    # Should work correctly
    results = search_queries_chunked(retriever, q_reps, p_lookup_correct, args)
    assert len(results) == 1
    
    # Wrong format: strings (non-chunked format)
    p_lookup_wrong = [f"doc_{i}" for i in range(num_chunks)]
    
    # Function will catch errors and continue, won't raise exception
    # but will log error messages
    results = search_queries_chunked(retriever, q_reps, p_lookup_wrong, args)
    # Due to format error, should return empty or partial results
    assert len(results) == 1


@pytest.mark.unit
def test_maxsim_aggregation_correctness():
    """Verify MaxSim aggregation correctness"""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    
    # Create a query
    q_rep = np.random.randn(1, hidden_size).astype(np.float32)
    q_rep = q_rep / np.linalg.norm(q_rep, axis=1, keepdims=True)
    
    # Create documents: doc_0 has 3 chunks, doc_1 has 2 chunks
    # Make doc_0's chunk 0 most similar, chunk 1 second most, chunk 2 less similar
    # Make doc_1's chunks less similar
    p_reps = np.random.randn(5, hidden_size).astype(np.float32)
    
    # doc_0's chunk 0: most similar
    p_reps[0] = q_rep[0] * 0.95 + np.random.randn(hidden_size) * 0.05
    # doc_0's chunk 1: second most similar
    p_reps[1] = q_rep[0] * 0.85 + np.random.randn(hidden_size) * 0.15
    # doc_0's chunk 2: less similar
    p_reps[2] = q_rep[0] * 0.50 + np.random.randn(hidden_size) * 0.50
    # doc_1's chunks: less similar
    p_reps[3] = q_rep[0] * 0.40 + np.random.randn(hidden_size) * 0.60
    p_reps[4] = q_rep[0] * 0.35 + np.random.randn(hidden_size) * 0.65
    
    # Normalize
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    
    p_lookup = [
        ("doc_0", 0),  # Most similar
        ("doc_0", 1),  # Second most similar
        ("doc_0", 2),  # Less similar
        ("doc_1", 0),  # Less similar
        ("doc_1", 1),  # Less similar
    ]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    class MockArgs:
        depth = 10
        batch_size = 0
        quiet = True
        chunk_multiplier = 1
    
    args = MockArgs()
    
    results = search_queries_chunked(retriever, q_rep, p_lookup, args)
    
    assert len(results) == 1
    assert len(results[0]) >= 1
    
    # doc_0 should be ranked first (because its max score is chunk 0's score, highest)
    top_doc = results[0][0][0]
    assert top_doc == "doc_0", f"doc_0 should be top (has best chunk), but got {top_doc}"
    
    # Verify each document appears only once (MaxSim aggregation)
    doc_ids = [doc_id for doc_id, _ in results[0]]
    assert len(doc_ids) == len(set(doc_ids)), "Each document should appear only once"
    
    # Verify scores are in descending order
    scores = [score for _, score in results[0]]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"


@pytest.mark.unit
def test_empty_doc_max_scores():
    """Test edge case when all results are -1"""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(1, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [("doc_0", 0)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    # Mock search to return all -1
    original_search = retriever.search
    
    def mock_search_all_negative(q_reps, k):
        scores = np.array([[-np.inf] * k])
        indices = np.array([[-1] * k])
        return scores, indices
    
    retriever.search = mock_search_all_negative
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 1
    
    args = MockArgs()
    
    results = search_queries_chunked(retriever, q_reps, p_lookup, args)
    
    # Should return empty results, not crash
    assert len(results) == 1
    assert len(results[0]) == 0, "Should return empty list when all indices are -1"


@pytest.mark.unit
def test_index_out_of_bounds_protection():
    """Test index out-of-bounds protection - if FAISS returns out-of-range indices"""
    _add_tevatron_src_to_path()
    from tevatron.retriever.driver.search import search_queries_chunked
    from tevatron.retriever.searcher import FaissFlatSearcher
    
    hidden_size = 64
    num_chunks = 5
    
    q_reps = np.random.randn(1, hidden_size).astype(np.float32)
    q_reps = q_reps / np.linalg.norm(q_reps, axis=1, keepdims=True)
    
    p_reps = np.random.randn(num_chunks, hidden_size).astype(np.float32)
    p_reps = p_reps / np.linalg.norm(p_reps, axis=1, keepdims=True)
    p_lookup = [(f"doc_{i}", 0) for i in range(num_chunks)]
    
    retriever = FaissFlatSearcher(p_reps)
    retriever.add(p_reps)
    
    # Mock search to return out-of-bounds indices
    original_search = retriever.search
    
    def mock_search_out_of_bounds(q_reps, k):
        # Return some valid indices and some out-of-bounds indices
        scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
        indices = np.array([[0, 1, 2, 10, 20]])  # 10 and 20 are out of bounds
        return scores, indices
    
    retriever.search = mock_search_out_of_bounds
    
    class MockArgs:
        depth = 5
        batch_size = 0
        quiet = True
        chunk_multiplier = 1
    
    args = MockArgs()
    
    # Function will catch out-of-bounds indices and log warnings, won't raise exception
    results = search_queries_chunked(retriever, q_reps, p_lookup, args)
    # Should handle gracefully, only using valid indices
    assert len(results) == 1
    # Since we have 3 valid indices (0, 1, 2), should have some results
    assert len(results[0]) <= 3  # At most 3 documents (corresponding to indices 0, 1, 2)
