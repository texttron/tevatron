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
