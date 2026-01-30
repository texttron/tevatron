"""
DDP chunking pipeline test — run with torchrun.

Usage:
    torchrun --nproc_per_node=2 tests/test_ddp.py

This test launches 2 ranks with hardcoded data that differs per rank,
then verifies:

  1. _pooling_chunked: all_reduce(MAX) syncs max_chunks across ranks
  2. Gather: all_gather concatenates q_reps, p_reps, chunk_mask along dim 0
  3. MaxSim: compute_maxsim_similarity on the gathered tensors gives correct scores
  4. Contrastive loss: cross_entropy targets are correct after gathering
  5. Search equivalence: training MaxSim matches search-style aggregation

Scenario:
  Rank 0 has 1 query and 2 passages (3 chunks, 1 chunk)
  Rank 1 has 1 query and 2 passages (2 chunks, 2 chunks)
  After gathering: 2 queries, 4 passages — full contrastive matrix
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
from unittest.mock import Mock

import torch
import torch.distributed as dist


def _add_tevatron_src_to_path():
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_tevatron_src_to_path()
from tevatron.retriever.modeling.dense import DenseModel
from tevatron.retriever.modeling.encoder import EncoderModel


class _TestEncoderModel(EncoderModel):
    def encode_query(self, qry):
        raise NotImplementedError
    def encode_passage(self, psg):
        raise NotImplementedError


def dist_gather_tensor(t, name="tensor"):
    """
    Simplified all_gather for testing (CPU/gloo compatible).
    Replicates the trainer's _dist_gather_tensor logic without the cuda hardcode.
    """
    if t is None:
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(all_tensors, t)
    # Keep local tensor for gradient flow
    all_tensors[rank] = t
    return torch.cat(all_tensors, dim=0)


def main():
    # Initialize distributed (torchrun sets env vars)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"This test requires exactly 2 ranks, got {world_size}"

    H = 4  # hidden size
    passed = 0
    total = 0

    def check(condition, msg):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
        else:
            print(f"  [Rank {rank}] FAIL: {msg}")

    print(f"[Rank {rank}] Starting DDP tests (world_size={world_size})")

    # =========================================================================
    # Test 1: _pooling_chunked syncs max_chunks via all_reduce(MAX)
    # =========================================================================
    print(f"\n[Rank {rank}] Test 1: _pooling_chunked max_chunks sync")

    model = DenseModel(encoder=Mock(), pooling='last', normalize=False)
    model.passage_chunk_size = 1  # Enable chunked mode

    if rank == 0:
        # Rank 0: 1 passage with 3 chunks
        #   hidden: [1, 10, 4], eos at positions [2, 5, 8]
        hidden = torch.zeros(1, 10, H)
        hidden[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # chunk 0
        hidden[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # chunk 1
        hidden[0, 8] = torch.tensor([0.0, 0.0, 1.0, 0.0])  # chunk 2
        eos_positions = [[2, 5, 8]]  # 3 chunks
        local_max_chunks = 3
    else:
        # Rank 1: 1 passage with 1 chunk
        #   hidden: [1, 10, 4], eos at position [4]
        hidden = torch.zeros(1, 10, H)
        hidden[0, 4] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # chunk 0
        eos_positions = [[4]]  # 1 chunk
        local_max_chunks = 1

    chunk_reps, chunk_mask = model._pooling_chunked(hidden, eos_positions)

    # After all_reduce(MAX), max_chunks should be 3 on BOTH ranks
    check(chunk_reps.shape == (1, 3, H),
          f"chunk_reps shape: expected (1,3,{H}), got {chunk_reps.shape}")
    check(chunk_mask.shape == (1, 3),
          f"chunk_mask shape: expected (1,3), got {chunk_mask.shape}")

    if rank == 0:
        check(torch.allclose(chunk_mask, torch.tensor([[1.0, 1.0, 1.0]])),
              f"rank 0 mask should be [1,1,1], got {chunk_mask}")
        check(torch.allclose(chunk_reps[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0])),
              "rank 0 chunk 0 embedding wrong")
        check(torch.allclose(chunk_reps[0, 1], torch.tensor([0.0, 1.0, 0.0, 0.0])),
              "rank 0 chunk 1 embedding wrong")
        check(torch.allclose(chunk_reps[0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0])),
              "rank 0 chunk 2 embedding wrong")
    else:
        # Rank 1 had 1 chunk but output padded to 3 chunks
        check(torch.allclose(chunk_mask, torch.tensor([[1.0, 0.0, 0.0]])),
              f"rank 1 mask should be [1,0,0], got {chunk_mask}")
        check(torch.allclose(chunk_reps[0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0])),
              "rank 1 chunk 0 embedding wrong")
        check(torch.allclose(chunk_reps[0, 1], torch.tensor([0.0, 0.0, 0.0, 0.0])),
              "rank 1 chunk 1 should be zero-padded")
        check(torch.allclose(chunk_reps[0, 2], torch.tensor([0.0, 0.0, 0.0, 0.0])),
              "rank 1 chunk 2 should be zero-padded")

    print(f"[Rank {rank}] Test 1 OK: max_chunks synced from local={local_max_chunks} → global=3")

    # =========================================================================
    # Test 2: Gather q_reps, p_reps, chunk_mask across ranks
    # =========================================================================
    print(f"\n[Rank {rank}] Test 2: all_gather for q_reps, p_reps, chunk_mask")

    if rank == 0:
        # Rank 0: query = "football fan", 2 passages (3 chunks, 1 chunk)
        q_reps = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # [1, 4]
        p_reps = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0],     # p0_c0: football
             [0.0, 1.0, 0.0, 0.0],     # p0_c1: basketball
             [0.8, 0.6, 0.0, 0.0]],    # p0_c2: general sports
            [[0.5, 0.5, 0.0, 0.0],     # p1_c0: mixed
             [0.0, 0.0, 0.0, 0.0],     # p1 padding
             [0.0, 0.0, 0.0, 0.0]],    # p1 padding
        ])  # [2, 3, 4]
        chunk_mask = torch.tensor([
            [1.0, 1.0, 1.0],   # p0: 3 chunks
            [1.0, 0.0, 0.0],   # p1: 1 chunk
        ])
    else:
        # Rank 1: query = "science lover", 2 passages (2 chunks, 2 chunks)
        q_reps = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # [1, 4]
        p_reps = torch.tensor([
            [[0.0, 0.0, 1.0, 0.0],     # p2_c0: physics
             [0.0, 0.3, 0.9, 0.0],     # p2_c1: chemistry
             [0.0, 0.0, 0.0, 0.0]],    # p2 padding
            [[0.0, 0.0, 0.5, 0.5],     # p3_c0: biology
             [0.0, 0.0, 0.0, 1.0],     # p3_c1: math
             [0.0, 0.0, 0.0, 0.0]],    # p3 padding
        ])  # [2, 3, 4]
        chunk_mask = torch.tensor([
            [1.0, 1.0, 0.0],   # p2: 2 chunks
            [1.0, 1.0, 0.0],   # p3: 2 chunks
        ])

    # Gather across ranks
    all_q = dist_gather_tensor(q_reps)
    all_p = dist_gather_tensor(p_reps)
    all_mask = dist_gather_tensor(chunk_mask)

    check(all_q.shape == (2, H),
          f"gathered q_reps shape: expected (2,{H}), got {all_q.shape}")
    check(all_p.shape == (4, 3, H),
          f"gathered p_reps shape: expected (4,3,{H}), got {all_p.shape}")
    check(all_mask.shape == (4, 3),
          f"gathered chunk_mask shape: expected (4,3), got {all_mask.shape}")

    # Verify gathered content (should be rank 0 data then rank 1 data)
    check(torch.allclose(all_q[0], torch.tensor([1.0, 0.0, 0.0, 0.0])),
          "gathered q[0] should be rank 0's query (football)")
    check(torch.allclose(all_q[1], torch.tensor([0.0, 0.0, 1.0, 0.0])),
          "gathered q[1] should be rank 1's query (science)")

    # Passage ordering: p0, p1 (from rank 0), p2, p3 (from rank 1)
    check(torch.allclose(all_p[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0])),
          "p0_c0 (football) wrong")
    check(torch.allclose(all_p[2, 0], torch.tensor([0.0, 0.0, 1.0, 0.0])),
          "p2_c0 (physics) wrong")

    expected_mask = torch.tensor([
        [1.0, 1.0, 1.0],   # p0
        [1.0, 0.0, 0.0],   # p1
        [1.0, 1.0, 0.0],   # p2
        [1.0, 1.0, 0.0],   # p3
    ])
    check(torch.allclose(all_mask, expected_mask),
          f"gathered mask wrong:\n{all_mask}\nexpected:\n{expected_mask}")

    print(f"[Rank {rank}] Test 2 OK: gathered Q=2, P=4, C=3")

    # =========================================================================
    # Test 3: MaxSim on gathered tensors
    # =========================================================================
    print(f"\n[Rank {rank}] Test 3: MaxSim on gathered tensors")

    """
    After gathering, all ranks have the same all_q [2,4], all_p [4,3,4], all_mask [4,3].

    Expected dot products and MaxSim:

    q0 = [1,0,0,0] ("football"):
      vs p0: c0=[1,0,0,0]→1.0, c1=[0,1,0,0]→0.0, c2=[0.8,0.6,0,0]→0.8  → max=1.0 (c0)
      vs p1: c0=[0.5,0.5,0,0]→0.5                                         → max=0.5 (c0)
      vs p2: c0=[0,0,1,0]→0.0, c1=[0,0.3,0.9,0]→0.0                      → max=0.0 (c0)
      vs p3: c0=[0,0,0.5,0.5]→0.0, c1=[0,0,0,1]→0.0                      → max=0.0 (c0)

    q1 = [0,0,1,0] ("science"):
      vs p0: c0→0.0, c1→0.0, c2→0.0                                       → max=0.0
      vs p1: c0→0.0                                                         → max=0.0
      vs p2: c0=[0,0,1,0]→1.0, c1=[0,0.3,0.9,0]→0.9                      → max=1.0 (c0)
      vs p3: c0=[0,0,0.5,0.5]→0.5, c1=[0,0,0,1]→0.0                      → max=0.5 (c0)
    """
    sim_model = _TestEncoderModel(encoder=Mock(), pooling='last', normalize=False)
    scores = sim_model.compute_maxsim_similarity(all_q, all_p, all_mask)

    check(scores.shape == (2, 4), f"scores shape: expected (2,4), got {scores.shape}")

    # q0 scores
    check(torch.allclose(scores[0, 0], torch.tensor(1.0)),
          f"q0 vs p0: expected 1.0, got {scores[0,0].item()}")
    check(torch.allclose(scores[0, 1], torch.tensor(0.5)),
          f"q0 vs p1: expected 0.5, got {scores[0,1].item()}")
    check(torch.allclose(scores[0, 2], torch.tensor(0.0)),
          f"q0 vs p2: expected 0.0, got {scores[0,2].item()}")
    check(torch.allclose(scores[0, 3], torch.tensor(0.0)),
          f"q0 vs p3: expected 0.0, got {scores[0,3].item()}")

    # q1 scores
    check(torch.allclose(scores[1, 0], torch.tensor(0.0)),
          f"q1 vs p0: expected 0.0, got {scores[1,0].item()}")
    check(torch.allclose(scores[1, 1], torch.tensor(0.0)),
          f"q1 vs p1: expected 0.0, got {scores[1,1].item()}")
    check(torch.allclose(scores[1, 2], torch.tensor(1.0)),
          f"q1 vs p2: expected 1.0, got {scores[1,2].item()}")
    check(torch.allclose(scores[1, 3], torch.tensor(0.5)),
          f"q1 vs p3: expected 0.5, got {scores[1,3].item()}")

    # Verify which chunk was selected via manual einsum
    chunk_scores = torch.einsum('qh,pch->qpc', all_q, all_p)  # [2, 4, 3]
    padding_mask = ~all_mask.unsqueeze(0).bool()
    chunk_scores_masked = chunk_scores.masked_fill(padding_mask, float('-inf'))
    _, max_idx = chunk_scores_masked.max(dim=-1)

    # q0 vs p0: chunk 0 (football) selected
    check(max_idx[0, 0].item() == 0,
          f"q0 vs p0: expected chunk 0 selected, got chunk {max_idx[0,0].item()}")
    # q1 vs p2: chunk 0 (physics) selected
    check(max_idx[1, 2].item() == 0,
          f"q1 vs p2: expected chunk 0 selected, got chunk {max_idx[1,2].item()}")

    print(f"[Rank {rank}] Test 3 OK: MaxSim scores and chunk selection verified")

    # =========================================================================
    # Test 4: Contrastive loss targets after gathering
    # =========================================================================
    print(f"\n[Rank {rank}] Test 4: Contrastive loss targets")

    """
    After gathering:
      Q=2 queries, P=4 passages (2 passages per query = train_group_size=2)
      scores shape: [2, 4]

      Target: query i should match passage i * num_passages_per_query
        q0 → p0 (index 0)
        q1 → p2 (index 2)

    Score matrix:
            p0    p1    p2    p3
      q0  [1.0,  0.5,  0.0,  0.0]   ← target is p0 (correct, highest)
      q1  [0.0,  0.0,  1.0,  0.5]   ← target is p2 (correct, highest)
    """
    num_passages_per_query = all_p.size(0) // all_q.size(0)  # 4 // 2 = 2
    target = torch.arange(scores.size(0)) * num_passages_per_query

    check(target.tolist() == [0, 2],
          f"target should be [0, 2], got {target.tolist()}")

    # The positive passage should have the highest score for each query
    for qi in range(2):
        target_idx = target[qi].item()
        best_idx = scores[qi].argmax().item()
        check(best_idx == target_idx,
              f"q{qi}: best passage is p{best_idx}, expected p{target_idx}")

    # Compute cross-entropy loss
    temperature = 1.0
    loss = torch.nn.functional.cross_entropy(scores / temperature, target)
    check(loss.item() > 0, f"loss should be positive, got {loss.item()}")
    check(loss.item() < 1.0,
          f"loss should be small (positives have highest scores), got {loss.item()}")

    print(f"[Rank {rank}] Test 4 OK: targets=[0,2], loss={loss.item():.4f}")

    # =========================================================================
    # Test 5: Training MaxSim matches search-style aggregation
    # =========================================================================
    print(f"\n[Rank {rank}] Test 5: Training vs search MaxSim equivalence")

    """
    Flatten the gathered p_reps into individual chunk embeddings (as FAISS would store),
    compute per-chunk dot products, aggregate by doc with max → compare to training scores.
    """
    doc_names = ["p0", "p1", "p2", "p3"]

    for qi in range(2):
        q = all_q[qi]  # [H]

        # Search-style: compute dot product per chunk, max per doc
        search_doc_scores = {}
        for pi in range(4):
            doc_id = doc_names[pi]
            best_score = float('-inf')
            for ci in range(3):
                if all_mask[pi, ci] > 0:
                    dot = torch.dot(q, all_p[pi, ci]).item()
                    best_score = max(best_score, dot)
            search_doc_scores[doc_id] = best_score

        # Training-style: already computed in scores[qi]
        for pi, doc_id in enumerate(doc_names):
            training_score = scores[qi, pi].item()
            search_score = search_doc_scores[doc_id]
            check(abs(training_score - search_score) < 1e-5,
                  f"q{qi} vs {doc_id}: training={training_score:.4f} != search={search_score:.4f}")

    print(f"[Rank {rank}] Test 5 OK: training and search MaxSim agree")

    # =========================================================================
    # Test 6: Gather with None (all ranks None → returns None)
    # =========================================================================
    print(f"\n[Rank {rank}] Test 6: Gather None tensors")

    result = dist_gather_tensor(None)
    check(result is None, f"gather(None) should return None, got {result}")

    print(f"[Rank {rank}] Test 6 OK: gather(None) returns None")

    # =========================================================================
    # Summary
    # =========================================================================
    dist.barrier()

    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Results: {passed}/{total} checks passed")
    if passed == total:
        print(f"[Rank {rank}] ALL TESTS PASSED")
    else:
        print(f"[Rank {rank}] SOME TESTS FAILED ({total - passed} failures)")
    print(f"{'='*60}")

    dist.destroy_process_group()

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
