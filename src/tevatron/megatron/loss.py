import torch
import torch.nn.functional as F
from torch import Tensor


def reranker_loss(scores: Tensor, group_size: int) -> Tensor:
    """Listwise cross-entropy loss for pointwise reranker.

    Args:
        scores: (batch * group_size,) scalar relevance scores.
        group_size: number of passages per query (1 positive + N negatives).

    Returns:
        Scalar loss. Target is always index 0 (first passage is positive).
    """
    grouped = scores.view(-1, group_size)
    target = torch.zeros(grouped.size(0), dtype=torch.long, device=scores.device)
    return F.cross_entropy(grouped, target)
