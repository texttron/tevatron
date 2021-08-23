import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:
    def __init__(self, n_target: int = 1):
        self.target_per_qry = n_target

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(
                0, x.size(0) * self.target_per_qry, self.target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(n_target=n_target)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)