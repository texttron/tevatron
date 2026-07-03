"""FLOPS-regularized trainer for SPLADE, kept separate from the dense
``TevatronTrainer`` so the core training path is untouched.

SPLADE needs an explicit sparsity pressure on the learned representations,
otherwise the encoder happily activates the whole vocabulary and the inverted
index blows up. We add the FLOPS regularizer of Paria et al. (2020,
"Minimizing FLOPs to Learn Efficient Sparse Representations",
https://arxiv.org/abs/2004.05665): penalize ``sum_j (mean_i |w_ij|)^2`` over the
batch, which pushes the expected per-dimension activation toward zero and is a
smooth surrogate for the index's retrieval FLOPs.

The penalty weight is ramped quadratically from 0 to its target over
``flops_warmup`` steps (same schedule as the paper) so early training isn't
dominated by the sparsity term before the model has learned anything useful.

This subclasses ``TevatronTrainer`` and only overrides ``compute_loss`` /
``training_step`` — the dense path, save logic, and distillation trainer are
left exactly as they are.
"""

import logging

import torch

from .trainer import TevatronTrainer

logger = logging.getLogger(__name__)


class FlopsScheduler:
    """Quadratic ramp of the FLOPS weight from 0 to ``target`` over ``T`` steps.

    Matches the schedule in Paria et al.: ``lambda_t = target * (t / T)^2`` for
    ``t <= T``, then held constant.
    """

    def __init__(self, target: float, T: int):
        self.target = target
        self.T = max(int(T), 1)
        self.t = 0
        self.value = 0.0

    def step(self):
        if self.t < self.T:
            self.t += 1
            self.value = self.target * (self.t / self.T) ** 2
        return self.value


class SpladeTrainer(TevatronTrainer):
    """``TevatronTrainer`` + FLOPS sparsity regularization on q/p reps.

    Expects the model to return ``q_reps`` / ``p_reps`` (the sparse vocab-space
    vectors), which ``SpladeModel`` / ``SpladeModelForCausalLM`` already do.
    Reads ``q_flops_loss_factor``, ``p_flops_loss_factor``, and ``flops_warmup``
    off ``self.args`` (see ``SpladeTrainingArguments``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_flops = FlopsScheduler(self.args.q_flops_loss_factor, self.args.flops_warmup)
        self.p_flops = FlopsScheduler(self.args.p_flops_loss_factor, self.args.flops_warmup)

    @staticmethod
    def _flops(reps: torch.Tensor) -> torch.Tensor:
        # sum_j (mean_i |w_ij|)^2  over the batch
        return torch.sum(torch.mean(torch.abs(reps), dim=0) ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query, passage = inputs
        output = model(query=query, passage=passage)

        # contrastive loss is already computed inside the model's forward (it
        # gathers across DDP ranks and applies the listwise CE); add the FLOPS
        # penalty on top, scaled by the warmup schedule.
        loss = output.loss
        q_loss = self.q_flops.value * self._flops(output.q_reps)
        p_loss = self.p_flops.value * self._flops(output.p_reps)
        if self.is_ddp:
            # the model scales its CE loss by world_size to counter DDP's grad
            # averaging; the FLOPS terms are computed on local reps, so match it.
            q_loss = q_loss * self._dist_loss_scale_factor
            p_loss = p_loss * self._dist_loss_scale_factor

        total = loss + q_loss + p_loss

        if self.state.global_step % max(self.args.logging_steps, 1) == 0 and self.is_world_process_zero():
            with torch.no_grad():
                q_nnz = (output.q_reps > 0).float().sum(-1).mean().item()
                p_nnz = (output.p_reps > 0).float().sum(-1).mean().item()
            self.log({
                "splade/q_flops_loss": float(q_loss),
                "splade/p_flops_loss": float(p_loss),
                "splade/flops_lambda": self.q_flops.value,
                "splade/q_nnz": q_nnz,
                "splade/p_nnz": p_nnz,
            })

        return total

    def training_step(self, *args, **kwargs):
        # advance the FLOPS warmup once per optimization step
        self.q_flops.step()
        self.p_flops.step()
        return super().training_step(*args, **kwargs)
