from typing import Tuple, Any, Union

import jax
from jax import numpy as jnp

from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax.struct import PyTreeNode

from .loss import p_contrastive_loss


class TiedParams(PyTreeNode):
    params: FrozenDict[str, Any]

    @property
    def q_params(self):
        return self.params

    @property
    def p_params(self):
        return self.params

    @classmethod
    def create(cls, params):
        return cls(params=params)


class DualParams(PyTreeNode):
    params: Tuple[FrozenDict[str, Any], FrozenDict[str, Any]]

    @property
    def q_params(self):
        return self.params[0]

    @property
    def p_params(self):
        return self.params[1]

    @classmethod
    def create(cls, *ps):
        if len(ps) == 1:
            return cls(params=ps*2)
        else:
            p_params, q_params = ps
            return cls(params=[p_params, q_params])


class RetrieverTrainState(TrainState):
    params: Union[TiedParams, DualParams]


def retriever_train_step(state, queries, passages, dropout_rng, axis='device'):
    q_dropout_rng, p_dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 3)

    def compute_loss(params):
        q_reps = state.apply_fn(**queries, params=params.q_params, dropout_rng=q_dropout_rng, train=True)[0][:, 0, :]
        p_reps = state.apply_fn(**passages, params=params.p_params, dropout_rng=p_dropout_rng, train=True)[0][:, 0, :]
        return jnp.mean(p_contrastive_loss(q_reps, p_reps, axis=axis))

    loss, grad = jax.value_and_grad(compute_loss)(state.params)
    loss, grad = jax.lax.pmean([loss, grad], axis)

    new_state = state.apply_gradients(grads=grad)

    return loss, new_state, new_dropout_rng

