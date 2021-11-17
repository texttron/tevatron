from functools import partial
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


def grad_cache_train_step(state, queries, passages, dropout_rng, axis='device', q_n_subbatch=1, p_n_subbatch=1):
    try:
        from grad_cache import cachex
    except ImportError:
        raise ModuleNotFoundError('GradCache packaged needs to be installed for running grad_cache_train_step')

    def encode_query(params, **kwargs):
        return state.apply_fn(**kwargs, params=params.q_params, train=True)[0][:, 0, :]

    def encode_passage(params, **kwargs):
        return state.apply_fn(**kwargs, params=params.p_params, train=True)[0][:, 0, :]

    queries, passages = cachex.tree_chunk(queries, q_n_subbatch), cachex.tree_chunk(passages, p_n_subbatch)
    q_rngs, p_rngs, new_rng = jax.random.split(dropout_rng, 3)
    q_rngs = jax.random.split(q_rngs, q_n_subbatch)
    p_rngs = jax.random.split(p_rngs, p_n_subbatch)

    q_reps = cachex.chunk_encode(partial(encode_query, state.params))(**queries, dropout_rng=q_rngs)
    p_reps = cachex.chunk_encode(partial(encode_passage, state.params))(**passages, dropout_rng=p_rngs)

    @cachex.unchunk_args(axis=0, argnums=(0, 1))
    def compute_loss(xx, yy):
        return jnp.mean(p_contrastive_loss(xx, yy, axis=axis))

    loss, (q_grads, p_grads) = jax.value_and_grad(compute_loss, argnums=(0, 1))(q_reps, p_reps)

    grads = jax.tree_map(lambda v: jnp.zeros_like(v), state.params)
    grads = cachex.cache_grad(encode_query)(state.params, grads, q_grads, **queries, dropout_rng=q_rngs)
    grads = cachex.cache_grad(encode_passage)(state.params, grads, p_grads, **passages, dropout_rng=p_rngs)

    loss, grads = jax.lax.pmean([loss, grads], axis)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state, new_rng
