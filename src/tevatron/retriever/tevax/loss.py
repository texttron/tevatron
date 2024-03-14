import math
import jax.numpy as jnp
import optax
import chex

from jax import lax


def _onehot(labels: chex.Array, num_classes: int) -> chex.Array:
    x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
    x = lax.select(x, jnp.ones(x.shape), jnp.zeros(x.shape))
    return x.astype(jnp.float32)


def p_contrastive_loss(ss: chex.Array, tt: chex.Array, axis: str = 'device') -> chex.Array:
    per_shard_targets = tt.shape[0]
    per_sample_targets = int(tt.shape[0] / ss.shape[0])
    labels = jnp.arange(0, per_shard_targets, per_sample_targets) + per_shard_targets * lax.axis_index(axis)

    tt = lax.all_gather(tt, axis).reshape((-1, ss.shape[-1]))
    scores = jnp.dot(ss, jnp.transpose(tt))

    return optax.softmax_cross_entropy(scores, _onehot(labels, scores.shape[-1]))

def contrastive_loss(ss, tt, scale_by_dim=False):
    if scale_by_dim:
        scale = 1.0 / math.sqrt(ss.shape[-1])
        ss, tt = ss * scale, tt * scale

    total_targets = tt.shape[0]
    per_sample_targets = int(tt.shape[0] / ss.shape[0])
    labels = jnp.arange(0, total_targets, per_sample_targets)
    scores = jnp.einsum('si,ti->st', ss, tt, preferred_element_type=jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(scores, labels)