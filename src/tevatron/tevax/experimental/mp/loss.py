import math
import optax
import jax.numpy as jnp

from functools import partial
from jax import lax
from jax.sharding import PartitionSpec as PS

# Contrastive loss over 2d-mesh with manually designed parallel collective procedures
def contrastive_loss_2dm(ss, tt, scale_by_dim=False):
    import jax._src.mesh as mesh_lib
    from jax.experimental import shard_map
    
    if scale_by_dim:
        scale = 1.0 / math.sqrt(ss.shape[-1])
        ss, tt = ss * scale, tt * scale

    mesh = mesh_lib.thread_resources.env.physical_mesh
    DATA_AXIS, MODEL_AXIS = mesh.axis_names
    
    @partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(PS(DATA_AXIS, MODEL_AXIS), PS(DATA_AXIS, MODEL_AXIS)),
        out_specs=PS(DATA_AXIS, None),
        check_rep=False,
    )
    def sim(ss, tt):
        dp_size = lax.psum(1, DATA_AXIS)
        dp_idx = lax.axis_index(DATA_AXIS)
        acc = jnp.empty((ss.shape[0], tt.shape[0]*dp_size), dtype=jnp.float32)
        for op_idx in range(0, dp_size):
            sim_chunk = jnp.einsum('si,ti->st', ss, tt, preferred_element_type=jnp.float32)
            ss = lax.ppermute(
                ss,
                axis_name=DATA_AXIS,
                perm=[(j, (j - 1) % dp_size) for j in range(dp_size)]
            )
            sim_chunk = lax.psum(sim_chunk, MODEL_AXIS)
            if op_idx > 0:
                sim_chunk = lax.ppermute(
                    sim_chunk,
                    axis_name=DATA_AXIS,
                    perm=[(j, (j + op_idx) % dp_size) for j in range(dp_size)]
                )
            update_idx = (dp_idx - op_idx) % dp_size
            acc = lax.dynamic_update_slice(acc, sim_chunk, (0, update_idx*tt.shape[0]))
        
        return acc

    total_targets = tt.shape[0]
    per_sample_targets = int(tt.shape[0] / ss.shape[0])
    labels = jnp.arange(0, total_targets, per_sample_targets)
    scores = sim(ss, tt)
    return optax.softmax_cross_entropy_with_integer_labels(scores, labels)