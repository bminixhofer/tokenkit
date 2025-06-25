"""
Tokenkit sharding mechanism and sharding utilities.

Defines sharding patterns for various model architectures.

- Assumes a mesh with "data" (less communication) and "model" (more communication) dimensions.
- Assumes sharding is applied to parameters in a state with keys "params" (parameters) and "opt_state" (optimizer state).
- Only FSDP is supported for now.
"""

import logging

import jax
import jax.experimental
import jax.experimental.mesh_utils
import regex as re
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import PartitionSpec as P
import numpy as np

from tokenkit import utils

logger = logging.getLogger(__name__)


SHARD_PATTERNS = {
    "hypernet": {
        "(opt_state|params).*ffn_layer1.linear": P(None, "model"),
        "(opt_state|params).*ffn_layer2.linear": P("model", None),
        "(opt_state|params).*self_attention.(query|key|value).w": P(None, "model"),
        "(opt_state|params).*self_attention.post.w": P("model", None),
        "(opt_state|params).*embeddings": P("model", None),
    },
    "llama": {
        "(opt_state|params).*embed_tokens.*embedding": P("model", "data"),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.a": P(
            "model", "data"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.b": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.w": P(
            "data", "model"
        ),
        "(opt_state|params).*norm.weight": P("model"),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.o_proj.kernel": P("model", "data"),
        "(opt_state|params).*lm_head.kernel": P("data", "model"),
        "(opt_state|params).*mlp.down_proj.kernel": P("model", "data"),
        "(opt_state|params).*mlp.up_proj.kernel": P("data", "model"),
        "(opt_state|params).*mlp.gate_proj.kernel": P("data", "model"),
        "(opt_state|params).*norm.kernel": P("model"),
        ".*(cached_value|cached_key)": P("data", None, "model", None),
    },
    "mistral": {
        "(opt_state|params).*embed_tokens.*embedding": P("model", None),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel": P(None, "model"),
        "(opt_state|params).*self_attn.o_proj.kernel": P("model", None),
        "(opt_state|params).*lm_head.kernel": P(None, "model"),
        "(opt_state|params).*mlp.down_proj.kernel": P("model", None),
        "(opt_state|params).*mlp.up_proj.kernel": P(None, "model"),
        "(opt_state|params).*mlp.gate_proj.kernel": P(None, "model"),
    },
    "gemma": {
        "(opt_state|params).*embed_tokens.*embedding": P("model", "data"),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.a": P(
            "model", "data"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.b": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.w": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.o_proj.kernel": P("model", "data"),
        "(opt_state|params).*lm_head.kernel": P("data", "model"),
        "(opt_state|params).*mlp.down_proj.kernel": P("model", "data"),
        "(opt_state|params).*mlp.up_proj.kernel": P("data", "model"),
        "(opt_state|params).*mlp.gate_proj.kernel": P("data", "model"),
        "(opt_state|params).*norm.kernel": P("model"),
    },
    "gemma2": {
        "(opt_state|params).*embed_tokens.*embedding": P("model", "data"),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.a": P(
            "model", "data"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.b": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel.w": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.(q_proj|k_proj|v_proj).kernel": P(
            "data", "model"
        ),
        "(opt_state|params).*self_attn.o_proj.kernel": P("model", "data"),
        "(opt_state|params).*lm_head.kernel": P("data", "model"),
        "(opt_state|params).*mlp.down_proj.kernel": P("model", "data"),
        "(opt_state|params).*mlp.up_proj.kernel": P("data", "model"),
        "(opt_state|params).*mlp.gate_proj.kernel": P("data", "model"),
        "(opt_state|params).*norm.kernel": P("model"),
    },
    "gpt2": {
        "(opt_state|params).*c_attn.kernel": P(None, "model"),
        "(opt_state|params).*c_proj.kernel": P("model", None),
        "(opt_state|params).*c_fc.kernel": P(None, "model"),
    },
    "xlm-roberta": {
        "(opt_state|params).*self.(query|key|value).kernel": P(None, "model"),
        "(opt_state|params).*output.dense.kernel": P("model", None),
        "(opt_state|params).*intermediate.dense.kernel": P(None, "model"),
    },
}
SHARD_PATTERNS["tpu_llama"] = SHARD_PATTERNS["llama"]
SHARD_PATTERNS["tpu_gemma2"] = SHARD_PATTERNS["gemma2"]
SHARD_PATTERNS["gemma3"] = SHARD_PATTERNS["gemma2"]
SHARD_PATTERNS["tpu_gemma3"] = SHARD_PATTERNS["gemma2"]

def get_shard_patterns(kind: str) -> dict:
    """
    Get the sharding patterns for a given model kind.
    """

    return SHARD_PATTERNS.get(kind, {})


def get_sharding_fn(shard_patterns: dict, mesh: jax.sharding.Mesh) -> callable:
    """
    Returns a function that, when applied to a pytree, returns a pytree of the same structure
    with `jax.sharding.PartitionSpec` objects for each leaf, based on the provided sharding patterns.

    Args:
        shard_patterns (dict): A dictionary where keys are regex patterns and values are partition specs.
        mesh (jax.sharding.Mesh): The JAX mesh to use for sharding.

    Returns:
        A function that takes a pytree and returns a pytree of sharding specs.
    """

    name_to_size = {name: size for name, size in mesh.shape_tuple}

    def get_pspec(path, v):
        # this is a dummy parameter for e.g. PEFT, so no need to shard
        if np.prod(v.shape) == 0:
            return P()

        path_tuple = tuple(str(utils.keystr(x)) for x in path)
        path = ".".join(path_tuple)

        for key, value in shard_patterns.items():
            if re.match(key, path):
                pspec = value
                for dim, name in enumerate(pspec):
                    if name is None:
                        continue

                    if name not in name_to_size:
                        raise ValueError(
                            f"Unknown sharding name {name} in {pspec} for {path}"
                        )

                    if v.shape[dim] % name_to_size[name] != 0:
                        logger.warning(
                            "Want to shard %s with %s, but shape %s is not divisible by %s.",
                            path,
                            pspec,
                            v.shape,
                            name_to_size[name],
                        )
                        return P()

                logger.debug("Sharding %s with %s.", path, pspec)
                return P(*pspec)

        return P()

    def get_tree_shardings(tree):
        pspecs = jax.tree_util.tree_map_with_path(get_pspec, tree)
        return jax.tree.map(
            lambda pspec: jax.sharding.NamedSharding(mesh, pspec), pspecs
        )

    return get_tree_shardings


def to_global_array(pytree: dict, pytree_sharding: dict | None = None) -> dict:
    """
    Converts a pytree of local arrays to a pytree of global arrays with the same shape and contents
    (plus the supplied sharding).

    Recall: a global array is a JAX array that is sharded across multiple global devices (i.e. multi-node, like on a TPU pod), whereas
    a local array is a JAX array that is only shardeda cross the local devices (e.g. a single TPU slice).

    For this to be correct, the pytree must be exactly the same on all local devices (e.g. no differently randomly initialized parameters).

    Args:
        pytree: A pytree of local arrays (e.g. parameters, optimizer state).
        pytree_sharding: A pytree of sharding specs (e.g. PartitionSpec
    
    Returns:
        A pytree of global arrays, where each leaf is a JAX array that is sharded across the global devices.
        If `pytree_sharding` is None, the returned pytree will be fully replicated.
    """


    if pytree_sharding is None:
        pytree_sharding = jax.tree.map(lambda _: None, pytree)

    def to_global_array_fn(array, sharding):
        if array is None:
            return None

        if sharding is None:
            return array

        def cb(index):
            return array[index]

        return jax.make_array_from_callback(array.shape, sharding, cb)

    return jax.tree.map(to_global_array_fn, pytree, pytree_sharding)


def sync_across_devices(pytree):
    """
    Synchronizes a pytree across all devices via allgather + setting the  first device's value.

    Should usually be avoided, but useful to sync the contents of e.g. data batches (which can be different on each device).

    Args:
        pytree: The pytree to synchronize across devices. Each leaf must be a JAX array.

    Returns:
        A pytree with the same structure as `pytree`, but with all leaves synchronized.
    """

    if jax.process_count() == 1:
        return pytree

    return jax.tree.map(lambda x: x[0], process_allgather(pytree))


def to_devices(pytree: dict, pytree_sharding=None, dtype=None):
    """
    Converts a pytree of local arrays to a pytree of global arrays with the same shape and contents,
    plus the supplied sharding and dtype.

    See `to_global_array` for more details.

    The reason to supply this function in addition to `to_global_array` has been lost to time.
    Potentially only convenience to update the dtype. Reinvestigate?

    Args:
        pytree: A pytree of local arrays (e.g. parameters, optimizer state).
        pytree_sharding: A pytree of sharding specs (e.g. PartitionSpec
        dtype: The dtype to cast the arrays to. If None, no casting is done.

    Returns:
        A pytree of global arrays, where each leaf is a JAX array that is sharded across the global devices.
        If `pytree_sharding` is None, the returned pytree will be fully replicated.
        If `dtype` is not None, the arrays will be cast to the specified dtype.
    """

    # TODO: handle non-numpy inputs?
    pytree = to_global_array(pytree, pytree_sharding)

    return jax.jit(
        lambda x: x if dtype is None else jax.tree.map(lambda x: x.astype(dtype), x),
        in_shardings=(pytree_sharding,) if pytree_sharding is not None else None,
        out_shardings=pytree_sharding,
    )(pytree)


def get_mesh(n_data_parallel:int=1, n_model_parallel:int=-1, devices:list[jax.Device]=None) -> jax.sharding.Mesh:
    """
    Creates a JAX mesh for data and model parallelism.

    Args:
        n_data_parallel (int): Number of data parallel devices. Default is 1.
        n_model_parallel (int): Number of model parallel devices. Default is -1, which uses all available devices.
        devices (list[jax.Device]): List of JAX devices to use. If None, uses all available devices.

    Returns:
        jax.sharding.Mesh: A JAX mesh with the specified data and model parallel dimensions.
    """

    if devices is None:
        devices = jax.devices()

    device_count = len(devices)

    if n_data_parallel == -1:
        n_data_parallel = device_count

    if n_model_parallel == -1:
        n_model_parallel = device_count

    devices = jax.experimental.mesh_utils.create_device_mesh(
        mesh_shape=(n_data_parallel, n_model_parallel),
        devices=devices,
    )
    return jax.sharding.Mesh(devices, ["data", "model"])
