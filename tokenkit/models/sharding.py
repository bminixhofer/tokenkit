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

def get_shard_patterns(kind):
    return SHARD_PATTERNS.get(kind, {})


def get_sharding_fn(shard_patterns, mesh):
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


def to_global_array(pytree, pytree_sharding=None):
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
    if jax.process_count() == 1:
        return pytree

    return jax.tree.map(lambda x: x[0], process_allgather(pytree))


def to_devices(pytree, pytree_sharding=None, dtype=None):
    # TODO: handle non-numpy inputs?
    pytree = to_global_array(pytree, pytree_sharding)

    return jax.jit(
        lambda x: x if dtype is None else jax.tree.map(lambda x: x.astype(dtype), x),
        in_shardings=(pytree_sharding,) if pytree_sharding is not None else None,
        out_shardings=pytree_sharding,
    )(pytree)


def get_mesh(n_data_parallel=1, n_model_parallel=-1, devices=None):
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
