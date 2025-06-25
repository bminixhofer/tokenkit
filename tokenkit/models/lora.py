import jax
import jax.numpy as jnp
import numpy as np
import regex as re

from tokenkit import utils

LORA_PATTERNS = {
    "llama": [
        ".*self_attn.(q_proj|k_proj|v_proj).kernel",
        ".*self_attn.o_proj.kernel",
        ".*mlp.down_proj.kernel",
        ".*mlp.up_proj.kernel",
        ".*mlp.gate_proj.kernel",
    ],
    "gemma2": [
        ".*self_attn.(q_proj|k_proj|v_proj).kernel",
        ".*self_attn.o_proj.kernel",
        ".*mlp.down_proj.kernel",
        ".*mlp.up_proj.kernel",
        ".*mlp.gate_proj.kernel",
    ],
}
LORA_PATTERNS["tpu_llama"] = LORA_PATTERNS["llama"]
LORA_PATTERNS["tpu_gemma2"] = LORA_PATTERNS["gemma2"]
LORA_PATTERNS["gemma3"] = LORA_PATTERNS["gemma2"]
LORA_PATTERNS["tpu_gemma3"] = LORA_PATTERNS["gemma2"]


def init_lora_params(args, params, model_type, seed, dtype=jnp.float32):
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(jax.random.PRNGKey(seed))

    lora_patterns = LORA_PATTERNS[model_type]
    lora_rank = args.model_lora_rank
    stddev = 1.0 / lora_rank

    def init_lora(path, param):
        path_tuple = tuple(str(utils.keystr(x)) for x in path)
        path = ".".join(path_tuple)

        lora_params = np.array([])  # indicates no lora params

        for key in lora_patterns:
            if re.match(key, path):
                assert len(param.shape) == 2
                b_dim, a_dim = param.shape

                b = np.zeros((b_dim, lora_rank), dtype=dtype)
                a = jax.device_get(
                    jax.random.normal(next(key_it), (lora_rank, a_dim), dtype=dtype)
                    * stddev
                )
                lora_params = {"a": a, "b": b}

        return lora_params

    return jax.tree_util.tree_map_with_path(init_lora, params)


def materialize_lora(param_tree, lora_param_tree, alpha):
    def materialize(param, lora_params):
        if not isinstance(lora_params, dict):
            assert lora_params.shape[0] == 0
            return param

        a, b = lora_params["a"], lora_params["b"]
        scale = alpha / b.shape[-1]

        return (param + scale * b @ a).astype(param.dtype)

    return jax.tree.map(materialize, param_tree, lora_param_tree)


# NOTE: not clear if this is save w.r.t. rounding errors. probably not? dangerous.
# NOTE: update: no instability so far, seems safe in fp32. but still dangerous.
def dematerialize_lora(param_tree, lora_param_tree, alpha):
    def dematerialize(param, lora_params):
        if not isinstance(lora_params, dict):
            return param

        a, b = lora_params["a"], lora_params["b"]
        scale = alpha / b.shape[-1]

        return (param - scale * b @ a).astype(param.dtype)

    return jax.tree.map(dematerialize, param_tree, lora_param_tree)
