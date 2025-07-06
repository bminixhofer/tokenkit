"""
Tokenkit LoRA implementation.
"""

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
    """
    Initializes LoRA a and b matrices.
    LoRA positions are hardcoded to Q/K/V/O projections and Up/Down/Gate projections.

    Args:
        args: training arguments.
        params: model parameters.
        model_type: HF model type, e.g. "llama", "gemma2".
        seed: random seed for initialization.

    Returns:
        A pytree of LoRA parameters with the same leaves as `params`, where every leaf is either:
            - (i) an empty array (indicating no LoRA params for this leaf),
            - (ii) a dict with keys "a" and "b", where:
                - "a" is a matrix of shape (lora_rank, a_dim) initialized
                  with random values scaled by 1/lora_rank,
                - "b" is a matrix of shape (b_dim, lora_rank) initialized
                  with zeros, where b_dim and a_dim are the dimensions of the original parameter.
    """

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
    """
    Materializes (adds) LoRA parameters into the original parameters.

    Args:
        param_tree: pytree of original model parameters.
        lora_param_tree: pytree of LoRA parameters, where each leaf is either:
            - (i) an empty array (indicating no LoRA params for this leaf),
            - (ii) a dict with keys "a" and "b", where:
                - "a" is a matrix of shape (lora_rank, a_dim),
                - "b" is a matrix of shape (b_dim, lora_rank).
        alpha: scaling factor for LoRA parameters.

    Returns:
        A pytree of modified parameters with LoRA parameters materialized (added).
    """

    def materialize(param, lora_params):
        if not isinstance(lora_params, dict):
            assert lora_params.shape[0] == 0
            return param

        a, b = lora_params["a"], lora_params["b"]
        scale = alpha / b.shape[-1]

        return (param + scale * b @ a).astype(param.dtype)

    return jax.tree.map(materialize, param_tree, lora_param_tree)


def dematerialize_lora(param_tree, lora_param_tree, alpha):
    """
    Dematerializes (removes) LoRA parameters from the original parameters.

    Args:
        param_tree: pytree of original model parameters.
        lora_param_tree: pytree of LoRA parameters, where each leaf is either:
            - (i) an empty array (indicating no LoRA params for this leaf),
            - (ii) a dict with keys "a" and "b", where:
                - "a" is a matrix of shape (lora_rank, a_dim),
                - "b" is a matrix of shape (b_dim, lora_rank).
        alpha: scaling factor for LoRA parameters.

    Returns:
        A pytree of restored original parameters.

    IMPORTANT: dematerialization does not exactly restore the original parameters due to
    floating-point imprecision. However, the introduced error appears sufficiently small.
    This makes Mat/Demat useful so that we do not need to store two copies of the model parameters
    (original and LoRA-modified) in memory at the same time.
    """

    def dematerialize(param, lora_params):
        if not isinstance(lora_params, dict):
            return param

        a, b = lora_params["a"], lora_params["b"]
        scale = alpha / b.shape[-1]

        return (param - scale * b @ a).astype(param.dtype)

    return jax.tree.map(dematerialize, param_tree, lora_param_tree)
