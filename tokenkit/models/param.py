import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util
from transformers import AutoConfig
from transformers.utils.hub import cached_file


def get_input_embedding_path(model_type):
    return {
        "gpt2": "transformer.wte.embedding",
        "roberta": "roberta.embeddings.word_embeddings.embedding",
        "xlm-roberta": "roberta.embeddings.word_embeddings.embedding",
        "xglm": "model.embed_tokens.embedding",
        "mistral": "model.embed_tokens.embedding",
        "llama": "model.embed_tokens.embedding",
        "tpu_llama": "model.embed_tokens.embedding",
        "gemma": "model.embed_tokens.embedding",
        "gemma2": "model.embed_tokens.embedding",
        "gemma3": "model.embed_tokens.embedding",
        "tpu_gemma2": "model.embed_tokens.embedding",
        "tpu_gemma3": "model.embed_tokens.embedding",
    }[model_type]


def get_output_embedding_path(model_type):
    return {
        "gpt2": "lm_head.kernel",
        "roberta": None,
        "xlm-roberta": None,
        "xglm": None,
        "mistral": "lm_head.kernel",
        "llama": "lm_head.kernel",
        "tpu_llama": "lm_head.kernel",
        "gemma": "lm_head.kernel",
        "gemma2": "lm_head.kernel",
        "gemma3": "lm_head.kernel",
        "tpu_gemma2": "lm_head.kernel",
        "tpu_gemma3": "lm_head.kernel",
    }[model_type]


def get_layer_path(model_type):
    return {
        "gemma2": "model.layers",
        "gemma3": "model.layers",
        "gpt2": "transformer.h",
        "llama": "model.layers",
        "tpu_llama": "model.layers",
        "tpu_gemma2": "model.layers",
        "tpu_gemma3": "model.layers",
    }[model_type]


def load_params(**kwargs):
    kwargs = copy.copy(kwargs)
    config = AutoConfig.from_pretrained(**kwargs)
    path = kwargs.pop("pretrained_model_name_or_path")
    embedding_path = kwargs.pop("embedding_path", None)

    try:
        index = cached_file(path, "flax_model.msgpack.index.json", **kwargs)
    except OSError:
        index = None

    if index is not None:
        index = json.load(open(index))
        files = [
            cached_file(path, x, **kwargs) for x in set(index["weight_map"].values())
        ]
    else:
        files = [cached_file(path, "flax_model.msgpack", **kwargs)]

    flat_params = {}
    for x in files:
        flat_params.update(
            traverse_util.flatten_dict(
                serialization.msgpack_restore(open(x, "rb").read())
            )
        )

    params = traverse_util.unflatten_dict(flat_params)

    if embedding_path is not None:
        embeddings = np.load(embedding_path)
        params = put(
            params, get_input_embedding_path(config.model_type), embeddings[:, 0]
        )
        if embeddings.shape[1] > 1:
            params = put(
                params, get_output_embedding_path(config.model_type), embeddings[:, 1].T
            )

    return params


def put(pytree, path, value):
    path_tuple = tuple(path.split("."))

    flat_pytree = traverse_util.flatten_dict(pytree)

    if isinstance(value, dict):
        # Flatten the value dict and insert each subkey at the correct subpath
        value_flat = traverse_util.flatten_dict(value)
        for subkey, subval in value_flat.items():
            full_key = path_tuple + subkey
            if full_key in flat_pytree and isinstance(flat_pytree[full_key], jnp.ndarray):
                flat_pytree[full_key] = flat_pytree[full_key].at[:].set(subval)
            else:
                flat_pytree[full_key] = subval
    else:
        if path_tuple in flat_pytree and isinstance(flat_pytree[path_tuple], jnp.ndarray):
            flat_pytree[path_tuple] = flat_pytree[path_tuple].at[:].set(value)
        else:
            flat_pytree[path_tuple] = value

    return traverse_util.unflatten_dict(flat_pytree)


def pop(pytree, path):
    path_tuple = tuple(path.split("."))
    flat_pytree = traverse_util.flatten_dict(pytree)
    
    keys_to_pop = [k for k in flat_pytree if k[:len(path_tuple)] == path_tuple]
    if not keys_to_pop:
        return pytree, None

    subtree = {}
    direct_value = None

    for k in keys_to_pop:
        subkey = k[len(path_tuple):]
        if not subkey:
            # Exact match — store the value directly
            direct_value = flat_pytree[k]
        else:
            subtree[subkey] = flat_pytree[k]
        del flat_pytree[k]

    # Decide what to return as the popped value
    if direct_value is None:
        value = traverse_util.unflatten_dict(subtree)
    else:
        value = direct_value

    return traverse_util.unflatten_dict(flat_pytree), value


def get(pytree, path):
    path_tuple = tuple(path.split("."))
    flat = traverse_util.flatten_dict(pytree)
    # Find all keys that start with the given path
    subkeys = {k[len(path_tuple):]: v for k, v in flat.items() if k[:len(path_tuple)] == path_tuple}
    if not subkeys:
        raise KeyError(f"Path '{path}' not found in pytree.")
    if () in subkeys and len(subkeys) == 1:
        # Only a single leaf at this path
        return subkeys[()]
    # Otherwise, return the subtree
    return traverse_util.unflatten_dict(subkeys)


def keys(pytree):
    return [".".join(x) for x in traverse_util.flatten_dict(pytree).keys()]


def assign_embeddings(model_params, embeddings, config):
    model_params = put(
        model_params,
        get_input_embedding_path(config.model_type),
        embeddings[:, 0],
    )
    if not config.tie_word_embeddings:
        model_params = put(
            model_params,
            get_output_embedding_path(config.model_type),
            embeddings[:, -1].T,
        )

    return model_params


def unassign_embeddings(model_params, config):
    model_params, x = pop(model_params, get_input_embedding_path(config.model_type))
    if isinstance(x, jnp.ndarray):
        x.delete()
    if get_output_embedding_path(config.model_type):
        model_params, x = pop(
            model_params, get_output_embedding_path(config.model_type)
        )
        if isinstance(x, jnp.ndarray):
            x.delete()

    return model_params


def stack_embeddings(model_params, config, pop_embeddings=False):
    if config.tie_word_embeddings:
        input_embeddings = get(
            model_params, get_input_embedding_path(config.model_type)
        )

        embeddings = input_embeddings[:, None, :]
    else:
        input_embeddings = get(
            model_params, get_input_embedding_path(config.model_type)
        )
        try:
            output_embeddings = get(
                model_params, get_output_embedding_path(config.model_type)
            )
        except KeyError:
            output_embeddings = input_embeddings.T

        embeddings = np.stack([input_embeddings, output_embeddings.T], axis=1)

    if pop_embeddings:
        model_params = unassign_embeddings(model_params, config)

    return embeddings, model_params


def get_num_layers(config):
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):  # gpt2
        return config.n_layer
    else:
        raise ValueError("Could not determine number of layers from config")


def set_num_layers(config, num_layers):
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    elif hasattr(config, "n_layer"):  # gpt2
        config.n_layer = num_layers
    else:
        raise ValueError("Could not determine number of layers from config")


def get_layer_n_mask(model_params, config, layer_idx):
    if layer_idx < 0:
        layer_idx = get_num_layers(config) + layer_idx

    flat_params = traverse_util.flatten_dict(model_params)
    mask = {}
    subpath = f"{get_layer_path(config.model_type)}.{layer_idx}"

    for key in flat_params.keys():
        if subpath in ".".join(key):
            mask[key] = True
        else:
            mask[key] = False

    return traverse_util.unflatten_dict(mask)


def strip_layers(model_params, config, n_keep=1, mode="start", offset=0, layer_multiplier=None):
    n_layers = max(int(x) for x in get(model_params, get_layer_path(config.model_type)).keys()) + 1

    example_layer_params = get(
        model_params, f"{get_layer_path(config.model_type)}.0"
    )

    if mode == "start":
        for layer_idx in list(range(offset)) + list(range(n_keep + offset, n_layers)):
            model_params, _ = pop(
                model_params, f"{get_layer_path(config.model_type)}.{layer_idx}"
            )

        # shift the remaining layers to the start if necessary
        for layer_idx in range(n_keep):
            model_params, layer_params = pop(
                model_params, f"{get_layer_path(config.model_type)}.{layer_idx + offset}"
            )
            if layer_params is None:
                layer_params = jax.tree.map(lambda x: jnp.zeros_like(x), example_layer_params)
            elif layer_multiplier is not None:
                layer_params = jax.tree.map(
                    lambda x: x * layer_multiplier[layer_idx].astype(x.dtype), layer_params
                )

            model_params = put(
                model_params,
                f"{get_layer_path(config.model_type)}.{layer_idx}",
                layer_params,
            )
    elif mode == "end":
        first_kept_layer_idx = n_layers - n_keep - offset
        for layer_idx in list(range(first_kept_layer_idx)) + list(range(n_layers - offset, n_layers)):
            model_params, _ = pop(
                model_params, f"{get_layer_path(config.model_type)}.{layer_idx}"
            )

        # shift the remaining layers to the start
        for layer_idx in range(n_keep):
            model_params, layer_params = pop(
                model_params, f"{get_layer_path(config.model_type)}.{layer_idx + first_kept_layer_idx}"
            )

            if layer_params is None:
                layer_params = jax.tree.map(lambda x: jnp.zeros_like(x), example_layer_params)
            elif layer_multiplier is not None:
                layer_params = jax.tree.map(
                    lambda x: x * layer_multiplier[layer_idx].astype(x.dtype), layer_params
                )

            model_params = put(
                model_params,
                f"{get_layer_path(config.model_type)}.{layer_idx}",
                layer_params,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    set_num_layers(config, n_keep)

    return model_params
