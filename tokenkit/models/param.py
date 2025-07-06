"""
Tokenkit utilities to work with PyTrees of parameters.

Tokenkit parameter paths frequently use a dot-separated string notation (e.g. "model.embed_tokens.embedding").
This notation is parsed on-demand to access the parameters in a PyTree structure.

The utilities here should work with both parameter trees of JAX and NumPy arrays.
"""

import copy
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization, traverse_util
from transformers import AutoConfig
from transformers.utils.hub import cached_file


def get_input_embedding_path(model_type: str) -> str:
    """
    Returns the path to the input embedding layer for a given model type. Must be present.

    Args:
        model_type (str): The type of the model, e.g. "gpt2", "roberta", "xlm-roberta", etc.

    Returns:
        str: The path to the input embedding layer in the model's parameter structure.
    """

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


def get_output_embedding_path(model_type: str) -> str | None:
    """
    Returns the path to the output embedding layer for a given model type, or None if embeddings are tied.

    Args:
        model_type (str): The type of the model, e.g. "gpt2", "roberta", "xlm-roberta", etc.

    Returns:
        str | None: The path to the output embedding layer in the model's parameter structure if present.
    """

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


def get_layer_path(model_type: str) -> str:
    """
    Returns the path to the layer dict of a model for a given model type.

    Args:
        model_type (str): The type of the model, e.g. "gpt2"

    Returns:
        str: The path to the layer dict in the model's parameter structure.
    """

    return {
        "gemma2": "model.layers",
        "gemma3": "model.layers",
        "gpt2": "transformer.h",
        "llama": "model.layers",
        "tpu_llama": "model.layers",
        "tpu_gemma2": "model.layers",
        "tpu_gemma3": "model.layers",
    }[model_type]


def load_params(**kwargs) -> dict:
    """
    Returns parameters as a Pytree of NumPy arrays from a pretrained model on the HF hub or locally.
    The parameters need to be put on TPU / converted to JAX arrays later.

    The parameter source must have a `flax_model.msgpack` file or a `flax_model.msgpack.index.json` file.
    If the index file is present, it will be used to load the parameters from multiple files
    (e.g. for large models that are split into multiple shards).

    Args:
        **kwargs: Keyword arguments to pass to `AutoConfig.from_pretrained` and `cached_file`.
            Must include `pretrained_model_name_or_path` and optionally e.g. `revision`.

    Returns:
        dict: A Pytree of NumPy arrays representing the model parameters.
    """

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


def put(pytree, path: str, value: dict | Any) -> dict:
    """
    Puts an array or subtree into a PyTree at the specified path out-of-place.

    Args:
        pytree (dict): The PyTree to modify.
        path (str): The dot-separated path where the value should be inserted.
        value (dict | jnp.ndarray): The value to insert.

    Returns:
        dict: A new PyTree with the value inserted at the specified path.
    """

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


def pop(pytree: dict, path: str)-> tuple[dict, Any]:
    """
    Pops a value or subtree from a PyTree at the specified path out-of-place.
    
    Args:
        pytree (dict): The PyTree to modify.
        path (str): The dot-separated path from which to pop the value.

    Returns:
        (params, popped_value): A tuple containing the modified PyTree and the popped value.
            If the path does not exist, returns the original pytree and None.
    """

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


def get(pytree: dict, path: str) -> Any:
    """
    Gets a value or subtree from a PyTree at the specified path.

    Args:
        pytree (dict): The PyTree to search.
        path (str): The dot-separated path to the value.

    Returns:
        The value at the specified path, or a subtree if the path is a prefix.

    Raises:
        KeyError: If the path does not exist in the PyTree.
    """

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


def keys(pytree: dict) -> list[str]:
    """Returns a list of all keys in the flattened PyTree as dot-separated strings."""

    return [".".join(x) for x in traverse_util.flatten_dict(pytree).keys()]


def assign_embeddings(model_params: dict, embeddings: np.ndarray | jnp.ndarray, config) -> dict:
    """
    Assigns embeddings to the input and output embedding layers in the model parameters.
    
    Embeddings are expected to use a (vocab_size, n_embeddings, embedding_dim) representation.
    (where n_embeddings=1 for tied embeddings and n_embeddings=2 for untied embeddings).

    Args:
        model_params (dict): The model parameters as a PyTree.
        embeddings (np.ndarray | jnp.ndarray): The embeddings to assign, shape should be (vocab_size, n_embeddings, embedding_dim).
            The first slice along the second dimension is used for input embeddings,
            and the last slice is used for output embeddings if `config.tie_word_embeddings` is False.
        config: The model configuration containing the model type and whether to tie embeddings.

    Returns:
        dict: The updated model parameters with the embeddings assigned.
    """

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


def unassign_embeddings(model_params: dict, config) -> dict:
    """
    Unassigns input and output embeddings from the model parameters and deletes their buffers.

    This is useful for freeing up memory when embeddings are not needed anymore.

    Args:
        model_params (dict): The model parameters as a PyTree.
        config: The model configuration containing the model type.

    Returns:
        dict: The updated model parameters with the embeddings removed.
    """

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


def stack_embeddings(model_params: dict, config, pop_embeddings: bool = False) -> tuple[np.ndarray, dict]:
    """
    Returns a stacked array of input and output embeddings from the model parameters
    of shape (vocab_size, n_embeddings, embedding_dim).
    (n_embeddings=1 for tied embeddings, n_embeddings=2 for untied embeddings).

    Args:
        model_params (dict): The model parameters as a PyTree.
        config: The model configuration containing the model type and whether to tie embeddings.
        pop_embeddings (bool): If True, removes the embeddings from the model parameters.

    Returns:
        tuple: A tuple containing:
            - embeddings (np.ndarray): The stacked embeddings of shape (vocab_size, n_embeddings, embedding_dim).
            - model_params (dict): The updated model parameters with embeddings removed if `pop_embeddings` is True.
                (otherwise  the original model parameters are returned).

    Raises:
        KeyError: If the input or output embedding paths are not found in the model parameters.
    """

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

        if isinstance(input_embeddings, jnp.ndarray):
            embeddings = jnp.stack(
                [input_embeddings, output_embeddings.T], axis=1
            )
        else:
            embeddings = np.stack(
                [input_embeddings, output_embeddings.T], axis=1
            )

    if pop_embeddings:
        model_params = unassign_embeddings(model_params, config)

    return embeddings, model_params


def get_num_layers(config) -> int:
    """
    Returns the number of layers in the model configuration.

    The embedding layer is *not* included in the count (although it is typically treated as an extra layer in this codebase).

    Args:
        config: The model configuration object.

    Returns:
        int: The number of layers in the model.

    Raises:
        ValueError: If the number of layers cannot be determined from the configuration.
    """

    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):  # gpt2
        return config.n_layer
    else:
        raise ValueError("Could not determine number of layers from config")


def set_num_layers(config, num_layers: int):
    """
    Sets the number of layers in the model configuration in-place.

    Only updates the config, the param tree should be updated separately.

    Args:
        config: The model configuration object.
        num_layers (int): The number of layers to set in the configuration.
    """

    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_layers
    elif hasattr(config, "n_layer"):  # gpt2
        config.n_layer = num_layers
    else:
        raise ValueError("Could not determine number of layers from config")


def get_layer_n_mask(model_params: dict, config, layer_idx: int) -> dict:
    """
    Get a Pytree of the same shape as `model_params` with boolean leaves which are 
    True if the parameters at this position are within the layer with index `layer_idx`.

    This is useful for masking parameters of a specific layer in the model.

    Args:
        model_params (dict): The model parameters as a PyTree.
        config: The model configuration containing the model type.
        layer_idx (int): The index of the layer to mask.

    Returns:
        A Pytree with the same structure as `model_params`, where each leaf
            is a boolean indicating whether the parameter belongs to the specified layer.
            True if the parameter is part of the layer, False otherwise.
    """

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


def strip_layers(
    model_params: dict,
    config,
    n_keep: int = 1,
    mode: str = "start",
    offset: int = 0,
    layer_multiplier: np.ndarray | jnp.ndarray | None = None
) -> dict:
    """
    Strips layers from the model parameters, keeping only a consecutive chunk of layers.
    Also modifies the passed config in-place to reflect the new number of layers.

    Args:
        model_params (dict): The model parameters as a PyTree.
        config: The model configuration containing the model type.
        n_keep (int): The number of layers to keep.
        mode (str): The mode of stripping layers. Can be "start" or "end".
            - "start": keeps the first `n_keep` layers and removes the rest.
            - "end": keeps the last `n_keep` layers and removes the rest.
        offset (int): The number of layers to skip before starting to keep layers. Examples:
            - If `offset=1` and `mode="start"`, the first layer will be skipped,
            and the next `n_keep` layers will be kept.
            - If  `offset=1` and `mode="end"`, the last layer will be skipped,
            and  `n_keep` layers before will be kept.
        layer_multiplier (np.ndarray | jnp.ndarray | None): Optional multiplier for the parameter magnitudes of the layers to keep.
            If provided, must be of shape (n_keep,).

    Returns:
        dict: The updated model parameters with the specified layers stripped.        
    """

    # check how many layers we have params for in total, +1 since zero-indexed layers
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
