import math
import logging
from pathlib import Path
import json
from pprint import pformat
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
)


import jax
import jax.numpy as jnp
from flax import serialization, traverse_util
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tokenkit.models.hypernet import Hypernet, HypernetConfig
from tokenkit.models import param, lora, roa
from tokenkit.hf import get_config

logger = logging.getLogger(__name__)

def save(
    path,
    params,
    param_shardings,
    mesh,
    train_mask,
    keys_to_keep={
        "hypernet",
    },
    batch_size=16,
):
    flat_keys_to_save = [
        k
        for k, trainable in traverse_util.flatten_dict(train_mask).items()
        if trainable or k[0] in keys_to_keep
    ]
    flat_params = traverse_util.flatten_dict(params)
    flat_shardings = traverse_util.flatten_dict(param_shardings)

    flat_params_to_save = {k: flat_params[k] for k in flat_keys_to_save}
    shardings_to_save = {k: flat_shardings[k] for k in flat_keys_to_save}

    none_shardings_to_save = jax.tree.map(
        lambda _: NamedSharding(mesh, P()), shardings_to_save
    )

    keys = list(flat_params_to_save.keys())
    n_batches = math.ceil(len(keys) / batch_size)

    all_flat_out_params = {}

    for i in range(n_batches):
        batch_keys = keys[i * batch_size : (i + 1) * batch_size]

        flat_device_params = jax.jit(
            lambda x: x,
            in_shardings=([shardings_to_save[k] for k in batch_keys],),
            out_shardings=[none_shardings_to_save[k] for k in batch_keys],
        )([flat_params_to_save[k] for k in batch_keys])

        for key, value in zip(batch_keys, flat_device_params):
            all_flat_out_params[key] = jax.device_get(value)
            value.delete()

    if jax.process_index() == 0:
        open(path, "wb").write(
            serialization.msgpack_serialize(
                traverse_util.unflatten_dict(all_flat_out_params), in_place=True
            )
        )

    multihost_utils.sync_global_devices("saved checkpoint")


def export(
    mesh,
    checkpoint_dir: str | Path,
    overwrite_args: str | None = None,
):
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_args = OmegaConf.load(checkpoint_dir / "args.yaml")

    if overwrite_args is not None:
        ckpt_args = OmegaConf.merge(
            ckpt_args,
            OmegaConf.create(json.loads(overwrite_args))
        )

    logger.info("Exporting with checkpoint args:")
    logger.info(pformat(ckpt_args))
    
    params = serialization.msgpack_restore(
        open(checkpoint_dir / "params.msgpack", "rb").read()
    )

    config = get_config(checkpoint_dir)
    config.mesh = mesh
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dtype = getattr(jnp, ckpt_args.dtype)

    model_kwargs = OmegaConf.to_object(ckpt_args.student)

    if "model" in params:
        model_params = params["model"]
        original_model_params = param.load_params(**model_kwargs)
    else:
        model_params = original_model_params = param.load_params(**model_kwargs)

    # model params may be partial at this point e.g. if trained with LoRA, merge them
    flat_merged_model_params = traverse_util.flatten_dict(original_model_params)
    flat_model_params = traverse_util.flatten_dict(model_params)

    for key in flat_model_params.keys():
        flat_merged_model_params[key] = flat_model_params[key]

    merged_model_params = traverse_util.unflatten_dict(flat_merged_model_params)

    if "model_lora" in params:
        # LoRA embeddigns may be unset - fix this here
        try:
            param.get(params["model_lora"], param.get_input_embedding_path(config.model_type))
            lora_params = params["model_lora"]
        except KeyError:
            lora_params = param.assign_embeddings(params["model_lora"], jnp.empty((0, 2)), config)

        logger.info("Materializing LoRA parameters...")
        merged_model_params = lora.materialize_lora(
            merged_model_params,
            lora_params,
            ckpt_args.model_lora_alpha,
        )

    if "model_roa" in params:
        logger.info("Materializing ROA parameters...")
        merged_model_params = roa.materialize_roa(
            merged_model_params,
            params["model_roa"],
        )

    if hasattr(ckpt_args, "hypernet"):
        # overwritten by hn outputs
        merged_model_params = param.unassign_embeddings(merged_model_params, config=config)

        n_embd = params["new_embeddings"].shape[-1]

        hypernet_config = HypernetConfig(
            hidden_size=n_embd,
            num_embeddings=1 if config.tie_word_embeddings else 2,
            max_seq_length=1,
            **ckpt_args.hypernet,
        )

        hypernet = Hypernet(config=hypernet_config, dtype=dtype)
        hypernet_fn = hypernet.apply

        def predict_embeddings(params):  # TODO: add indices for subsampling
            embeddings = params["new_embeddings"]

            predicted_embeddings = hypernet_fn(
                params["hypernet"],
                embeddings[:, None, :, :],
                jnp.ones((embeddings.shape[0], 1), dtype=bool),
                jnp.arange(embeddings.shape[0], dtype=jnp.int32),
            )

            return predicted_embeddings

        embeddings = jax.device_get(predict_embeddings(params))
        embeddings = embeddings.copy()  # not writeable otherwise

        # remove padding
        config.vocab_size = len(tokenizer)
        embeddings = embeddings[: len(tokenizer)]  # remove padding

        merged_model_params = param.assign_embeddings(merged_model_params, embeddings, config=config)

    model = FlaxAutoModelForCausalLM.from_config(config)

    return model, merged_model_params, tokenizer, config, ckpt_args