from dataclasses import dataclass
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    AutoModelForCausalLM,
)
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
from omegaconf import OmegaConf
from flax import serialization, traverse_util
from pathlib import Path
from pickle import UnpicklingError
from flax.serialization import from_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from tokenkit.models.hypernet import Hypernet, HypernetConfig
from tokenkit.models import param, lora, sharding
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.hf import get_config
from tokenkit import gcs_utils, utils, constants
import json
import os
import torch
from pprint import pformat
import logging

logger = logging.getLogger(__name__)


# transformers `load_flax_checkpoint_in_pytorch_model` does not support custom models
# so we patch it here
def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path, flax_cls):
    """Load flax checkpoints in a PyTorch model"""
    flax_checkpoint_path = os.path.abspath(flax_checkpoint_path)
    logger.info(f"Loading Flax weights from {flax_checkpoint_path}")

    # load flax weight dict
    if flax_checkpoint_path.endswith(".safetensors"):
        from safetensors.flax import load_file as safe_load_file

        flax_state_dict = safe_load_file(flax_checkpoint_path)
        flax_state_dict = unflatten_dict(flax_state_dict, sep=".")
    else:
        with open(flax_checkpoint_path, "rb") as state_f:
            try:
                flax_state_dict = from_bytes(flax_cls, state_f.read())
            except UnpicklingError:
                raise EnvironmentError(f"Unable to convert {flax_checkpoint_path} to Flax deserializable object. ")

    return load_flax_weights_in_pytorch_model(model, flax_state_dict)



@dataclass
class Args:
    checkpoint: str = "outputs/patch"
    output: str = "outputs/export"
    use_cpu: bool = True
    tmp_save_dir: str = "/tmp/tokenkit/"
    with_pt: bool = False
    expand_input_ids_model: str | None = None
    expand_input_ids_tokenizer: str | None = None
    overwrite_args: str | None = None


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    tmp_checkpoint_dir = Path(args.tmp_save_dir) / "checkpoint"
    tmp_output_dir = Path(args.tmp_save_dir) / "output"

    tmp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    else:
        mesh = sharding.get_mesh()

    if gcs_utils.is_gcs_path(args.checkpoint):
        checkpoint_bucket, checkpoint_blob = gcs_utils.parse_gcs_path(args.checkpoint)
        checkpoint_dir = tmp_checkpoint_dir

        for filename in ["args.yaml", "params.msgpack", "config.json", "tokenizer.json", "tokenizer_config.json"]:
            gcs_utils.download_from_gcs(checkpoint_bucket, f"{checkpoint_blob}/{filename}", checkpoint_dir / filename)
    else:
        checkpoint_dir = Path(args.checkpoint)

    ckpt_args = OmegaConf.load(checkpoint_dir / "args.yaml")
    if args.overwrite_args is not None:
        ckpt_args = OmegaConf.merge(
            ckpt_args, OmegaConf.create(json.loads(args.overwrite_args))
        )

    logger.info("Using checkpoint args:")
    logger.info(pformat(ckpt_args))

    params = serialization.msgpack_restore(
        open(checkpoint_dir / "params.msgpack", "rb").read()
    )

    config = get_config(checkpoint_dir)
    config.mesh = mesh
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dtype = getattr(jnp, ckpt_args.dtype)

    n_embd = params["new_embeddings"].shape[-1]

    hypernet_config = HypernetConfig(
        hidden_size=n_embd,
        num_embeddings=1 if config.tie_word_embeddings else 2,
        max_seq_length=1,
        **ckpt_args.hypernet,
    )
    hypernet = Hypernet(config=hypernet_config, dtype=dtype)
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
    # assigned later
    merged_model_params = param.unassign_embeddings(merged_model_params, config=config)

    if "model_lora" in params:
        logger.info("Materializing LoRA parameters...")
        merged_model_params = lora.materialize_lora(
            merged_model_params,
            params["model_lora"],
            ckpt_args.model_lora_alpha,
        )

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

    model_to_save = FlaxAutoModelForCausalLM.from_config(config)
    if gcs_utils.is_gcs_path(args.output):
        output_dir = tmp_output_dir
    else:
        output_dir = Path(args.output)

    del config.mesh

    # from_flax does not work with multiple shards so it is more convenient to save the model as a single shard
    model_to_save.save_pretrained(
        output_dir, params=merged_model_params, max_shard_size="100GB"
    )

    if args.with_pt:
        if args.expand_input_ids_model is not None:
            byteify_tokenizer = load_byteify_tokenizer(ckpt_args.target_tokenizer_name)
            expand_tokenizer = load_byteify_tokenizer(args.expand_input_ids_tokenizer)

            expand_input_ids_dict = utils.get_expand_input_ids_dict(
                byteify_tokenizer,
                expand_tokenizer.get_vocab(),
                max_length=constants.EXPAND_INPUT_IDS_MAX_LENGTH,
            )

            config.expand_input_ids = True
            config.expand_input_ids_maxlen = constants.EXPAND_INPUT_IDS_MAX_LENGTH
            config.expand_input_ids_vocab_size = len(expand_tokenizer)
            # make json serializable - will be deserialized in PT model init
            config.expand_input_ids_dict = (
                {",".join([str(n) for n in k]): int(v) for k, v in expand_input_ids_dict[0].items()},
                [int(n) for n in expand_input_ids_dict[1]],
            )

        pt_model = AutoModelForCausalLM.from_config(config)
        pt_model = load_flax_checkpoint_in_pytorch_model(pt_model, output_dir / "flax_model.msgpack", type(model_to_save))

        # set expansion embedding data
        if args.expand_input_ids_model is not None:
            expand_input_ids_model_config = get_config(args.expand_input_ids_model)
            expand_input_ids_model_params = param.load_params(pretrained_model_name_or_path=args.expand_input_ids_model)
            expand_input_ids_embeddings = param.get(
                expand_input_ids_model_params,
                param.get_input_embedding_path(expand_input_ids_model_config.model_type),
            )

            pt_model.model.expand_embed_tokens.weight.data[:] = torch.from_numpy(expand_input_ids_embeddings)

        pt_model.save_pretrained(output_dir)
    else:
        pt_model = None

        if args.expand_input_ids_model is not None:
            raise ValueError("expand_input_ids_model is not supported when with_pt is False")

    config.auto_map = {
        "AutoConfig": f"configuration_{config.model_type}.{type(config).__name__}",
        "FlaxAutoModelForCausalLM": f"modelling_flax_{config.model_type}.{type(model_to_save).__name__}"
    }

    if pt_model is not None:
        config.auto_map["AutoModelForCausalLM"] = f"modelling_{config.model_type}.{type(pt_model).__name__}"

    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    if gcs_utils.is_gcs_path(args.output):
        output_bucket, output_blob = gcs_utils.parse_gcs_path(args.output)
        for filename in ["config.json", "flax_model.msgpack", "tokenizer.json", "tokenizer_config.json"]:
            gcs_utils.upload_to_gcs(output_bucket, output_dir / filename, f"{output_blob}/{filename}")