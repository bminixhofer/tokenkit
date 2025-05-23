"""
Example Usage:

python3 scripts/eval_lockstep.py models=llama_qwen +eval.limit=100
"""

import logging
from pathlib import Path
from pprint import pformat, pprint
from dataclasses import dataclass, asdict
import os
import yaml
import datasets
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from transformers import FlaxAutoModelForCausalLM

from tokenkit import parse_args
from tokenkit.hf import get_config
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.eval import evaluate_lockstep
from tokenkit.models import param, sharding

logger = logging.getLogger(__name__)

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
    True  # careful about this, required for lm_eval
)

@dataclass
class EvalLockstepScriptArgs:
    combine_strategy: str
    models: list[parse_args.ModelArgs]
    eval: parse_args.EvalArgs
    baseline_mined_mapping_paths: list[str] | None = None
    output: str | None = None
    pad_to_multiple_of: int = 128
    use_cpu: bool = False


def pad_embeddings(embeddings, tokenizer):
    n_embed_diff = len(tokenizer) - len(embeddings)

    embeddings_mean = embeddings.mean(0)
    embeddings_std = embeddings.std(0)

    return np.concatenate(
        [
            embeddings,
            np.random.normal(
                size=(n_embed_diff, *embeddings.shape[1:]),
            )
            * embeddings_std[None]
            + embeddings_mean[None],
        ]
    )


def main(args: EvalLockstepScriptArgs) -> None:
    logger.info(pformat(args))

    eval_kwargs = asdict(args.eval)

    if args.output is not None:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "args.yaml", "w") as f:
            yaml.dump(asdict(args), f)
    else:
        output_dir = None

    if args.use_cpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = sharding.get_mesh(devices=jax.devices("cpu"))
    else:
        mesh = sharding.get_mesh()

    all_models = []
    all_configs = []
    all_params = []
    all_tokenizers = []
    all_logit_masks = []

    eval_kwargs.pop("add_bos")
    all_add_bos = []

    for model_idx, model_kwargs in enumerate(args.models):
        print("Loading model...")

        config = get_config(model_kwargs["pretrained_model_name_or_path"])

        config.max_length = eval_kwargs["lengths"][-1]
        config.mesh = mesh

        tokenizer = load_byteify_tokenizer(model_kwargs.pop("tokenizer_name"))

        model = FlaxAutoModelForCausalLM.from_config(config, _do_init=False)
        params = param.load_params(
            pretrained_model_name_or_path=model_kwargs["pretrained_model_name_or_path"]
        )

        input_embeddings = param.get(
            params, param.get_input_embedding_path(config.model_type)
        )

        if len(input_embeddings) < len(tokenizer):
            print("Padding input embeddings...")
            input_embeddings = pad_embeddings(input_embeddings, tokenizer)

        if not config.tie_word_embeddings:
            output_embeddings = param.get(
                params, param.get_output_embedding_path(config.model_type)
            )
            print("Padding output embeddings...")
            output_embeddings = pad_embeddings(output_embeddings.T, tokenizer).T
        else:
            output_embeddings = None

        n_overflow = input_embeddings.shape[0] % args.pad_to_multiple_of
        if n_overflow > 0:
            n_pad = args.pad_to_multiple_of - n_overflow
        else:
            n_pad = 0

        input_embeddings = np.pad(
            input_embeddings,
            ((0, n_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        if output_embeddings is not None:
            output_embeddings = np.pad(
                output_embeddings,
                ((0, 0), (0, n_pad)),
                mode="constant",
                constant_values=0,
            )
        logit_mask = np.zeros((input_embeddings.shape[0],), dtype=bool)
        logit_mask[: model.config.vocab_size] = True
        model.config.vocab_size = input_embeddings.shape[0]

        params = param.put(
            params, param.get_input_embedding_path(config.model_type), input_embeddings
        )
        if output_embeddings is not None:
            params = param.put(
                params,
                param.get_output_embedding_path(config.model_type),
                output_embeddings,
            )

        shard_patterns = sharding.get_shard_patterns(config.model_type)
        param_shardings = sharding.get_sharding_fn(shard_patterns, mesh)(
            {"params": params}
        )["params"]
        params = sharding.to_devices(params, param_shardings, dtype=jnp.float32)

        multihost_utils.sync_global_devices("loaded weights")

        if args.baseline_mined_mapping_paths is not None:
            if args.baseline_mined_mapping_paths[model_idx] is not None:
                config.mined_mapping = np.load(
                    Path(args.baseline_mined_mapping_paths[model_idx]) / "mined_mapping.npy"
                )
            else:
                config.mined_mapping = None

        all_models.append(model)
        all_configs.append(config)
        all_params.append(params)
        all_tokenizers.append(tokenizer)
        all_logit_masks.append(logit_mask)
        all_add_bos.append(model_kwargs["add_bos"])

    # static combine fn for the moment
    def combine_fn(hidden_states, logits, combine_params, output_embeddings):
        if args.combine_strategy == "mean_prob":
            aggregated_probs = None
            for model_logits in logits:
                model_probs = jax.nn.softmax(model_logits, axis=-1)
                if aggregated_probs is None:
                    aggregated_probs = model_probs
                else:
                    aggregated_probs += model_probs

            aggregated_probs /= len(logits)
            return jnp.log(aggregated_probs)
        elif args.combine_strategy == "mean_logits":
            aggregated_logits = None
            for model_logits in logits:
                if aggregated_logits is None:
                    aggregated_logits = model_logits
                else:
                    aggregated_logits += model_logits

            aggregated_logits /= len(logits)
            return aggregated_logits
        else:
            raise ValueError(f"Unknown combine strategy: {args.combine_strategy}")

    results = evaluate_lockstep(
        models=all_models,
        configs=all_configs,
        params=all_params,
        tokenizers=all_tokenizers,
        logit_masks=all_logit_masks,
        add_bos=all_add_bos,
        combine_fn=combine_fn,
        combine_params={},
        jaxlm_kwargs={"precompile": not args.use_cpu},
        output=output_dir,
        **eval_kwargs,
    )

    if jax.process_index() == 0:
        pprint(results[0])


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    main(parse_args.parse_args(EvalLockstepScriptArgs))