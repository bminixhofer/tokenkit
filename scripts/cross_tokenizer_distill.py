"""
Run cross-tokenizer distillation.
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any
from dataclasses import dataclass, asdict, field
import yaml

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax import traverse_util, struct
from flax.training import common_utils, train_state
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import FlaxAutoModelForCausalLM

import wandb
from tokenkit import data, eval, gcs_utils, parse_args, utils
from tokenkit.hf import get_config
from tokenkit.byteify import load_byteify_tokenizer
from tokenkit.models import lora, param, sharding
from tokenkit.models.hypernet import Hypernet, HypernetConfig
from tokenkit.training import checkpoint, collators, losses, lr, opt, multitask
from tokenkit.utils import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BaselineArgs:
    divergence: str = "srkl"
    adaptive_kl_alpha: float = 0.5
    skew_lambda: float = 0.1
    teacher_temperature: float = 1.0
    kd_rate: float = 0.5
    kd_temp: float = 2.0
    gradnorm_alpha: float = 0.12
    gradnorm_lr: float = 2.5e-2


@dataclass
class CrossTokenizerDistillArgs:
    # list of losses to use, e.g. "[sft,alm_unconstrained]" to use SFT and cross-tokenizer distillation via ALM.
    losses: list[str]
    # number of steps to train for.
    steps: int
    # number of steps to warmup the learning rate linearly.
    warmup_steps: int
    # name of the experiment, used for logging and saving checkpoints.
    name: str
    # output directory for checkpoints and logs. CAREFUL: will be deleted if it exists.
    output: str
    # number of CPU workers to use for e.g. data loading.
    num_workers: int
    # interval to log training metrics at.
    log_interval: int
    # interval to sync logged training metrics to the host.
    # `sync_interval` is separate from `log_interval` since we might sometimes want to log very frequently, but not move the tensors to the host every time.
    sync_interval: int
    # interval to evaluate the model.
    eval_interval: int
    # interval to save the model checkpoint.
    save_interval: int
    # name of the target tokenizer to transfer to as a byteify spec (see https://github.com/bminixhofer/tokenkit/blob/main/docs/byteification.md).
    target_tokenizer_name: str
    # training data specification
    data: dict[str, Any]
    # hypernet configuration
    hypernet: parse_args.HypernetArgs
    # optimizer configuration
    optimizer: dict[str, Any]
    # eval configuration, e.g. which LM harness likelihood scoring/generation tasks to run.
    eval: parse_args.EvalArgs
    # student model configuration (pretrained weights path + tokenizer)
    student: parse_args.ModelArgs
    # teacher model configuration (pretrained weights path + tokenizer)
    teacher: parse_args.ModelArgs | None = None
    # baseline configuration, e.g. for MinED and DSKD. Likely only necessary to replicate experiments from a paper.
    baseline: BaselineArgs = field(default_factory=BaselineArgs)
    # lowest-precision dtype. some parameters (e.g. trainable) will always be kept in fp32.
    dtype: str = "bfloat16"
    # debug mode: run on CPU, disable optimizations.
    debug: bool = False
    # seed for e.g. randomly initialized parameters, data order.
    seed: int = 1234
    # maximum length (in tokens) of the teacher inputs.
    max_teacher_length: int = 512
    # maximum length (in tokens) of the student inputs.
    max_student_length: int = 512
    # multiple to pad to along the embedding (vocabulary) dimension.
    pad_to_multiple_of: int = 64
    # whether to eval after the first training step (useful for debugging).
    eval_at_step_zero: bool = False
    # whether to save after the first training step (useful for debugging).
    save_at_step_zero: bool = False
    # whether to skip LM harness evaluation (useful for debugging).
    skip_lm_eval: bool = False
    # output embedding mode: "preserve" to keep as is, "untie" to train input/output embeddings separately even if they were tied originally.
    output_embeddings_mode: str = "preserve"
    # whether the data is in chat template format and should be decoded as such. usually true.
    use_chat_template: bool = True
    # chat template mode, see `tokenkit.utils.preprocess_prompt` for details.
    chat_template_mode: str = "direct_encode"
    # loss mask mode (only partially supported): None to compute the loss over all input tokens, "dolly" or "openmath2" for corpus-specific prompt masking.
    loss_mask_mode: str | None = None
    # whether to use gradient checkpointing to save memory at the cost of compute. not implemeted for all models.
    gradient_checkpointing: bool = False
    # whether to analyze training step cost (FLOPS and memory) then exit, or to run the training loop.
    do_cost_analysis: bool = False
    # whether to run the training loop in dry-run mode, i.e. without actually training the model, only iterating over the data (useful for debugging).
    dry_run: bool = False
    # FSDP data parallelism axis size
    n_data_parallel: int = 1
    # FSDP model parallelism axis size
    n_model_parallel: int = 8
    # loss weights. CAREFUL: does not have an effect for `approx_gradmag*` losses at the moment since the magnitude balancing cancels out the weights.
    loss_weights: list[float] | None = None
    # loss schedules, e.g. ["linear", "cosine", "constant"] to use a linear warmup schedule for the first loss, cosine for the second, and constant for the third.
    loss_schedules: list[str] | None = None
    # how to aggregate the losses. `None` uses a simple arithmetic sum of (weighted) losses. `approx_gradmag_preserve_mag` uses GradMag (see https://arxiv.org/pdf/2503.20083).
    multitask_aggregation_fn: str | None = None
    # temperature to calculate the ALM loss with.
    binarization_temp: float = 100.0
    # DEPRECATED distillation chunk sizes. should always be one.
    distill_chunk_sizes: list[int] = field(default_factory=lambda: [1])
    # ALM loss distance function, e.g. "binary_ce" for binary cross-entropy.
    alm_diff_fn: str = "binary_ce"
    # ALM loss numerator. usually keep "chunk_count".
    distill_main_path_numerator: str = "chunk_count"
    # ALM loss denominator. usually keep "chunk_count".
    distill_main_path_denominator: str = "chunk_count"
    # model training mode: "lora" to only train LoRA adapters, "full" to train the full model instead.
    train_model_mode: str = "lora"
    # LoRA rank.
    model_lora_rank: int = 64
    # LoRA alpha scaling factor.
    model_lora_alpha: int = 64
    # whether to train or freeze the input embeddings. If a hypernet is used, and train_embeddings=False, the embeddings will still be
    # updated (through the hypernet predictions), but not trained directly.
    train_embeddings: bool = True
    # tokens to add to the target tokenizer.
    tokens_to_add: list[str] | None = None
    # which latents to align, e.g. "last_hidden_state" for the last hidden state of the model.
    latents_to_align: str = "last_hidden_state"
    # loss function to use for latent alignment.
    latents_normalization: str = "l2_channelwise"
    # whether to use a naive or a more complex chunking strategy for the latents, usually "naive".
    latents_chunks: str = "naive"
    # whether to project the latents, necessary if using the latents loss in a non-self-distillation setting.
    # CAREFUL: we have not observed it helping in this setting, it is probably better to disable the latent loss when not self-distilling.
    latents_do_project: bool = False
    # ALM loss mode, "append_space" means debiasing, should usually be added (see https://arxiv.org/abs/2503.20083).
    # "merge_by_space_prob" means joining chunks such that the endings have a debiasing probability above the threshold, should usually be added.
    alm_mode: str = "merge_by_space_prob+append_space"
    # which bytes to assume to not cross token boundaries for debiasing.
    space_mask_mode: str = "space+tab+newline+special"
    # path to tokenizer data directory, needed for e.g. the MinED baseline, but not necessary for the default ALM setting.
    tokenizer_pair_data_path: str | None = None
    # chunk threshold, used for `merge_by_space_prob` chunk combination.
    tokenizer_pair_bias_threshold: float = 0.1
    # whether to expand the input IDs for conversion to the byte-level (see "Adjustments for Transfer to Bytes" section in https://arxiv.org/abs/2503.20083).
    expand_input_ids: bool = False
    # GCP bucket to export the model checkpoints to. Only the name without the prefix, e.g. "my-bucket".
    export_to_gcs_bucket: str | None = None
    # Dataset to use for perplexity evaluation. LM harness evaluation is usually more informative.
    ppl_eval_data: dict[str, Any] | None = None


class TrainState(train_state.TrainState):
    # logit masks are fp32 arrays which are 0 if not-masked, large negative number if masked.
    logit_mask_teacher: jnp.ndarray
    logit_mask_new: jnp.ndarray
    train_mask: Any
    # space masks are boolean masks indicating which tokens start with a token-ending byte (as specified by `space_mask_mode`).
    space_mask_teacher: jnp.ndarray
    space_mask_new: jnp.ndarray
    gradnorm_state: Any = struct.field(pytree_node=True)
    gradnorm_tx: optax.GradientTransformation | None = struct.field(pytree_node=False)


def _unimplemented_apply_fn(*args, **kwargs):
    raise NotImplementedError("state.apply_fn is not used.")


def get_state(
    model_params,
    original_embeddings,
    new_embeddings,
    teacher_model_params,
    teacher_embeddings,
    space_mask_teacher,
    space_mask_new,
    teacher_config,
    student_config,
    args,
    hypernet,
    learning_rate_fn,
    optimizer_kwargs,
    shard_patterns,
):
    """Creates the train state and puts all state data on the correct devices."""

    dtype = getattr(jnp, args.dtype)

    # if teacher embeddings are not separately supplied, we are doing self-distillation
    # in this case, the original embeddings are equivalent to the teacher embeddings
    if teacher_embeddings is None:
        teacher_embeddings = original_embeddings

    # pad to multiple
    n_pad_teacher = utils.get_n_pad(
        teacher_embeddings.shape[0], args.pad_to_multiple_of
    )
    n_pad_new = utils.get_n_pad(new_embeddings.shape[0], args.pad_to_multiple_of)

    logit_mask_teacher = np.ones(
        (teacher_embeddings.shape[0] + n_pad_teacher,), dtype=np.float32
    )
    logit_mask_teacher[: teacher_embeddings.shape[0]] = 0.0
    logit_mask_teacher *= utils.get_large_negative_number(
        logit_mask_teacher.dtype, module=np
    )

    logit_mask_new = np.ones((new_embeddings.shape[0] + n_pad_new,), dtype=np.float32)
    logit_mask_new[: new_embeddings.shape[0]] = 0.0
    logit_mask_new *= utils.get_large_negative_number(logit_mask_new.dtype, module=np)

    teacher_embeddings = np.pad(
        teacher_embeddings,
        ((0, n_pad_teacher), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    new_embeddings = np.pad(
        new_embeddings,
        ((0, n_pad_new), (0, 0), (0, 0)),
        mode="edge",
    )
    space_mask_teacher = np.pad(
        space_mask_teacher,
        ((0, n_pad_teacher),),
        mode="constant",
        constant_values=False,
    )
    space_mask_new = np.pad(
        space_mask_new, ((0, n_pad_new),), mode="constant", constant_values=False
    )

    if teacher_model_params is None:
        if args.train_model_mode == "full":
            # we need a separate copy of the teacher parameters
            teacher_model_params = jax.tree.map(lambda x: x.astype(dtype), model_params)
        else:
            logger.info(
                "Using a single copy of the model parameters for the teacher and the student."
            )
            # we can share the student and teacher parameters, indicated via empty teacher_model_params
            teacher_model_params = jnp.array([])
    else:
        teacher_model_params = jax.tree.map(lambda x: x.astype(dtype), teacher_model_params)

    params = {
        "model": model_params,
        "teacher_model": teacher_model_params,
        "teacher_embeddings": teacher_embeddings,
        "new_embeddings": new_embeddings,
    }

    if args.expand_input_ids:
        n_pad_original = utils.get_n_pad(
            original_embeddings.shape[0], args.pad_to_multiple_of
        )
        original_embeddings = np.pad(
            original_embeddings,
            ((0, n_pad_original), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        model_params["original_embeddings"] = original_embeddings

    # add latent projectors
    if args.latents_do_project:
        assert (
            args.latents_to_align == "last_hidden_state"
        ), "Latent projectors only implemented for last_hidden_state at the moment"

        model_params["projector_latents"] = utils.init_linear(
            args.seed, new_embeddings.shape[-1], teacher_embeddings.shape[-1], dtype
        )

    # add projectors for dskd baseline
    if "baseline_dskd" in args.losses:
        model_params["projector_t2s"] = utils.init_linear(
            args.seed, teacher_embeddings.shape[-1], new_embeddings.shape[-1], dtype
        )
        model_params["projector_s2t"] = utils.init_linear(
            args.seed, new_embeddings.shape[-1], teacher_embeddings.shape[-1], dtype
        )
        model_params["projector_query"] = utils.init_linear(
            args.seed,
            new_embeddings.shape[-1] * 2,
            teacher_embeddings.shape[-1] * 2,
            dtype,
        )

    if args.train_model_mode == "lora":
        params["model_lora"] = lora.init_lora_params(
            args,
            params["model"],
            model_type=student_config.model_type,
            seed=args.seed,
        )

    # init the hypernetwork on CPU using a small subset of the embeddings
    n_embed_init = 128
    params["hypernet"] = jax.tree.map(
        jax.device_get,
        jax.jit(hypernet.init, backend="cpu")(
            jax.random.PRNGKey(args.seed),
            params["new_embeddings"][:n_embed_init, None, :, :],
            jnp.ones((n_embed_init, 1), dtype=bool),
        ),
    )

    train_mask = utils.label_by_prefix(
        params,
        [
            ["hypernet.*rescaler.*", False],
            [("hypernet",), True],
            [
                ("teacher_model",),
                False,
            ],
            [
                ("teacher_embeddings",),
                False,
            ],
            [
                (
                    "model",
                    "original_embeddings",
                ),
                False,
            ],
            [
                "model.*(projector_query|projector_s2t|projector_t2s|projector_latents).*",
                True,
            ],
            [
                ("model",),
                (True if args.train_model_mode == "full" else False),
            ],
            [
                ("model_lora",),
                True,
            ],
            [
                ("new_embeddings",),
                True if args.train_embeddings else False,
            ],
        ],
    )
    params = jax.tree.map(
        lambda x, trainable: (x.astype(jnp.float32) if trainable else x.astype(dtype)),
        params,
        train_mask,
    )

    if args.multitask_aggregation_fn == "gradnorm":
        # we need an extra GradNorm optimizer and state
        gradnorm_opt = optax.adamw(
            learning_rate=args.baseline.gradnorm_lr,
        )

        gradnorm_state = {
            "weights": jnp.ones(len(args.losses)),
            "l0": jnp.empty(len(args.losses)),
            "opt_state": gradnorm_opt.init(jnp.ones(len(args.losses))),
        }
        gradnorm_tx = gradnorm_opt
    else:
        gradnorm_state = None
        gradnorm_tx = None

    def _jit_init_state(params):
        return TrainState.create(
            apply_fn=_unimplemented_apply_fn,
            params=params,
            tx=opt.get_optimizer(train_mask, learning_rate_fn, **optimizer_kwargs),
            logit_mask_teacher=logit_mask_teacher,
            logit_mask_new=logit_mask_new,
            train_mask=train_mask,
            space_mask_teacher=space_mask_teacher,
            space_mask_new=space_mask_new,
            gradnorm_state=gradnorm_state,
            gradnorm_tx=gradnorm_tx,
        )

    state_shape = jax.eval_shape(_jit_init_state, params)
    state_shardings = sharding.get_sharding_fn(shard_patterns, student_config.mesh)(state_shape)
    params = sharding.to_global_array(params, state_shardings.params)

    state = jax.jit(
        _jit_init_state,
        in_shardings=(state_shardings.params,),
        out_shardings=state_shardings,
        donate_argnums=(0,),
    )(params)

    return state, state_shardings


def cross_entropy(
    logits,
    labels,
    attention_mask,
    logits_already_shifted=False,
    logit_mask=None,
    denom=None,
):
    shift_logits = logits[..., :-1, :] if not logits_already_shifted else logits
    shift_labels = labels[..., 1:]
    shift_attention_mask = attention_mask[..., 1:]

    if logit_mask is not None:
        shift_logits = shift_logits + logit_mask[None, None, :]

    return (
        optax.softmax_cross_entropy(
            shift_logits, common_utils.onehot(shift_labels, shift_logits.shape[-1])
        )
        * shift_attention_mask
    ).mean() / (denom if denom is not None else shift_attention_mask.mean())


def pad_embeddings_with_random(embeddings, tokenizer, seed=1234):
    n_embed_diff = len(tokenizer) - len(embeddings)

    embeddings_mean = embeddings.mean(0)
    embeddings_std = embeddings.std(0)

    return np.concatenate(
        [
            embeddings,
            np.random.RandomState(seed).standard_normal(
                (n_embed_diff, *embeddings.shape[1:]),
            )
            * embeddings_std[None]
            + embeddings_mean[None],
        ]
    )


def main(args: CrossTokenizerDistillArgs):
    logger.info(pformat(args))

    if args.debug:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        mesh = jax.sharding.Mesh([jax.devices("cpu")], ["data", "model"])
    else:
        mesh = sharding.get_mesh(args.n_data_parallel, args.n_model_parallel)

    output_dir = Path(args.output)
    # clear previous output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "args.yaml", "w") as f:
        yaml.dump(asdict(args), f)

    if args.teacher is not None:
        teacher_config = get_config(**asdict(args.teacher))
    else:
        teacher_config = get_config(**asdict(args.student))

    student_config = get_config(**asdict(args.student))

    teacher_config.max_length = args.max_teacher_length
    if not args.debug and args.max_teacher_length % 128 == 0 and jax.devices()[0].platform == "tpu":
        teacher_config._attn_implementation = "pallas_flash_attention"
    else:
        logger.warning(
            "Using eager attention implementation for teacher (max length not divisible by 128 or debug or not TPU)"
        )
        teacher_config._attn_implementation = "eager"

    student_config.max_length = args.max_student_length
    if not args.debug and args.max_student_length % 128 == 0 and jax.devices()[0].platform == "tpu":
        student_config._attn_implementation = "pallas_flash_attention"
    else:
        logger.warning(
            "Using eager attention implementation for student (max length not divisible by 128 or debug or not TPU)"
        )
        student_config._attn_implementation = "eager"

    teacher_config.mesh = student_config.mesh = mesh

    dtype = getattr(jnp, args.dtype)

    # prepare dataset
    dataset = data.get_dataset(**args.data, seed=args.seed)
    if args.ppl_eval_data is not None:
        ppl_eval_data = data.get_dataset(**args.ppl_eval_data, seed=args.seed)
    else:
        ppl_eval_data = None

    student_model_kwargs = asdict(args.student)
    teacher_model_kwargs = (
        asdict(args.teacher) if args.teacher is not None else asdict(args.student)
    )
    original_student_tokenizer_name = student_model_kwargs.pop("tokenizer_name")
    teacher_tokenizer_name = teacher_model_kwargs.pop("tokenizer_name")
    target_tokenizer_name = args.target_tokenizer_name

    tokenizer_teacher = load_byteify_tokenizer(teacher_tokenizer_name)
    tokenizer_student_original = load_byteify_tokenizer(original_student_tokenizer_name)
    target_tokenizer = load_byteify_tokenizer(target_tokenizer_name)

    if args.tokens_to_add is not None:
        logger.info("Adding tokens: %s", args.tokens_to_add)
        target_tokenizer.add_tokens(args.tokens_to_add)

    if "baseline_mined" in args.losses:
        mined_mapping = np.load(
            Path(args.tokenizer_pair_data_path) / "mined_mapping.npy"
        )
        mined_distances = json.load(
            open(Path(args.tokenizer_pair_data_path) / "mined_distances.json")
        )
        logger.info("Average MinED distance:", np.mean(list(mined_distances.values())))
    else:
        mined_mapping = mined_distances = None

    # _do_init=False -> do not load parameters, parameters will be loaded separately
    # since we want to manually manage their device placement
    teacher_model = FlaxAutoModelForCausalLM.from_config(
        teacher_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.max_teacher_length),
    )
    new_model = FlaxAutoModelForCausalLM.from_config(
        student_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.max_student_length),
    )
    if args.gradient_checkpointing:
        # only supported for some models
        new_model.enable_gradient_checkpointing()

    model_params = param.load_params(**student_model_kwargs)
    # we manage embeddings separately from the non-embedding parameters, and add them as needed.
    embeddings, model_params = param.stack_embeddings(
        model_params,
        student_config,
        pop_embeddings=True,
    )
    embeddings = embeddings[: len(tokenizer_student_original)]

    # embeddings can sometimes spuriously be smaller than the tokenizer, e.g if the tokenizer has some special unused tokens.
    if len(embeddings) < len(tokenizer_student_original):
        logger.warning(
            "Student embeddings are smaller than student tokenizer, padding embeddings with random embeddings."
        )
        embeddings = pad_embeddings_with_random(
            embeddings, tokenizer_student_original, seed=args.seed
        )

    if args.teacher is not None:
        teacher_model_params = param.load_params(**teacher_model_kwargs)
        teacher_embeddings, teacher_model_params = param.stack_embeddings(
            teacher_model_params,
            teacher_config,
            pop_embeddings=True,
        )
        teacher_embeddings = teacher_embeddings[: len(tokenizer_teacher)]

        if len(teacher_embeddings) < len(tokenizer_teacher):
            logger.warning(
                "Teacher embeddings are smaller than teacher tokenizer, padding embeddings with random embeddings."
            )
            teacher_embeddings = pad_embeddings_with_random(
                teacher_embeddings, tokenizer_teacher, seed=args.seed
            )
    else:
        # self-distillation, teacher is student at t=0 and before tokenizer transfer.
        teacher_embeddings = None
        teacher_model_params = None

    if target_tokenizer_name == original_student_tokenizer_name:
        new_embeddings = embeddings
        if len(new_embeddings) < len(target_tokenizer):
            logger.warning(
                "Student embeddings are smaller than target tokenizer, padding embeddings with random embeddings."
            )
            new_embeddings = pad_embeddings_with_random(
                new_embeddings, target_tokenizer, seed=args.seed
            )
    else:
        diff_embeddings, original_to_new_indices, diff_indices = utils.fvt(
            tokenizer_student_original,
            target_tokenizer,
            embeddings,
        )
        new_embeddings = embeddings[original_to_new_indices]
        if len(diff_indices) > 0:
            new_embeddings[diff_indices] = diff_embeddings

        overlapping_embeddings_mask = np.ones(len(target_tokenizer), dtype=bool)
        overlapping_embeddings_mask[diff_indices] = False
        logger.warning(
            f"{sum(~overlapping_embeddings_mask)} non-overlapping embeddings"
        )

    if args.output_embeddings_mode == "untie" and new_embeddings.shape[1] == 1:
        # embeddings have shape (vocab_size, n_embeddings, n_embd)
        # to duplicate them if embeddings_mode="untie", we tile them across the n_embeddings dimension.
        student_config.tie_word_embeddings = False
        new_embeddings = jnp.tile(new_embeddings, (1, 2, 1))

    space_mask_teacher = utils.get_space_mask(tokenizer_teacher, args.space_mask_mode)
    space_mask_new = utils.get_space_mask(target_tokenizer, args.space_mask_mode)

    logger.info(f"Space mask teacher sum: {space_mask_teacher.sum()}")
    logger.info(f"Space mask new sum: {space_mask_new.sum()}")

    hypernet_config = HypernetConfig(
        hidden_size=embeddings.shape[-1],
        num_embeddings=1 if student_config.tie_word_embeddings else 2,
        max_seq_length=1,
        **asdict(args.hypernet),
    )
    hypernet = Hypernet(config=hypernet_config, dtype=dtype)

    optimizer_kwargs = args.optimizer
    learning_rate_fn = lr.linear_warmup_linear_decay_with_linear_prefix(
        optimizer_kwargs.pop("learning_rate"),
        args.steps,
        args.warmup_steps,
        0,
        0,
    )

    shard_patterns = {
        **sharding.get_shard_patterns(teacher_config.model_type),
        **sharding.get_shard_patterns(student_config.model_type),
        **sharding.get_shard_patterns("hypernet"),
    }
    state, state_shardings = get_state(
        model_params=model_params,
        original_embeddings=embeddings,
        new_embeddings=new_embeddings,
        teacher_model_params=teacher_model_params,
        teacher_embeddings=teacher_embeddings,
        space_mask_teacher=space_mask_teacher,
        space_mask_new=space_mask_new,
        teacher_config=teacher_config,
        student_config=student_config,
        args=args,
        hypernet=hypernet,
        learning_rate_fn=learning_rate_fn,
        optimizer_kwargs=optimizer_kwargs,
        shard_patterns=shard_patterns,
    )

    teacher_config.vocab_size = teacher_model.config.vocab_size = state.params[
        "teacher_embeddings"
    ].shape[0]
    student_config.vocab_size = new_model.config.vocab_size = state.params[
        "new_embeddings"
    ].shape[0]
    logger.info(
        f"Updated source vocab size: {teacher_config.vocab_size} (after padding)"
    )
    logger.info(
        f"Updated target vocab size: {student_config.vocab_size} (after padding)"
    )

    hypernet_fn, teacher_model_fn, new_model_fn = (
        hypernet.apply,
        teacher_model.__call__,
        new_model.__call__,
    )
    utils.param_report(state.params, state.train_mask)
    train_mask = jax.device_get(state.train_mask)

    if args.expand_input_ids:
        expand_input_ids_dict = utils.get_expand_input_ids_dict(
            target_tokenizer,
            tokenizer_student_original.get_vocab(),
        )
    else:
        expand_input_ids_dict = None

    collator = collators.TokenizerAlignerCollator(
        tokenizer_teacher,
        target_tokenizer,
        max_teacher_length=args.max_teacher_length,
        max_student_length=args.max_student_length,
        use_chat_template=args.use_chat_template,
        chat_template_mode=args.chat_template_mode,
        expand_input_ids_dict=expand_input_ids_dict,
        loss_mask_mode=args.loss_mask_mode,
        tokenizer_pair_data_path=args.tokenizer_pair_data_path,
        tokenizer_pair_bias_threshold=args.tokenizer_pair_bias_threshold,
        require_bias_matrices=any("unbiased" in x for x in args.losses),
    )

    train_dataloader = StatefulDataLoader(
        dataset.get_torch_dataset(),
        batch_size=1,  # batched internally
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    if ppl_eval_data is not None:
        ppl_eval_dataloader = torch.utils.data.DataLoader(
            ppl_eval_data.get_torch_dataset(),
            batch_size=1,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
    else:
        ppl_eval_dataloader = None

    if jax.process_index() == 0:
        wandb.init(project="tokenkit", name=args.name, config=asdict(args))
        wandb.run.log_code()

    def predict_embeddings(params):
        """Predict embeddings using the hypernet (identity if hypernet is disabled)."""

        embeddings = params["new_embeddings"]
        embeddings = jax.lax.with_sharding_constraint(
            embeddings, NamedSharding(mesh, P("model", None, "data"))
        )

        predicted_embeddings = hypernet_fn(
            params["hypernet"],
            embeddings[:, None, :, :],
            jnp.ones((embeddings.shape[0], 1), dtype=bool),
        )

        return jax.lax.with_sharding_constraint(
            predicted_embeddings, NamedSharding(mesh, P("model", None, "data"))
        )

    def compute_inputs_embeds(model_params, input_ids, expanded_input_ids):
        """Compute the input embeddings, expanding if enabled."""

        input_embeddings = param.get(
            model_params, param.get_input_embedding_path(student_config.model_type)
        )

        # NOTE: this assumes Llama/Gemma-style where position embeddings are part of attention mechanism
        if args.expand_input_ids:
            standard_inputs_embeds = jnp.take(
                input_embeddings,
                input_ids,
                axis=0,
            )
            expanded_inputs_embeds = jnp.take(
                model_params["original_embeddings"][:, 0, :],
                expanded_input_ids,
                axis=0,
            )

            inputs_embeds = standard_inputs_embeds + expanded_inputs_embeds
        else:
            inputs_embeds = jnp.take(
                input_embeddings,
                input_ids,
                axis=0,
            )

        return inputs_embeds

    def train_step(state, batch, global_batch):
        def compute_loss(non_trainable_params, *trainable_params):
            params = jax.tree.map(
                lambda *args: next(x for x in args if x is not None),
                *trainable_params,
                non_trainable_params,
                is_leaf=lambda x: x is None,
            )

            scalar_report = {}  # extra logging

            if args.train_model_mode == "lora":
                model_params_with_lora = lora.materialize_lora(
                    params["model"],
                    params["model_lora"],
                    alpha=args.model_lora_alpha,
                )
            else:
                model_params_with_lora = params["model"]
            predicted_embeddings = predict_embeddings(params)
            model_params_with_predicted_embeddings = param.assign_embeddings(
                model_params_with_lora,
                predicted_embeddings,
                config=student_config,
            )

            uses_dummy_teacher_params = (
                isinstance(params["teacher_model"], jnp.ndarray)
                and params["teacher_model"].size == 0
            )

            teacher_model_params = param.assign_embeddings(
                (
                    params["model"]
                    if uses_dummy_teacher_params
                    else params["teacher_model"]
                ),
                params["teacher_embeddings"],
                config=teacher_config,
            )

            need_teacher = len([loss for loss in args.losses if loss != "sft"]) > 0
            need_hidden_states = len([loss for loss in args.losses if loss in {"alm_latents", "baseline_dskd"}]) > 0

            if need_teacher:
                teacher_out = teacher_model_fn(
                    input_ids=batch["input_ids_original"],
                    params=teacher_model_params,
                    dropout_rng=None,
                    train=False,
                    output_hidden_states=need_hidden_states,
                    output_attentions=need_hidden_states,
                )
                teacher_logits = (
                    teacher_out.logits.astype(jnp.float32)
                    + state.logit_mask_teacher[None, None]
                )
                # inexcplicably, `{student,teacher}_log_softmax_max` may be >0 without the jnp.clip(..)
                # clip and keep monitoring...
                teacher_logprobs = jnp.clip(
                    jax.nn.log_softmax(teacher_logits, axis=-1), max=0.0
                )
                scalar_report["teacher_log_softmax_max"] = teacher_logprobs.max()
                teacher_probs = jnp.exp(teacher_logprobs)
                scalar_report["teacher_exp_log_softmax_max"] = teacher_probs.max()
            else:
                teacher_out = teacher_logits = teacher_logprobs = teacher_probs = None

            inputs_embeds_new = compute_inputs_embeds(
                model_params_with_predicted_embeddings,
                batch["input_ids_new"],
                batch.get("expanded_input_ids_new"),
            )

            student_out = new_model_fn(
                input_ids=None,
                inputs_embeds=inputs_embeds_new,
                params=model_params_with_predicted_embeddings,
                dropout_rng=None,
                train=False,
                output_hidden_states=need_hidden_states,
                output_attentions=need_hidden_states,
            )
            student_logits = (
                student_out.logits.astype(jnp.float32)
                + state.logit_mask_new[None, None]
            )

            # inexplicably, `{student,teacher}_log_softmax_max` may be >0 without the jnp.clip(..)
            # clip and keep monitoring...
            student_logprobs = jnp.clip(
                jax.nn.log_softmax(student_logits, axis=-1), max=0.0
            )
            scalar_report["student_log_softmax_max"] = student_logprobs.max()
            student_probs = jnp.exp(student_logprobs)
            scalar_report["student_exp_log_softmax_max"] = student_probs.max()

            scalar_report["n_teacher_used_tokens"] = (
                global_batch["alignment_matrix_b_unconstrained"].any(-1).sum(-1).mean()
            )
            scalar_report["n_student_used_tokens"] = (
                global_batch["alignment_matrix_a_unconstrained"].any(-1).sum(-1).mean()
            )
            scalar_report["loss_mask_original_mean"] = global_batch[
                "loss_mask_original"
            ][:, 1:].mean()
            scalar_report["loss_mask_new_mean"] = global_batch["loss_mask_new"][
                :, 1:
            ].mean()

            loss_args = losses.LossArgs(
                params=params,
                batch=batch,
                global_batch=global_batch,
                teacher_config=teacher_config,
                new_config=student_config,
                teacher_out=teacher_out,
                student_out=student_out,
                tokenizer_teacher=tokenizer_teacher,
                tokenizer_new=target_tokenizer,
                teacher_probs=teacher_probs,
                teacher_logprobs=teacher_logprobs,
                teacher_logits=teacher_logits,
                student_probs=student_probs,
                student_logprobs=student_logprobs,
                student_logits=student_logits,
                predicted_embeddings=predicted_embeddings,
                scalar_report=scalar_report,
                space_mask_teacher=state.space_mask_teacher,
                space_mask_new=state.space_mask_new,
                logit_mask_teacher=state.logit_mask_teacher,
                logit_mask_new=state.logit_mask_new,
            )

            loss_values = jnp.zeros(len(args.losses), dtype=jnp.float32)

            for loss_idx, loss in enumerate(args.losses):
                if loss == "sft":
                    current_loss = losses.compute_sft_loss(args, loss_args)
                elif loss == "alm_latents":
                    current_loss = losses.compute_alm_latents_loss(args, loss_args)
                elif loss.startswith("alm"):
                    kind = loss[len("alm_") :]
                    if len(kind) == 0:
                        kind = "unbiased"
                    current_loss = losses.compute_alm_loss(
                        chunk_kind=kind,
                        args=args,
                        loss_args=loss_args,
                    )
                elif loss == "baseline_dskd":
                    current_loss = losses.compute_baseline_dskd_loss(args, loss_args)
                elif loss == "baseline_uld":
                    current_loss = losses.compute_baseline_uld_loss(args, loss_args)
                elif loss == "baseline_mined":
                    current_loss = losses.compute_baseline_mined_loss(
                        mined_mapping, args, loss_args
                    )
                else:
                    raise ValueError(f"Invalid loss: {loss}")

                weight = (
                    args.loss_weights[loss_idx]
                    if args.loss_weights is not None
                    else 1.0
                )
                if args.loss_schedules is not None:
                    if args.loss_schedules[loss_idx] == "cosine":
                        weight = (
                            weight * (1 + jnp.cos(jnp.pi * state.step / args.steps)) / 2
                        )
                    elif args.loss_schedules[loss_idx] == "reverse_cosine":
                        weight = (
                            weight * (1 - jnp.cos(jnp.pi * state.step / args.steps)) / 2
                        )
                    elif args.loss_schedules[loss_idx] == "linear":
                        weight = weight * state.step / args.steps
                    elif args.loss_schedules[loss_idx] == "constant":
                        pass
                    else:
                        raise ValueError(
                            "Invalid loss schedule: {}".format(
                                args.loss_schedules[loss_idx]
                            )
                        )

                scalar_report[f"loss/{loss}"] = current_loss
                scalar_report[f"loss/{loss}_weight"] = weight

                loss_values = loss_values.at[loss_idx].set(weight * current_loss)

            return loss_values, scalar_report

        trainable_params = jax.tree.map(
            lambda x, m: x if m else None, state.params, train_mask
        )
        last_layer_trainable_params = jax.tree.map(
            lambda x, m1, m2: x if m1 and m2 else None,
            state.params,
            train_mask,
            param.get_layer_n_mask(state.params, student_config, -1),
        )

        if args.multitask_aggregation_fn is None:

            def compute_loss_avg(*pargs):
                loss_values, scalar_report = compute_loss(*pargs)
                return jnp.mean(loss_values), scalar_report

            grad_fn = jax.value_and_grad(compute_loss_avg, has_aux=True, argnums=1)
            (loss, scalar_report), grad = grad_fn(state.params, trainable_params)
        elif args.multitask_aggregation_fn in {
            "approx_gradmag",
            "approx_gradmag_preserve_mag",
        }:
            jac_fn = jax.jacrev(compute_loss, has_aux=True, argnums=1)
            (last_layer_grads, _) = jac_fn(
                state.params, last_layer_trainable_params, trainable_params
            )
            approx_grad_norm = multitask.compute_global_grad_norm(last_layer_grads)
            # stop grad is not necessary here since the var is defined outside compute_loss_weighted, but added for clarity
            approx_loss_weights = jax.lax.stop_gradient(
                multitask.compute_inv_global_grad_norm(last_layer_grads)
            )

            if args.multitask_aggregation_fn == "approx_gradmag_preserve_mag":
                denominator = jnp.sum(approx_loss_weights)
            else:
                denominator = 1.0

            last_layer_grad = jax.tree.map(
                lambda x: jnp.sum(x, axis=0) / denominator,
                multitask.gradmag(last_layer_grads),
            )

            def compute_loss_weighted(*pargs):
                loss_values, scalar_report = compute_loss(*pargs)
                return (
                    jnp.sum(loss_values * approx_loss_weights) / denominator,
                    scalar_report,
                )

            grad_fn = jax.value_and_grad(compute_loss_weighted, has_aux=True, argnums=2)
            (loss, scalar_report), non_last_layer_grad = grad_fn(
                state.params, last_layer_trainable_params, trainable_params
            )

            for loss_idx, loss_name in enumerate(args.losses):
                scalar_report[f"loss/{loss_name}_approx_grad_norm"] = approx_grad_norm[
                    loss_idx
                ]
                scalar_report[f"loss/{loss_name}_approx_loss_weight"] = (
                    approx_loss_weights[loss_idx]
                )

            grad = jax.tree.map(
                lambda x, y: x if x is not None else y,
                last_layer_grad,
                non_last_layer_grad,
                is_leaf=lambda x: x is None,
            )
        elif args.multitask_aggregation_fn == "gradnorm":
            def compute_loss_with_aux_value(*args):
                loss_values, scalar_report = compute_loss(*args)
                return loss_values, (scalar_report, loss_values)

            jac_fn = jax.jacrev(compute_loss_with_aux_value, has_aux=True, argnums=1)
            (last_layer_grads, (_, loss_values)) = jac_fn(
                state.params, last_layer_trainable_params, trainable_params
            )
            approx_grad_norm = multitask.compute_global_grad_norm(last_layer_grads)

            l0 = jax.lax.cond(
                state.step == 0,
                lambda: jax.lax.stop_gradient(loss_values),
                lambda: state.gradnorm_state["l0"],
            )

            def compute_loss_weighted(*pargs):
                loss_values, scalar_report = compute_loss(*pargs)
                return (
                    jnp.sum(loss_values * state.gradnorm_state["weights"]),
                    scalar_report,
                )

            last_layer_grad = jax.tree.map(
                lambda x: jnp.sum(x * jnp.expand_dims(state.gradnorm_state["weights"], axis=tuple(range(1, x.ndim))), axis=0),
                last_layer_grads,
            )

            grad_fn = jax.value_and_grad(compute_loss_weighted, has_aux=True, argnums=2)
            (loss, scalar_report), non_last_layer_grad = grad_fn(
                state.params, last_layer_trainable_params, trainable_params
            )

            loss_ratio = loss_values / l0
            rt = loss_ratio / loss_ratio.mean()

            def compute_gradnorm_loss(w, approx_grad_norm):
                approx_grad_norm = jax.tree.map(lambda x, y: x * y, approx_grad_norm, w)

                gw_avg = approx_grad_norm.mean()
                constant = (gw_avg * rt ** args.baseline.gradnorm_alpha)
                return jnp.abs(approx_grad_norm - constant).mean()

            weight_grad = jax.grad(compute_gradnorm_loss, argnums=0)(
                state.gradnorm_state["weights"], approx_grad_norm
            )
            grad_norm_updates, grad_norm_new_opt_state = state.gradnorm_tx.update(
                weight_grad, state.gradnorm_state["opt_state"], state.gradnorm_state["weights"]
            )
            new_weights = optax.apply_updates(state.gradnorm_state["weights"], grad_norm_updates)
            new_weights = new_weights / new_weights.sum() * len(args.losses)
            state = state.replace(gradnorm_state={
                "weights": new_weights,
                "opt_state": grad_norm_new_opt_state,
                "l0": l0,
            })

            for loss_idx, loss_name in enumerate(args.losses):
                scalar_report[f"loss/{loss_name}_approx_grad_norm"] = approx_grad_norm[
                    loss_idx
                ]
                scalar_report[f"loss/{loss_name}_weight"] = (
                    new_weights[loss_idx]
                )
                scalar_report[f"loss/{loss_name}_l0"] = (
                    l0[loss_idx]
                )
                scalar_report[f"loss/{loss_name}_loss_ratio"] = (
                    loss_ratio[loss_idx]
                )

            grad = jax.tree.map(
                lambda x, y: x if x is not None else y,
                last_layer_grad,
                non_last_layer_grad,
                is_leaf=lambda x: x is None,
            )
        else:

            def compute_loss_with_aux_value(*args):
                loss_values, scalar_report = compute_loss(*args)
                return loss_values, (scalar_report, loss_values)

            jac_fn = jax.jacrev(compute_loss_with_aux_value, has_aux=True, argnums=1)
            (grads, (scalar_report, loss_values)) = jac_fn(
                state.params, trainable_params
            )

            global_grad_norms_before_agg = multitask.compute_global_grad_norm(grads)

            mt_agg_fns = args.multitask_aggregation_fn.split("+")
            for mt_agg_fn in mt_agg_fns:
                mt_agg_args = mt_agg_fn.split(":")
                mt_agg_fn = mt_agg_args[0]
                mt_agg_extra_args = tuple(json.loads(x) for x in mt_agg_args[1:])

                if mt_agg_fn == "pcgrad":
                    grads = multitask.pcgrad(grads, *mt_agg_extra_args)
                elif mt_agg_fn == "gradmag":
                    grads = multitask.gradmag(grads, *mt_agg_extra_args)
                elif mt_agg_fn == "gradclip":
                    grads = multitask.gradclip(grads, *mt_agg_extra_args)
                elif mt_agg_fn == "identity":
                    pass
                else:
                    raise ValueError(
                        f"Invalid multitask aggregation function: {mt_agg_fn}"
                    )

            global_grad_norms_after_agg = multitask.compute_global_grad_norm(grads)

            for loss_idx, loss_name in enumerate(args.losses):
                scalar_report[f"loss/{loss_name}_global_grad_norm_before_agg"] = (
                    global_grad_norms_before_agg[loss_idx]
                )
                scalar_report[f"loss/{loss_name}_global_grad_norm_after_agg"] = (
                    global_grad_norms_after_agg[loss_idx]
                )

            grad = jax.tree.map(lambda task_grads: jnp.sum(task_grads, axis=0), grads)
            loss = jnp.sum(loss_values)

        grad = jax.tree.map(
            lambda g, p: g if g is not None else jnp.zeros_like(p),
            grad,
            state.params,
            is_leaf=lambda x: x is None,
        )
        new_state = state.apply_gradients(grads=grad)

        metrics = {
            "loss": loss,
            "learning_rate": learning_rate_fn(state.step),
            **scalar_report,
        }
        return new_state, metrics

    def eval_step(state, batch):
        if args.train_model_mode == "lora":
            model_params_with_lora = lora.materialize_lora(
                state.params["model"],
                state.params["model_lora"],
                alpha=args.model_lora_alpha,
            )
        else:
            model_params_with_lora = state.params["model"]
        predicted_embeddings = predict_embeddings(state.params)
        model_params_with_embeddings = param.assign_embeddings(
            model_params_with_lora,
            predicted_embeddings,
            config=student_config,
        )

        inputs_embeds_new = compute_inputs_embeds(
            model_params_with_embeddings,
            batch["input_ids_new"],
        )

        logits = new_model_fn(
            input_ids=None,
            inputs_embeds=inputs_embeds_new,
            params=model_params_with_embeddings,
            dropout_rng=None,
            train=False,
        ).logits

        loss = cross_entropy(
            logits,
            batch["input_ids_new"],
            batch["attention_mask_new"],
            logit_mask=state.logit_mask_new,
        )

        return {"loss": loss}

    batch_shardings = jax.tree.map(
        lambda x: NamedSharding(mesh, x), collator.get_batch_pspecs()
    )

    jtrain_step = jax.jit(
        train_step,
        in_shardings=(state_shardings, batch_shardings, batch_shardings),
        out_shardings=(state_shardings, None),
        donate_argnums=(0,),
    )
    jeval_step = jax.jit(
        eval_step,
        in_shardings=(state_shardings, batch_shardings),
    )
    jpredict_embeddings = jax.jit(
        predict_embeddings,
        in_shardings=(state_shardings.params,),
        out_shardings=NamedSharding(mesh, P("model", None, "data")),
    )
    if args.train_model_mode == "lora":
        jmaterialize_lora = jax.jit(
            lora.materialize_lora,
            in_shardings=(
                state_shardings.params["model"],
                state_shardings.params["model_lora"],
            ),
            out_shardings=state_shardings.params["model"],
            donate_argnums=(0,),
            static_argnums=(2,),
        )
        jdematerialize_lora = jax.jit(
            lora.dematerialize_lora,
            in_shardings=(
                state_shardings.params["model"],
                state_shardings.params["model_lora"],
            ),
            out_shardings=state_shardings.params["model"],
            donate_argnums=(0,),
            static_argnums=(2,),
        )
    else:
        jmaterialize_lora = lambda x, y, z: x
        jdematerialize_lora = lambda x, y, z: x

    def eval_loop(dataloader):
        eval_metrics = []

        for batch in tqdm(dataloader, desc="Running PPL evaluation..."):
            batch = sharding.sync_across_devices(batch)
            batch = sharding.to_global_array(batch, batch_shardings)
            step_metrics = jeval_step(state, batch)
            eval_metrics.append(step_metrics)

        eval_metrics = jax.tree.map(np.mean, common_utils.stack_forest(eval_metrics))
        return eval_metrics

    diter = iter(train_dataloader)
    first_batch = next(iter(train_dataloader))

    if args.do_cost_analysis:
        compiled_train_step_fn = jtrain_step.lower(
            state, first_batch, first_batch
        ).compile()
        flops_per_step = compiled_train_step_fn.cost_analysis()["flops"]
        memory_per_step = (
            compiled_train_step_fn.memory_analysis().output_size_in_bytes
            + compiled_train_step_fn.memory_analysis().temp_size_in_bytes
        )
        logger.info("TFLOPs per step: %.2f", flops_per_step / (10**12))
        logger.info("Memory (MB) per step: %.2f", memory_per_step / (1024**2))
        sys.exit()

    utils.print_example_alignments(
        first_batch["alignment_matrix_b_unconstrained"][0],
        first_batch["alignment_matrix_a_unconstrained"][0],
        tokenizer_teacher.convert_ids_to_tokens(first_batch["input_ids_original"][0]),
        target_tokenizer.convert_ids_to_tokens(first_batch["input_ids_new"][0]),
    )

    train_metrics = []
    start_time = time.time()

    upload_executor = None
    upload_name = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + args.name

    grad_acc_steps = (
        args.optimizer["grad_acc_steps"]
        if args.optimizer.get("grad_acc_steps") is not None
        else 1
    )
    assert args.data["batch_size"] % grad_acc_steps == 0
    local_batch_size = args.data["batch_size"] // grad_acc_steps

    for step in tqdm(range(args.steps)):
        try:
            batch = next(diter)
        except StopIteration:
            diter = iter(train_dataloader)
            batch = next(diter)

        batch = sharding.sync_across_devices(batch)
        global_batch = sharding.to_global_array(batch, batch_shardings)

        if args.dry_run:
            continue

        for grad_acc_step in range(grad_acc_steps):
            if grad_acc_steps > 1:
                start, end = (
                    grad_acc_step * local_batch_size,
                    (grad_acc_step + 1) * local_batch_size,
                )

                def split_local(arr):
                    if arr.shape[0] == args.data["batch_size"]:
                        return arr[start:end]
                    else:
                        assert len(arr.shape) == 1  # otherwise ambiguous
                        return arr

                local_batch = jax.tree.map(split_local, batch)
            else:
                local_batch = batch

            local_batch = sharding.to_global_array(local_batch, batch_shardings)

            state, step_metrics = jtrain_step(
                state.replace(step=step), local_batch, global_batch
            )

            train_metrics.append(step_metrics)

        if (step + 1) % args.sync_interval == 0:
            stacked_train_metrics = jax.tree.map(
                jax.device_get, common_utils.stack_forest(train_metrics)
            )
            end_step = step + 1
            start_step = end_step - args.sync_interval
            for i in range(start_step, end_step, args.log_interval):
                for key, values in stacked_train_metrics.items():
                    avg_value = values[
                        (i - start_step)
                        * grad_acc_steps : (i - start_step + args.log_interval)
                        * grad_acc_steps
                    ].mean()
                    utils.log({f"train/{key}": avg_value}, step=i + args.log_interval)

            utils.log({"step": end_step}, step=i + args.log_interval)
            train_metrics = []

        if (step + 1) % args.eval_interval == 0 or (
            step == 0 and args.eval_at_step_zero
        ):
            # TODO: probably extract into eval function doing everything here
            if ppl_eval_dataloader is not None:
                logger.info("PPL Eval:")
                ppl_metrics = eval_loop(ppl_eval_dataloader)
                ppl_metrics = {f"eval_{k}": v for k, v in ppl_metrics.items()}
                utils.log(ppl_metrics, step=step + 1)

            if not args.skip_lm_eval:
                predicted_embeddings = jpredict_embeddings(state.params)
                model_params_with_embeddings = param.assign_embeddings(
                    jmaterialize_lora(
                        state.params["model"],
                        state.params.get("model_lora"),
                        args.model_lora_alpha,
                    ),
                    predicted_embeddings,
                    config=student_config,
                )
                model_params_with_embeddings_shardings = sharding.get_sharding_fn(
                    shard_patterns, mesh
                )({"params": model_params_with_embeddings})["params"]

                @partial(
                    jax.jit,
                    static_argnames=("model_fn", "atol"),
                    in_shardings=(
                        model_params_with_embeddings_shardings,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                    out_shardings=(None, None),
                )
                def jaxlm_inner_score_fn(
                    model_fn,
                    params,
                    input_ids,
                    expanded_input_ids,
                    labels,
                    suffix_mask,
                    space_mask,
                    logit_mask,
                    atol=eval.ATOL,
                ):
                    inputs_embeds = compute_inputs_embeds(
                        params,
                        input_ids,
                        expanded_input_ids,
                    )
                    return eval.score(
                        model_fn,
                        params,
                        (None, inputs_embeds),
                        labels=labels,
                        suffix_mask=suffix_mask,
                        space_mask=space_mask,
                        logit_mask=logit_mask,
                        atol=atol,
                    )

                def jaxlm_score_fn(model_fn, params, model_args, *pargs):
                    (input_ids,) = model_args
                    if args.expand_input_ids:
                        expanded_input_ids = utils.np_expand_input_ids(
                            input_ids,
                            expand_input_ids_dict,
                        )
                    else:
                        expanded_input_ids = None

                    return jaxlm_inner_score_fn(
                        model_fn,
                        params,
                        input_ids,
                        expanded_input_ids,
                        *pargs,
                    )

                jaxlm_kwargs = {"score_fn": jaxlm_score_fn}

                if args.expand_input_ids:
                    jaxlm_kwargs["expand_input_ids"] = True
                    jaxlm_kwargs["expand_input_ids_vocab"] = tokenizer_student_original.get_vocab()

                lm_eval_metrics, post_eval_params_buffer = eval.evaluate(
                    model=new_model,
                    config=student_config,
                    params=model_params_with_embeddings,
                    tokenizer=target_tokenizer,
                    logit_mask=state.logit_mask_new == 0,
                    output=output_dir / f"step_{step + 1}" / "lm_eval",
                    jaxlm_kwargs=jaxlm_kwargs,
                    **asdict(args.eval),
                )
                state.params["model"] = jdematerialize_lora(
                    param.unassign_embeddings(post_eval_params_buffer, student_config),
                    state.params.get("model_lora"),
                    args.model_lora_alpha,
                )

                logger.info("LM Eval:")
                lm_eval_metrics = {
                    "_".join(k): v
                    for k, v in traverse_util.flatten_dict(lm_eval_metrics).items()
                }
                lm_eval_metrics = {
                    f"lm_eval_{k}": v for k, v in lm_eval_metrics.items()
                }

                utils.log(lm_eval_metrics, step=step + 1)

        if (step + 1) % args.save_interval == 0 or (
            step == 0 and args.save_at_step_zero
        ):
            if upload_executor is not None:
                upload_executor.shutdown(wait=True)
            multihost_utils.sync_global_devices("uploaded previous checkpoint")

            del student_config.mesh
            student_config.save_pretrained(output_dir)
            student_config.mesh = mesh
            target_tokenizer.save_pretrained(output_dir)
            checkpoint.save(
                output_dir / "params.msgpack",
                state.params,
                state_shardings.params,
                mesh,
                train_mask,
                keys_to_keep={"hypernet", "new_embeddings"},
            )

            if jax.process_index() == 0 and args.export_to_gcs_bucket is not None:
                upload_executor = gcs_utils.upload_directory_to_gcs(
                    args.export_to_gcs_bucket,
                    output_dir,
                    os.path.join(upload_name, f"step_{step + 1}"),
                )

        if (step + 1) % args.sync_interval == 0:
            if jax.process_index() == 0:
                utils.log(
                    {
                        "step": step + 1,
                        "time": time.time() - start_time,
                        "epoch": step / len(train_dataloader),
                    },
                    step=step + 1,
                    commit=True,
                )

    if upload_executor is not None:
        upload_executor.shutdown(wait=True)
    multihost_utils.sync_global_devices("uploaded final checkpoint")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "100"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = (
        True  # careful about this, required for lm_eval
    )

    main(parse_args.parse_args(CrossTokenizerDistillArgs))
