import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jax.experimental import multihost_utils
from transformers import FlaxAutoModelForCausalLM
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
import torch
import sys
from flax.training import train_state, common_utils
from flax import traverse_util
import wandb
import os
import logging
from pathlib import Path
from pprint import pformat
import numpy as np
import datasets
from dataclasses import dataclass, field, asdict
import copy
from typing import Any
from functools import partial
from torchdata.stateful_dataloader import StatefulDataLoader
import time
import yaml
from datetime import datetime
import shutil

from tokenkit import utils, data, eval, parse_args, gcs_utils
from tokenkit.hf import get_config
from tokenkit.training import losses, opt, lr, collators, checkpoint, multitask
from tokenkit.utils import tqdm
from tokenkit.models import param, sharding, lora
from tokenkit.models.hypernet import Hypernet
from tokenkit.byteify import load_byteify_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenizerSamplerCollatorArgs:
    do_tokenizer_sampling: bool = True
    sample_text_span: bool = True
    n_pools: int = 1
    add_prefix_space: bool = False
    hn_surface_maxlen: int = 8
    n_token_subsample: int | None = None
    identity_n_token_subsample: int = 16384
    pad_to_multiple_of: int = 64
    tokenizer_sample_max: int = 32768
    tokenizer_sample_mean: int = 32768
    tokenizer_sample_min: int = 32768
    tokenizer_sample_std: int = 0
    tokenizer_batch_size: int = 2048
    tokenizer_noise_std: int = 4
    tokenizer_noise_mean: float = 1e-5
    block_size: int = 128


@dataclass
class BaselineArgs:
    divergence: str = "srkl"
    adaptive_kl_alpha: float = 0.5
    skew_lambda: float = 0.1
    teacher_temperature: float = 1.0
    kd_rate: float = 0.5
    kd_temp: float = 2.0


@dataclass
class HnEvalArgs(parse_args.EvalArgs):
    tokenizers: list[dict] = field(default_factory=list)


@dataclass
class TrainZettHnArgs:
    losses: list[str]
    steps: int
    warmup_steps: int
    name: str
    output: str
    num_workers: int
    log_interval: int
    sync_interval: int
    eval_interval: int
    save_interval: int
    collator: TokenizerSamplerCollatorArgs
    data: dict[str, Any]
    hypernet: parse_args.HypernetArgs
    optimizer: dict[str, Any]
    eval: HnEvalArgs
    model: parse_args.ModelArgs
    baseline: BaselineArgs = field(default_factory=BaselineArgs)
    dtype: str = "bfloat16"
    debug: bool = False
    seed: int = 1234
    max_teacher_length: int = 512
    max_student_length: int = 512
    pad_to_multiple_of: int = 64
    eval_at_step_zero: bool = False
    save_at_step_zero: bool = False
    skip_lm_eval: bool = False
    output_embeddings_mode: str = "preserve"
    use_chat_template: bool = True
    chat_template_mode: str = "direct_encode"
    loss_mask_mode: str | None = None
    gradient_checkpointing: bool = False
    do_cost_analysis: bool = False
    dry_run: bool = False
    n_data_parallel: int = 1
    n_model_parallel: int = 8
    loss_weights: list[float] | None = None
    loss_schedules: list[str] | None = None
    multitask_aggregation_fn: str | None = None
    binarization_temp: float = 100.0
    distill_chunk_sizes: list[int] = field(default_factory=lambda: [1])
    alm_diff_fn: str = "binary_ce"
    distill_main_path_numerator: str = "chunk_count"
    distill_main_path_denominator: str = "chunk_count"
    train_model_mode: str = "no"
    train_embeddings: bool = True
    model_lora_rank: int = 64
    model_lora_alpha: int = 64
    tokens_to_add: list[str] | None = None
    latents_to_align: str = "last_hidden_state"
    latents_normalization: str = "l2_channelwise"
    latents_chunks: str = "naive"
    latents_do_project: bool = False
    alm_mode: str = "append_space"
    space_mask_mode: str = "space+tab+newline+special"
    tokenizer_pair_data_path: str | None = None
    tokenizer_pair_bias_threshold: float = 1e-4
    expand_input_ids: bool = False
    export_to_gcs_bucket: str | None = None
    ppl_eval_data: dict[str, Any] | None = None
    # hn specific
    identity_steps: int = 0
    identity_lr: float = 3e-4
    compat: bool = False


def get_last_index_per_column(matrix):
    matrix_last_only = (
        jnp.asarray(matrix).at[:, :-1].set(matrix[:, :-1] & ~matrix[:, 1:])
    )
    return matrix_last_only.argmax(-2), matrix_last_only.max(-2) != 0


def predict_embeddings(
    hypernet_fn,
    mesh,
    n_vocab,
    hypernet_params,
    surface_forms,
    surface_forms_attention_mask,
    source_embeddings,
    extra_embeddings,
    special_indices,
    special_indices_in_reference,
):
    main_ids = jnp.minimum(surface_forms, n_vocab - 1)

    if extra_embeddings.shape[0] == 0:
        embeddings_to_compose = jnp.take(source_embeddings, main_ids, axis=0)
    else:
        use_extra = surface_forms >= n_vocab
        extra_ids = jnp.maximum(surface_forms - n_vocab, 0)
        embeddings_to_compose = jnp.where(
            use_extra[..., None, None],
            jnp.take(extra_embeddings, extra_ids, axis=0),
            jnp.take(source_embeddings, main_ids, axis=0),
        )

    embeddings_to_compose = jax.lax.with_sharding_constraint(
        embeddings_to_compose, NamedSharding(mesh, P("model", None, "data"))
    )

    predicted_embeddings = hypernet_fn(
        hypernet_params,
        embeddings_to_compose,
        surface_forms_attention_mask,
    )

    if special_indices is not None:
        assert special_indices_in_reference is not None

        predicted_embeddings = predicted_embeddings.at[special_indices].set(
            source_embeddings[special_indices_in_reference]
        )

    return jax.lax.with_sharding_constraint(
        predicted_embeddings, NamedSharding(mesh, P("model", None, "data"))
    )


def assign_embeddings(model_params, embeddings, config):
    model_params = param.put(
        model_params,
        param.get_input_embedding_path(config.model_type),
        embeddings[:, 0],
    )
    if not config.tie_word_embeddings:
        model_params = param.put(
            model_params,
            param.get_output_embedding_path(config.model_type),
            embeddings[:, 1].T,
        )

    return model_params


def compute_embeddings_and_out(
    args,
    params,
    input_ids,
    surface_forms,
    surface_forms_attention_mask,
    priors,
    mask,
    special_indices,
    special_indices_in_reference,
    source_embeddings,
    extra_embeddings,
    original_n_vocab,
    hypernet_fn,
    model,
):
    predicted_embeddings = predict_embeddings(
        hypernet_fn,
        model.config.mesh,
        original_n_vocab,
        params["hypernet"],
        surface_forms,
        surface_forms_attention_mask,
        source_embeddings,
        extra_embeddings,
        special_indices,
        special_indices_in_reference,
    )

    if args.train_model_mode == "lora":
        model_params_with_lora = lora.materialize_lora(
            params["model"],
            params["model_lora"],
            alpha=args.model_lora_alpha,
        )
    else:
        model_params_with_lora = params["model"]

    params_with_updated_embeddings = assign_embeddings(
        model_params_with_lora, predicted_embeddings, model.config
    )

    model.config.vocab_size = len(predicted_embeddings)

    out = model(
        input_ids=input_ids,
        params=params_with_updated_embeddings,
        dropout_rng=None,
        output_hidden_states=True,
        output_attentions=True,
        train=False,
    )

    out = FlaxCausalLMOutput(
        logits=jnp.where(
            mask[None, None, :],
            out.logits,
            utils.get_large_negative_number(out.logits.dtype),
        ),
        hidden_states=out.hidden_states,
        attentions=out.attentions,
    )

    # TODO: impl bias
    # if training_args.learnable_bias:
    #     logits = logits + biases[None, None, :]

    return predicted_embeddings, out


class TrainState(train_state.TrainState):
    train_mask: Any
    original_n_vocab: int


def main(args: TrainZettHnArgs):
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

    teacher_config = get_config(**asdict(args.model))
    student_config = get_config(**asdict(args.model))
    dtype = getattr(jnp, args.dtype)

    dataset = data.get_dataset(**args.data, seed=args.seed)
    if args.ppl_eval_data is not None:
        ppl_eval_data = data.get_dataset(**args.ppl_eval_data, seed=args.seed)
    else:
        ppl_eval_data = None

    model_kwargs = asdict(args.model)

    tokenizer_name = model_kwargs.pop("tokenizer_name")
    original_tokenizer = load_byteify_tokenizer(tokenizer_name)
    hn_tokenizer = load_byteify_tokenizer(tokenizer_name)

    # make sure pad token is set, we need it for the hypernet mask
    teacher_config.pad_token_id = hn_tokenizer.pad_token_id
    student_config.pad_token_id = hn_tokenizer.pad_token_id
    teacher_config.max_length = args.collator.block_size
    student_config.max_length = args.collator.block_size

    teacher_config.mesh = mesh
    student_config.mesh = mesh

    # TODO: add MLM support
    teacher_model = FlaxAutoModelForCausalLM.from_config(
        teacher_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.collator.block_size),
    )
    student_model = FlaxAutoModelForCausalLM.from_config(
        student_config,
        dtype=dtype,
        _do_init=False,
        input_shape=(args.n_data_parallel, args.collator.block_size),
    )
    model_params = param.load_params(**model_kwargs)

    n_embd = param.get(
        model_params, param.get_input_embedding_path(teacher_config.model_type)
    ).shape[-1]

    hypernet = Hypernet(
        dtype=dtype,
        hidden_size=n_embd,
        num_embeddings=1 if teacher_config.tie_word_embeddings else 2,
        max_seq_length=args.collator.hn_surface_maxlen,
        **asdict(args.hypernet),
    )

    # TODO: impl bias
    if args.compat:
        import tokenkit.compat.hypernet

        del teacher_config.mesh
        hn_config = copy.deepcopy(teacher_config)
        teacher_config.mesh = mesh
        # write defaults
        for key, value in asdict(tokenkit.compat.hypernet.HypernetArgs()).items():
            setattr(hn_config, key, value)

        hn_config.n_embd = n_embd
        hn_config.hn_use_attention_mask = hypernet.use_attention_mask
        hn_config.hn_rescale_embeddings = True
        hn_config.hn_model_name_or_path = "roberta-base"
        hn_config.hn_surface_maxlen = args.collator.hn_surface_maxlen
        hn_config.hn_n_layers = hypernet.num_layers
        hn_config.hn_hidden_size = hypernet.hidden_size
        hn_config.hn_intermediate_size = (
            hypernet.hidden_size * hypernet.hidden_expansion_factor
        )
        hn_config.hn_num_attention_heads = hypernet.num_heads
        hn_config.hn_n_extra_tokens = teacher_config.vocab_size - len(hn_tokenizer)
        hn_config.hn_embed_using_source_embeddings = True
        hn_config.separate_out_embeddings = not teacher_config.tie_word_embeddings

        hypernet = tokenkit.compat.hypernet.Hypernet(hn_config, dtype=dtype)

    optimizer_kwargs = args.optimizer
    learning_rate_fn = lr.linear_warmup_cosine_decay_with_linear_prefix(
        optimizer_kwargs.pop("learning_rate"),
        args.steps,
        args.warmup_steps,
        prefix_steps=args.identity_steps,
        prefix_lr=args.identity_lr,
    )

    def init_state(model_params):
        # TODO: impl tied embeddings
        model_params, input_embeddings = param.pop(
            model_params, param.get_input_embedding_path(teacher_config.model_type)
        )
        if teacher_config.tie_word_embeddings:
            embeddings = input_embeddings[:, jnp.newaxis, :]
        else:
            model_params, output_embeddings = param.pop(
                model_params, param.get_output_embedding_path(teacher_config.model_type)
            )
            embeddings = jnp.stack([input_embeddings, output_embeddings.T], axis=1)

        # pad to multiple
        original_n_vocab = embeddings.shape[0]
        n_pad = utils.get_n_pad(original_n_vocab, args.pad_to_multiple_of)

        embeddings = jnp.pad(
            embeddings,
            ((0, n_pad), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        hypernet_params = hypernet.init(
            jax.random.PRNGKey(args.seed),
            embeddings[:, None, :, :],
            jnp.ones((len(embeddings), 1), dtype=bool),
        )

        if args.compat:
            hypernet_params = hypernet.apply(
                hypernet_params,
                embeddings[:, None, :, :],  # add seq dimension
                embeddings[:, 0],
                embeddings[:, 1] if not teacher_config.tie_word_embeddings else None,
                method=hypernet.init_rescaler,
            )

        params = {
            "model": model_params,
            "hypernet": hypernet_params,
            "embeddings": embeddings,
        }
        if args.train_embeddings:
            # keep around a copy of the original embeddings for the teacher (condition could be tightened)
            params["original_embeddings"] = jnp.copy(params["embeddings"])

        if args.train_model_mode == "lora":
            params["model_lora"] = lora.init_lora_params(
                args,
                params["model"],
                model_type=student_config.model_type,
                seed=args.seed,
            )

        n_extra_embeddings = 256  # allocate space for 256 byte fallback embeddings
        params["extra_embeddings"] = jax.random.normal(
            jax.random.PRNGKey(args.seed),
            (n_extra_embeddings, *embeddings.shape[1:]),
            dtype=embeddings.dtype,
        ) * embeddings.std(axis=0, keepdims=True) + embeddings.mean(
            axis=0, keepdims=True
        )

        train_mask = utils.label_by_prefix(
            params,
            [
                [("hypernet", "non_trainable"), False],  # pax / praxis convention
                [("hypernet",), True],
                [("hypernet", "in_scaler"), False],  # compat,
                [("hypernet", "out_scaler"), False],  # compat,
                [("hypernet", "scaler"), False],  # compat,
                [("model",), True if args.train_model_mode == "full" else False],
                [("model_lora",), True],
                [("embeddings",), args.train_embeddings],
                [("original_embeddings",), False],
                [("extra_embeddings",), True],
            ],
        )

        params = jax.tree.map(
            lambda x, trainable: x.astype(jnp.float32) if trainable else x,
            params,
            train_mask,
        )

        optimizer = opt.get_optimizer(train_mask, learning_rate_fn, **optimizer_kwargs)

        return TrainState.create(
            apply_fn=hypernet.apply,
            params=params,
            train_mask=train_mask,
            original_n_vocab=original_n_vocab,
            tx=optimizer,
        )

    state_shape = jax.eval_shape(init_state, model_params)

    shard_patterns = {
        **sharding.get_shard_patterns(teacher_config.model_type),
        **sharding.get_shard_patterns("compat_hypernet" if args.compat else "hypernet"),
    }
    state_shardings = sharding.get_sharding_fn(shard_patterns, mesh)(state_shape)
    state = jax.jit(init_state, out_shardings=state_shardings)(model_params)

    space_mask_original = np.zeros(state.params["embeddings"].shape[0], dtype=bool)
    space_mask_original_unpadded = utils.get_space_mask(
        hn_tokenizer, args.space_mask_mode
    )
    space_mask_original[: len(space_mask_original_unpadded)] = (
        space_mask_original_unpadded
    )

    utils.param_report(state.params, state.train_mask)
    train_mask = jax.device_get(state.train_mask)

    def identity_train_step(state, batch):
        surface_forms = batch.pop("target_surface_forms")
        priors = batch.pop("target_priors")
        ids_to_embed = batch.pop("ids_to_embed")
        lang_index = batch.pop("lang_index")

        if args.train_embeddings:
            source_embeddings = state.params["original_embeddings"]
        else:
            source_embeddings = state.params["embeddings"]

        def compute_loss(non_trainable_params, *trainable_params):
            params = jax.tree.map(
                lambda *args: next(x for x in args if x is not None),
                *trainable_params,
                non_trainable_params,
                is_leaf=lambda x: x is None,
            )

            predicted_embeddings = predict_embeddings(
                state.apply_fn,
                mesh,
                state.original_n_vocab,
                params["hypernet"],
                surface_forms,
                surface_forms != hn_tokenizer.pad_token_id,
                source_embeddings,
                params["extra_embeddings"],
                None,
                None,
            )

            target_embeddings = jnp.take(source_embeddings, ids_to_embed, axis=0)
            loss = jnp.square(predicted_embeddings - target_embeddings).sum(-1).mean()

            return loss

        trainable_params = jax.tree.map(
            lambda x, m: x if m else None, state.params, train_mask
        )

        grad_fn = jax.value_and_grad(compute_loss, argnums=1)
        loss, grad = grad_fn(state.params, trainable_params)

        grad = jax.tree.map(
            lambda g, p: g if g is not None else jnp.zeros_like(p),
            grad,
            state.params,
            is_leaf=lambda x: x is None,
        )
        new_state = state.apply_gradients(grads=grad)

        metrics = {
            "identity_loss": loss,
            "learning_rate": learning_rate_fn(state.step),
        }
        return new_state, metrics

    def compute_lexical_loss(
        predicted_embeddings, target_surface_forms, source_embeddings, epsilon=1e-8
    ):
        lexical_overlap_mask = (
            target_surface_forms[:, 1:] == hn_tokenizer.pad_token_id
        ).all(axis=1)
        target_embeddings = source_embeddings[target_surface_forms[:, 0]]

        def distance_fn(x, y):
            return jnp.square(x - y).sum(axis=-1)

        # shape: [n_vocab, n_embeddings]
        lexical_loss = (
            distance_fn(predicted_embeddings, target_embeddings)
            * lexical_overlap_mask[:, None]
        )
        # shape: [n_embeddings]
        lexical_loss = (
            lexical_loss.mean(axis=0)
            / (lexical_overlap_mask.mean() + epsilon)
            / (jnp.linalg.norm(target_embeddings, axis=-1).mean(0) + epsilon)
        )
        return lexical_loss.mean(), lexical_overlap_mask.mean()

    def train_step(state, batch):
        target_surface_forms = batch["target_surface_forms"]
        target_priors = batch["target_priors"]

        if args.train_embeddings:
            source_embeddings = state.params["original_embeddings"]
        else:
            source_embeddings = state.params["embeddings"]

        def compute_loss(non_trainable_params, *trainable_params, epsilon=1e-8):
            params = jax.tree.map(
                lambda *args: next(x for x in args if x is not None),
                *trainable_params,
                non_trainable_params,
                is_leaf=lambda x: x is None,
            )

            scalar_report = {}

            (
                predicted_embeddings,
                student_out,
            ) = compute_embeddings_and_out(
                args,
                params,
                batch["input_ids_new"],
                target_surface_forms,
                target_surface_forms != hn_tokenizer.pad_token_id,
                target_priors,
                batch["mask"],
                batch["special_indices"],
                batch["special_indices_in_reference"],
                source_embeddings=source_embeddings,
                extra_embeddings=params["extra_embeddings"],
                original_n_vocab=state.original_n_vocab,
                hypernet_fn=state.apply_fn,
                model=student_model,
            )

            need_teacher = len([loss for loss in args.losses if loss != "sft"]) > 0
            if need_teacher:
                teacher_config.vocab_size = len(source_embeddings)

                teacher_out = teacher_model(
                    input_ids=batch["input_ids_original"],
                    params=assign_embeddings(
                        state.params["model"], source_embeddings, teacher_config
                    ),
                    dropout_rng=None,
                    output_hidden_states=True,
                    output_attentions=True,
                    train=False,
                )

                teacher_logits = teacher_out.logits.astype(jnp.float32)
                teacher_mask = (
                    jnp.arange(teacher_logits.shape[-1]) < state.original_n_vocab
                )
                teacher_logits = jnp.where(
                    teacher_mask[None, None, :],
                    teacher_logits,
                    utils.get_large_negative_number(teacher_logits.dtype),
                )
                teacher_logprobs = jnp.clip(
                    jax.nn.log_softmax(teacher_logits, axis=-1), max=0.0
                )
                teacher_probs = jnp.exp(teacher_logprobs)
            else:
                teacher_out = teacher_logits = teacher_logprobs = teacher_probs = (
                    teacher_mask
                ) = None

            student_logits = student_out.logits.astype(jnp.float32)
            student_logprobs = jnp.clip(
                jax.nn.log_softmax(student_logits, axis=-1), max=0.0
            )
            student_probs = jnp.exp(student_logprobs)

            loss_args = losses.LossArgs(
                params=params,
                batch=batch,
                global_batch=batch,  # TODO: impl
                teacher_config=teacher_config,
                new_config=student_config,
                teacher_out=teacher_out,
                student_out=student_out,
                tokenizer_teacher=original_tokenizer,
                tokenizer_new=original_tokenizer,  # TODO: impl pass eos_token and pad_token ids
                teacher_probs=teacher_probs,
                teacher_logprobs=teacher_logprobs,
                teacher_logits=teacher_logits,
                student_probs=student_probs,
                student_logprobs=student_logprobs,
                student_logits=student_logits,
                predicted_embeddings=predicted_embeddings,
                scalar_report=scalar_report,
                space_mask_teacher=space_mask_original,
                space_mask_new=batch["space_mask"],
                logit_mask_teacher=(
                    jnp.where(
                        teacher_mask,
                        0.0,
                        utils.get_large_negative_number(teacher_logits.dtype),
                    )
                    if teacher_mask is not None
                    else None
                ),
                logit_mask_new=jnp.where(
                    batch["mask"],
                    0.0,
                    utils.get_large_negative_number(student_logits.dtype),
                ),
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
                elif loss == "lexical":
                    current_loss, _ = compute_lexical_loss(
                        predicted_embeddings, target_surface_forms, source_embeddings
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

                loss_values = loss_values.at[loss_idx].set(weight * current_loss)

                scalar_report[f"loss/{loss}"] = current_loss
                scalar_report[f"loss/{loss}_weight"] = weight

            # report lexical loss & overlap (regardless of whether it's minimized or not)
            lexical_loss, lexical_overlap = compute_lexical_loss(
                predicted_embeddings, target_surface_forms, source_embeddings
            )
            scalar_report["loss/lexical"] = lexical_loss
            scalar_report["loss/lexical_overlap"] = lexical_overlap

            return loss_values, scalar_report

        trainable_params = jax.tree.map(
            lambda x, m: x if m else None, state.params, train_mask
        )
        # last layer is usually not trainable (at least training not impl at the moment)
        # here, a question arises: should we just approx. the gradient using all (potentially trainable) layers?
        # this might give a different (more or less stable?) estimate also in the case of LoRA.
        # should investigate more, make this an option, and gain some intuition on the difference.
        last_layer_params = jax.tree.map(
            lambda x, m: x if m else None,
            state.params,
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
                state.params, last_layer_params, trainable_params
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
                state.params, last_layer_params, trainable_params
            )

            for loss_idx, loss_name in enumerate(args.losses):
                scalar_report[f"loss/{loss_name}_approx_grad_norm"] = approx_grad_norm[
                    loss_idx
                ]
                scalar_report[f"loss/{loss_name}_approx_loss_weight"] = (
                    approx_loss_weights[loss_idx] / denominator
                )

            grad = jax.tree.map(
                lambda x, y: x if x is not None else y,
                last_layer_grad,
                non_last_layer_grad,
                is_leaf=lambda x: x is None,
            )
        else:
            raise NotImplementedError()

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
        target_surface_forms = batch["target_surface_forms"]
        target_priors = batch["target_priors"]

        if args.train_embeddings:
            source_embeddings = state.params["original_embeddings"]
        else:
            source_embeddings = state.params["embeddings"]

        def compute_loss(params):
            (
                _,
                student_out,
            ) = compute_embeddings_and_out(
                args,
                params,
                batch["input_ids_new"],
                target_surface_forms,
                target_surface_forms != hn_tokenizer.pad_token_id,
                target_priors,
                batch["mask"],
                batch["special_indices"],
                batch["special_indices_in_reference"],
                source_embeddings=source_embeddings,
                extra_embeddings=params["extra_embeddings"],
                config=student_config,
                original_n_vocab=state.original_n_vocab,
                hypernet_fn=state.apply_fn,
                model=student_model,
            )

            return losses.cross_entropy(
                student_out.logits,
                batch["input_ids_new"],
                batch["loss_mask_new"],
            )

        loss = compute_loss(state.params)

        metrics = {
            "loss": loss,
        }
        return metrics

    initial_texts = [
        x["text"] for x in dataset.get_texts(args.collator.tokenizer_batch_size)
    ]
    collator = collators.TokenizerSamplerCollator(
        hn_tokenizer,
        args.collator,
        batch_size=args.data["batch_size"],
        initial_texts=initial_texts,  # TODO: impl !mix_languages
        with_consistent_whitespace=False,
        with_alignments=True,
        original_tokenizer=original_tokenizer,
        space_mask_mode=args.space_mask_mode,
    )
    batch_shardings = jax.tree.map(
        lambda x: NamedSharding(mesh, x), collator.get_batch_pspecs()
    )

    eval_tokenizers = []

    for targs in args.eval.tokenizers:
        eval_tokenizer = load_byteify_tokenizer(targs["tokenizer"])
        special_indices = np.array(eval_tokenizer.all_special_ids)
        special_indices_in_reference = np.array(
            [
                hn_tokenizer.convert_tokens_to_ids(token)
                for token in eval_tokenizer.all_special_tokens
            ]
        )
        n_pad = utils.get_n_pad(len(eval_tokenizer), args.pad_to_multiple_of)

        eval_surface_forms, _ = utils.get_surface_form_matrix(
            eval_tokenizer,
            maxlen=args.collator.hn_surface_maxlen,
            hn_tokenizer=hn_tokenizer,
            padding=n_pad,
        )
        eval_logit_mask = np.zeros((len(eval_tokenizer) + n_pad,), dtype=bool)
        eval_logit_mask[: len(eval_tokenizer)] = True

        eval_tokenizers.append(
            {
                "tokenizer": eval_tokenizer,
                "surface_forms": eval_surface_forms,
                "logit_mask": eval_logit_mask,
                "special_indices": special_indices,
                "special_indices_in_reference": special_indices_in_reference,
                "name": targs["name"],
            }
        )

    if args.identity_steps > 0:
        # TODO: fix / update
        identity_collator_args = copy.deepcopy(args.collator)
        identity_collator_args.do_tokenizer_sampling = False
        identity_collator_args.hn_surface_maxlen = 1

        if args.collator.identity_n_token_subsample is not None:
            identity_collator_args.n_token_subsample = (
                args.collator.identity_n_token_subsample
            )

        identity_collator = collators.TokenizerSamplerCollator(
            hn_tokenizer,
            identity_collator_args,
            fixed_tokenizer=original_tokenizer,
            with_consistent_whitespace=False,
        )

        identity_train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.zeros(args.identity_steps + 1)
            ),  # dummy data, +1 for initial step
            batch_size=1,  # batched internally
            collate_fn=partial(identity_collator, for_identity_step=True),
        )
        identity_batch_shardings = jax.tree.map(
            lambda x: NamedSharding(mesh, x), identity_collator.get_identity_batch_pspecs()
        )
    else:
        identity_train_dataloader = None
        identity_batch_shardings = None

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

    def track_metrics(batch):
        # TODO: impl
        _batch_metrics = batch.pop("metrics", None)
        _lang_code = batch.pop("lang_code")

        return batch

    identity_diter = (
        iter(identity_train_dataloader) if identity_train_dataloader else None
    )
    diter = iter(train_dataloader)
    first_batch = track_metrics(next(diter))

    if args.identity_steps > 0:
        first_identity_batch = track_metrics(next(identity_diter))

        jidentity_train_step = jax.jit(
            identity_train_step,
            donate_argnums=(0,),
            in_shardings=(state_shardings, identity_batch_shardings),
            out_shardings=(state_shardings, None),
        )
    else:
        first_identity_batch = None
        jidentity_train_step = None

    jtrain_step = jax.jit(
        train_step,
        donate_argnums=(0,),
        in_shardings=(state_shardings, batch_shardings),
        out_shardings=(state_shardings, None),
    )
    jeval_step = jax.jit(
        eval_step,
        in_shardings=(state_shardings, batch_shardings),
        out_shardings=None,
    )
    jpredict_embeddings = jax.jit(
        predict_embeddings,
        # TODO: move to sharding or not?
        in_shardings=(
            None,  # n_vocab,
            state_shardings.params["hypernet"],  # hypernet_params
            batch_shardings["target_surface_forms"],  # surface_forms
            batch_shardings["target_surface_forms"],  # surface_forms_attention_mask
            state_shardings.params["embeddings"],  # source_embeddings
            state_shardings.params["embeddings"],  # extra_embeddings
            None,  # special_indices
            None,  # special_indices_in_reference
        ),
        static_argnums=(0, 1),
        out_shardings=state_shardings.params["embeddings"],  # predicted_embeddings
    )
    if args.train_model_mode == "lora":
        jmaterialize_lora = jax.jit(
            lora.materialize_lora,
            in_shardings=(
                state_shardings.params["model"],
                state_shardings.params["model_lora"],
            ),
            out_shardings=state_shardings.params["model"],
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
            batch = track_metrics(batch)
            batch = sharding.sync_across_devices(batch)
            batch = sharding.to_global_array(batch, batch_shardings)

            step_metrics = jeval_step(state, batch)
            eval_metrics.append(step_metrics)

        eval_metrics = jax.tree.map(np.mean, common_utils.stack_forest(eval_metrics))
        return eval_metrics

    if args.do_cost_analysis:
        compiled_train_step_fn = jtrain_step.lower(state, first_batch).compile()
        flops_per_step = compiled_train_step_fn.cost_analysis()["flops"]
        memory_per_step = (
            compiled_train_step_fn.memory_analysis().output_size_in_bytes
            + compiled_train_step_fn.memory_analysis().temp_size_in_bytes
        )
        logger.info("TFLOPs per step: %.2f", flops_per_step / (10**12))
        logger.info("Memory (MB) per step: %.2f", memory_per_step / (1024**2))
        sys.exit()

    train_metrics = []
    start_time = time.time()

    upload_executor = None
    upload_name = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + args.name

    for step in tqdm(range(args.steps)):
        current_diter = identity_diter if step < args.identity_steps else diter
        current_first_batch = (
            first_identity_batch if step < args.identity_steps else first_batch
        )
        current_batch_shardings = (
            identity_batch_shardings if step < args.identity_steps else batch_shardings
        )
        current_step_fn = (
            jidentity_train_step if step < args.identity_steps else jtrain_step
        )

        if jax.process_index() == 0:
            try:
                batch = track_metrics(next(current_diter))
            except StopIteration:
                if step < args.identity_steps:
                    identity_diter = iter(identity_train_dataloader)
                    batch = track_metrics(next(identity_diter))
                else:
                    diter = iter(train_dataloader)
                    batch = track_metrics(next(diter))
        else:
            batch = current_first_batch

        batch = sharding.sync_across_devices(batch)
        batch = sharding.to_global_array(batch, current_batch_shardings)

        if args.dry_run:
            continue

        state, step_metrics = current_step_fn(state, batch)
        train_metrics.append(step_metrics)

        # must be the first call because it logs to time steps smaller than the current step
        if (step + 1) % args.sync_interval == 0:
            stacked_train_metrics = jax.tree.map(
                jax.device_get, common_utils.stack_forest(train_metrics)
            )

            if jax.process_index() == 0:
                end_step = step + 1
                start_step = end_step - args.sync_interval
                for i in range(start_step, end_step, args.log_interval):
                    for key, values in stacked_train_metrics.items():
                        avg_value = values[
                            i - start_step : i - start_step + args.log_interval
                        ].mean()
                        utils.log(
                            {f"train/{key}": avg_value}, step=i + args.log_interval
                        )

            train_metrics = []

        if (step + 1) % args.eval_interval == 0 or (
            step == 0 and args.eval_at_step_zero
        ):
            # TODO: probably extract into eval function doing everything here
            if ppl_eval_dataloader is not None:
                ppl_metrics = eval_loop(ppl_eval_dataloader)
                ppl_metrics = {f"ppl_eval/{k}": v for k, v in ppl_metrics.items()}
                if jax.process_index() == 0:
                    logger.info("PPL Eval:")
                utils.log(ppl_metrics, step=step + 1)

            lm_eval_metrics = {}

            eval_kwargs = asdict(args.eval)
            eval_kwargs.pop("tokenizers", None)
            for eval_tokenizer in eval_tokenizers:
                predicted_embeddings = jpredict_embeddings(
                    # TODO: pass state?
                    state.apply_fn,
                    mesh,
                    state.original_n_vocab,
                    state.params["hypernet"],
                    sharding.to_global_array(
                        eval_tokenizer["surface_forms"],
                        batch_shardings["target_surface_forms"],
                    ),
                    sharding.to_global_array(
                        eval_tokenizer["surface_forms"] != hn_tokenizer.pad_token_id,
                        batch_shardings["target_surface_forms"],
                    ),
                    # TODO: remove train_embeddings everywhere?
                    (
                        state.params["original_embeddings"]
                        if args.train_embeddings
                        else state.params["embeddings"]
                    ),
                    state.params["extra_embeddings"],
                    sharding.to_global_array(eval_tokenizer["special_indices"]),
                    sharding.to_global_array(
                        eval_tokenizer["special_indices_in_reference"]
                    ),
                )
                model_params_with_embeddings = assign_embeddings(
                    jmaterialize_lora(
                        state.params["model"],
                        state.params.get("model_lora"),
                        args.model_lora_alpha,
                    ),
                    predicted_embeddings,
                    student_config,
                )

                student_config.vocab_size = len(predicted_embeddings)
                current_lm_eval_metrics, post_eval_params_buffer = eval.evaluate(
                    model=student_model,
                    config=student_config,
                    params=model_params_with_embeddings,
                    tokenizer=eval_tokenizer["tokenizer"],
                    logit_mask=eval_tokenizer["logit_mask"],
                    **eval_kwargs,
                )
                state.params["model"] = jdematerialize_lora(
                    param.unassign_embeddings(post_eval_params_buffer, student_config),
                    state.params.get("model_lora"),
                    args.model_lora_alpha,
                )

                lm_eval_metrics.update(
                    {
                        f"lm_eval/{eval_tokenizer['name']}/" + "_".join(k): v
                        for k, v in traverse_util.flatten_dict(
                            current_lm_eval_metrics
                        ).items()
                    }
                )

                if jax.process_index() == 0:
                    logger.info(f"LM Eval: {eval_tokenizer['name']}")
                    logger.info(pformat(current_lm_eval_metrics))

            if jax.process_index() == 0:
                utils.log(lm_eval_metrics, step=step + 1)

        if (step + 1) % args.save_interval == 0 or (
            step == 0 and args.save_at_step_zero
        ):
            if upload_executor is not None:
                upload_executor.shutdown(wait=True)
            multihost_utils.sync_global_devices("uploaded previous checkpoint")

            checkpoint.save(
                output_dir / "params.msgpack",
                state.params,
                state_shardings.params,
                mesh,
                train_mask,
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
                    {"step": step + 1, "time": time.time() - start_time},
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

    main(parse_args.parse_args(TrainZettHnArgs))
