from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import common_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tokenkit import baseline_utils, utils
from tokenkit.models import param

def get_last_index_per_column(matrix):
    matrix_last_only = (
        jnp.asarray(matrix).at[:, :-1].set(matrix[:, :-1] & ~matrix[:, 1:])
    )
    return matrix_last_only.argmax(-2), matrix_last_only.max(-2) != 0


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


@dataclass
class LossArgs:
    params: Any
    batch: Any
    global_batch: Any
    teacher_config: Any
    new_config: Any
    teacher_out: Any
    student_out: Any
    tokenizer_teacher: Any
    tokenizer_new: Any
    teacher_probs: Any
    teacher_logprobs: Any
    teacher_logits: Any
    student_probs: Any
    student_logprobs: Any
    student_logits: Any
    predicted_embeddings: Any
    scalar_report: Any
    space_mask_teacher: Any
    space_mask_new: Any
    logit_mask_teacher: Any
    logit_mask_new: Any


def compute_alm_latents_loss(args, loss_args, epsilon=1e-8):
    if args.latents_chunks == "naive":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_unconstrained"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_unconstrained"]
        )
        _, global_mask = get_last_index_per_column(
            loss_args.global_batch["alignment_matrix_a_unconstrained"]
        )
    elif args.latents_chunks == "space":
        alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
            loss_args.batch["alignment_matrix_b_space"]
        )
        alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
            loss_args.batch["alignment_matrix_a_space"]
        )
        _, global_mask = get_last_index_per_column(
            loss_args.global_batch["alignment_matrix_a_space"]
        )

    if "last_hidden_state" in args.latents_to_align:
        layer_indices = [(-1, -1)]
    elif "all_hidden_states" in args.latents_to_align:
        layer_indices = [
            (i, i + args.n_prefix_layers)
            for i in range(param.get_num_layers(loss_args.new_config) + 1)
        ]  # +1 for embeddings
    else:
        layer_indices = []

    hidden_state_latent_loss = 0.0
    attention_latent_loss = 0.0

    for teacher_idx, student_idx in layer_indices:
        t_aligned_last_hidden_state = jnp.take_along_axis(
            loss_args.teacher_out.hidden_states[teacher_idx],
            alignment_matrix_b_last_only_index[..., None],
            axis=-2,
        )
        s_aligned_last_hidden_state = jnp.take_along_axis(
            loss_args.student_out.hidden_states[student_idx],
            alignment_matrix_a_last_only_index[..., None],
            axis=-2,
        )

        if args.latents_do_project:
            projector_latents_params = loss_args.params["model"]["projector_latents"]
            s_aligned_last_hidden_state = (
                jnp.matmul(
                    s_aligned_last_hidden_state,
                    projector_latents_params["kernel"],
                )
                + projector_latents_params["bias"]
            )

        if args.latents_normalization.startswith("l2"):
            elementwise_layer_latent_loss = jnp.square(
                s_aligned_last_hidden_state - t_aligned_last_hidden_state
            )
        elif args.latents_normalization.startswith("l1"):
            elementwise_layer_latent_loss = jnp.abs(
                s_aligned_last_hidden_state - t_aligned_last_hidden_state
            )

        layer_latent_loss = elementwise_layer_latent_loss * mask[..., None]
        if args.latents_normalization == "l2":
            layer_latent_loss /= (
                jnp.square(t_aligned_last_hidden_state * mask[..., None]).mean()
                / mask.mean()
            ) + epsilon
        elif args.latents_normalization == "l2_channelwise":
            layer_latent_loss /= (
                jnp.square(t_aligned_last_hidden_state * mask[..., None]).mean(
                    [0, 1], keepdims=True
                )
                / mask.mean()
            ) + epsilon
        elif args.latents_normalization == "l1":
            layer_latent_loss /= (
                jnp.abs(t_aligned_last_hidden_state * mask[..., None]).mean()
                / mask.mean()
            ) + epsilon
        elif args.latents_normalization == "l1_channelwise":
            layer_latent_loss /= (
                jnp.abs(t_aligned_last_hidden_state * mask[..., None]).mean(
                    [0, 1], keepdims=True
                )
                / mask.mean()
            ) + epsilon

        layer_latent_loss = layer_latent_loss.mean() / global_mask.mean()
        hidden_state_latent_loss += layer_latent_loss / len(layer_indices)

    if "qkv" in args.latents_to_align:
        for layer_idx in range(len(loss_args.teacher_out.attentions)):
            teacher_idx = layer_idx
            student_idx = layer_idx + args.n_prefix_layers

            teacher_qkv = jnp.concatenate(
                loss_args.teacher_out.attentions[teacher_idx], -1
            )
            student_qkv = jnp.concatenate(
                loss_args.student_out.attentions[student_idx], -1
            )

            t_aligned_qkv = jnp.take_along_axis(
                teacher_qkv,
                alignment_matrix_b_last_only_index[..., None],
                axis=-2,
            )
            s_aligned_qkv = jnp.take_along_axis(
                student_qkv,
                alignment_matrix_a_last_only_index[..., None],
                axis=-2,
            )

            elementwise_layer_latent_loss = jnp.square(s_aligned_qkv - t_aligned_qkv)
            layer_latent_loss = elementwise_layer_latent_loss * mask[..., None]
            if args.latents_normalization == "l2":
                layer_latent_loss /= (
                    jnp.square(t_aligned_qkv * mask[..., None]).mean() / mask.mean()
                )
            elif args.latents_normalization == "l2_channelwise":
                layer_latent_loss /= (
                    jnp.square(t_aligned_qkv * mask[..., None]).mean(
                        [0, 1], keepdims=True
                    )
                    / mask.mean()
                )

            layer_latent_loss = layer_latent_loss.mean() / global_mask.mean()
            attention_latent_loss += layer_latent_loss / len(
                loss_args.teacher_out.attentions
            )

    loss = hidden_state_latent_loss + attention_latent_loss

    loss_args.scalar_report["hidden_state_latent_loss"] = hidden_state_latent_loss
    loss_args.scalar_report["attention_latent_loss"] = attention_latent_loss
    return loss


def compute_sft_loss(args, loss_args):
    sft_loss = cross_entropy(
        loss_args.student_logits,
        loss_args.batch["input_ids_new"],
        loss_args.batch["loss_mask_new"],
        denom=loss_args.global_batch["loss_mask_new"][:, 1:].mean(),
    )
    return sft_loss


def log1mexp(x):
    """Computes log(1 - exp(x)) in a numerically stable way for x < 0."""
    # For x < log(0.5), use log1p(-exp(x)) directly
    # For x >= log(0.5), use log(-expm1(x)) to avoid precision issues
    log_half = -jnp.log(2)
    return jnp.where(x < log_half, jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x)))


def compute_alm_loss(chunk_kind, args, loss_args, epsilon=1e-6):
    original_shift_labels = loss_args.batch["input_ids_original"][..., 1:]

    if chunk_kind == "unconstrained":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unconstrained"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unconstrained"]
        global_alignment_matrix_a = loss_args.global_batch[
            "alignment_matrix_a_unconstrained"
        ]
        global_alignment_matrix_b = loss_args.global_batch[
            "alignment_matrix_b_unconstrained"
        ]
    elif chunk_kind == "unbiased":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_unbiased"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_unbiased"]
        global_alignment_matrix_a = loss_args.global_batch[
            "alignment_matrix_a_unbiased"
        ]
        global_alignment_matrix_b = loss_args.global_batch[
            "alignment_matrix_b_unbiased"
        ]
    elif chunk_kind == "space":
        alignment_matrix_a = loss_args.batch["alignment_matrix_a_space"]
        alignment_matrix_b = loss_args.batch["alignment_matrix_b_space"]
        global_alignment_matrix_a = loss_args.global_batch["alignment_matrix_a_space"]
        global_alignment_matrix_b = loss_args.global_batch["alignment_matrix_b_space"]
    else:
        raise ValueError(f"Unknown chunk kind: {chunk_kind}")

    if args.alm_diff_fn == "abs":
        diff_fn = lambda log_y_true, log_y_pred: jnp.abs(log_y_true - log_y_pred)
    elif args.alm_diff_fn == "binary_ce":

        def binary_ce(log_y_true, log_y_pred):
            log_y_true = (log_y_true.astype(jnp.float32) / args.binarization_temp) - epsilon
            log_y_pred = (log_y_pred.astype(jnp.float32) / args.binarization_temp) - epsilon

            return -(
                jnp.exp(log_y_true) * log_y_pred
                + (-jnp.expm1(log_y_true) * log1mexp(log_y_pred))
            )

        diff_fn = binary_ce
    elif args.alm_diff_fn == "reverse_binary_kl":

        def reverse_binary_kl(log_y_true, log_y_pred):
            log_y_true = (log_y_true.astype(jnp.float32) / args.binarization_temp) - epsilon
            log_y_pred = (log_y_pred.astype(jnp.float32) / args.binarization_temp) - epsilon

            return jnp.exp(log_y_pred) * (log_y_pred - log_y_true) + (
                -jnp.expm1(log_y_pred) * (log1mexp(log_y_pred) - log1mexp(log_y_true))
            )

        diff_fn = reverse_binary_kl
    elif args.alm_diff_fn == "binary_kl_temp_limit":

        def binary_kl_temp_limit(log_y_true, log_y_pred):
            log_y_true = log_y_true - epsilon
            log_y_pred = log_y_pred - epsilon

            return (log_y_true - log_y_pred) + (
                log_y_true * jnp.log(-log_y_pred) - log_y_true * jnp.log(-log_y_true)
            )

        diff_fn = binary_kl_temp_limit
    elif args.alm_diff_fn == "abs_exp":

        def abs_exp(log_y_true, log_y_pred):
            log_y_true = (log_y_true.astype(jnp.float32) / args.binarization_temp) - epsilon
            log_y_pred = (log_y_pred.astype(jnp.float32) / args.binarization_temp) - epsilon

            return jnp.abs(jnp.exp(log_y_true) - jnp.exp(log_y_pred))

        diff_fn = abs_exp
    elif args.alm_diff_fn == "renyi":

        def renyi(log_y_true, log_y_pred):
            log_y_true = log_y_true.astype(jnp.float32) - epsilon
            log_y_pred = log_y_pred.astype(jnp.float32) - epsilon

            y_true = jnp.exp(log_y_true)
            y_pred = jnp.exp(log_y_pred)

            log_one_minus_y_true = jnp.log1p(-y_true)
            log_one_minus_y_pred = jnp.log1p(-y_pred)

            term1 = args.renyi_alpha * log_y_true + (1 - args.renyi_alpha) * log_y_pred
            term2 = (
                args.renyi_alpha * log_one_minus_y_true
                + (1 - args.renyi_alpha) * log_one_minus_y_pred
            )

            return jnp.logaddexp(term1, term2) / (args.renyi_alpha - 1)

        diff_fn = renyi
    elif args.alm_diff_fn == "joschu_k2":
        def joschu_k2(log_y_true, log_y_pred):
            logr = (log_y_true - log_y_pred) / args.binarization_temp
            return (logr ** 2) / 2

        diff_fn = joschu_k2
    elif args.alm_diff_fn == "joschu_k3":
        def joschu_k3(log_y_true, log_y_pred):
            logr = (log_y_true - log_y_pred) / args.binarization_temp

            return (jnp.exp(logr) - 1) - logr

        diff_fn = joschu_k3
    else:
        raise NotImplementedError(f"Unknown diff function: {args.diff_fn}")

    alignment_matrix_a = (
        alignment_matrix_a * loss_args.batch["loss_mask_new"][:, :, None]
    )
    alignment_matrix_b = (
        alignment_matrix_b * loss_args.batch["loss_mask_original"][:, :, None]
    )
    global_alignment_matrix_a = (
        global_alignment_matrix_a * loss_args.global_batch["loss_mask_new"][:, :, None]
    )
    global_alignment_matrix_b = (
        global_alignment_matrix_b
        * loss_args.global_batch["loss_mask_original"][:, :, None]
    )

    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
        alignment_matrix_b
    )
    alignment_matrix_a_last_only_index, mask = get_last_index_per_column(
        alignment_matrix_a
    )
    student_chunk_sums = alignment_matrix_a.sum(-2)
    teacher_chunk_sums = alignment_matrix_b.sum(-2)
    student_avg_chunk_lengths = (
        student_chunk_sums.mean() / (student_chunk_sums > 0).mean()
    )
    teacher_avg_chunk_lengths = (
        teacher_chunk_sums.mean() / (teacher_chunk_sums > 0).mean()
    )
    loss_args.scalar_report["student_avg_chunk_lengths"] = student_avg_chunk_lengths
    loss_args.scalar_report["teacher_avg_chunk_lengths"] = teacher_avg_chunk_lengths

    teacher_main_path_logprobs = jnp.squeeze(
        jnp.take_along_axis(
            loss_args.teacher_logprobs[:, :-1],
            original_shift_labels[..., None],
            axis=-1,
        ),
        -1,
    )
    t_aligned_main_logp = jnp.clip(
        jnp.squeeze(
            jnp.matmul(
                teacher_main_path_logprobs[:, None],
                alignment_matrix_b[:, 1:],
            ),
            1,
        ),
        max=0.0,
    )

    if "eos_as_space" in args.alm_mode:
        t_space_logp = loss_args.teacher_logprobs[
            :, :, loss_args.tokenizer_teacher.eos_token_id
        ]
    else:
        t_space_logp = jnp.clip(
            jnp.log(
                jnp.dot(loss_args.teacher_probs, loss_args.space_mask_teacher)
            ),
            max=0.0,
        )
    t_aligned_space_logp = jnp.take_along_axis(
        t_space_logp, alignment_matrix_b_last_only_index, axis=-1
    )
    new_shift_labels = loss_args.batch["input_ids_new"][..., 1:]
    student_main_path_logprobs = jnp.squeeze(
        jnp.take_along_axis(
            loss_args.student_logprobs[:, :-1],
            new_shift_labels[..., None],
            axis=-1,
        ),
        -1,
    )
    s_aligned_main_logp = jnp.clip(
        jnp.squeeze(
            jnp.matmul(
                student_main_path_logprobs[:, None],
                alignment_matrix_a[:, 1:],
            ),
            1,
        ),
        max=0.0,
    )

    if "eos_as_space" in args.alm_mode:
        s_space_logp = loss_args.student_logprobs[
            :, :, loss_args.tokenizer_new.eos_token_id
        ]
    else:
        s_space_logp = jnp.clip(
            jnp.log(jnp.dot(loss_args.student_probs, loss_args.space_mask_new)),
            max=0.0,
        )

    s_aligned_space_logp = jnp.take_along_axis(
        s_space_logp, alignment_matrix_a_last_only_index, axis=-1
    )

    aligned_count = alignment_matrix_b[:, 1:].sum(-2)
    global_aligned_count = global_alignment_matrix_b[:, 1:].sum(-2)

    if "merge_by_space_prob" in args.alm_mode:
        batch_size = t_aligned_space_logp.shape[0]
        chunk_count = t_aligned_space_logp.shape[-1]

        t_aligned_space_chunk_mask = (
            jnp.exp(t_aligned_space_logp) > args.tokenizer_pair_bias_threshold
        )
        chunk_merging_indices = jnp.cumsum(t_aligned_space_chunk_mask[:, ::-1], -1)[
            :, ::-1
        ]
        chunk_merging_indices = (
            chunk_merging_indices.max(-1, keepdims=True) - chunk_merging_indices
        )
        chunk_merging_values = aligned_count > 0
        chunk_merging_matrix = (
            jnp.zeros(
                (
                    batch_size * chunk_count,
                    chunk_count,
                ),
                dtype=alignment_matrix_a.dtype,
            )
            .at[jnp.arange(batch_size * chunk_count), chunk_merging_indices.reshape(-1)]
            .set(chunk_merging_values.reshape(-1))
            .reshape((batch_size, chunk_count, chunk_count))
        )
        chunk_merging_matrix_last_only_index, _ = get_last_index_per_column(
            chunk_merging_matrix
        )

        t_aligned_main_logp = jnp.squeeze(
            jnp.matmul(t_aligned_main_logp[:, None], chunk_merging_matrix), 1
        )
        s_aligned_main_logp = jnp.squeeze(
            jnp.matmul(s_aligned_main_logp[:, None], chunk_merging_matrix), 1
        )
        t_aligned_space_logp = jnp.take_along_axis(
            t_aligned_space_logp, chunk_merging_matrix_last_only_index, axis=-1
        )
        s_aligned_space_logp = jnp.take_along_axis(
            s_aligned_space_logp, chunk_merging_matrix_last_only_index, axis=-1
        )

        aligned_count = jnp.squeeze(aligned_count[:, None] @ chunk_merging_matrix, 1)
        global_aligned_count = jnp.squeeze(
            aligned_count[:, None] @ chunk_merging_matrix,
            1,  # NB: cant merge global chunks :(
        )

        teacher_chunk_sums_after_merge = aligned_count
        teacher_avg_chunk_lengths_after_merge = (
            teacher_chunk_sums_after_merge.mean() / (teacher_chunk_sums_after_merge > 0).mean()
        )

        loss_args.scalar_report["teacher_avg_chunk_lengths_after_merge"] = teacher_avg_chunk_lengths_after_merge
        loss_args.scalar_report["t_min_aligned_space_logp"] = (
            t_aligned_space_logp * (aligned_count > 0)
        ).min()

    s_aligned_main_logp = s_aligned_main_logp * (aligned_count > 0)
    t_aligned_space_logp = t_aligned_space_logp * (aligned_count > 0)
    s_aligned_space_logp = s_aligned_space_logp * (aligned_count > 0)

    all_aligned_s_logps = []
    all_aligned_t_logps = []
    all_aligned_counts = []
    global_all_aligned_counts = []

    batch_size = loss_args.batch["input_ids_original"].shape[0]
    global_batch_size = loss_args.global_batch["input_ids_original"].shape[0]

    for size in args.distill_chunk_sizes:
        size_s_logp = s_aligned_main_logp.reshape(batch_size, -1, size).sum(-1)
        size_t_logp = t_aligned_main_logp.reshape(batch_size, -1, size).sum(-1)
        size_count = aligned_count.reshape(batch_size, -1, size)
        global_size_count = global_aligned_count.reshape(global_batch_size, -1, size)

        if "append_space" in args.alm_mode:
            last_position_in_chunk = (
                jnp.cumsum((size_count > 0)[..., ::-1], axis=-1) == 1
            )[..., ::-1]

            size_s_logp = size_s_logp + (
                s_aligned_space_logp.reshape(batch_size, -1, size)
                * last_position_in_chunk
            ).sum(-1)
            size_t_logp = size_t_logp + (
                t_aligned_space_logp.reshape(batch_size, -1, size)
                * last_position_in_chunk
            ).sum(-1)

        all_aligned_s_logps.append(size_s_logp)
        all_aligned_t_logps.append(size_t_logp)
        all_aligned_counts.append(size_count.sum(-1))
        global_all_aligned_counts.append(global_size_count.sum(-1))

    s_full_aligned_main_logp = jnp.concatenate(all_aligned_s_logps, -1)
    t_full_aligned_main_logp = jnp.concatenate(all_aligned_t_logps, -1)
    full_aligned_counts = jnp.concatenate(all_aligned_counts, -1)
    global_full_aligned_counts = jnp.concatenate(global_all_aligned_counts, -1)

    t_full_aligned_main_logp = jnp.where(
        full_aligned_counts > 0,
        t_full_aligned_main_logp,
        utils.get_large_negative_number(t_full_aligned_main_logp.dtype),
    )
    s_full_aligned_main_logp = jnp.where(
        full_aligned_counts > 0,
        s_full_aligned_main_logp,
        utils.get_large_negative_number(s_full_aligned_main_logp.dtype),
    )

    loss_args.scalar_report["legacy_loss"] = (
        jnp.abs(s_aligned_main_logp - t_aligned_main_logp) * (aligned_count > 0)
    ).mean() / (aligned_count > 0).mean()

    loss_args.scalar_report["t_min_p"] = (
        t_full_aligned_main_logp * (full_aligned_counts > 0)
    ).min()
    loss_args.scalar_report["t_mean_p"] = (
        t_full_aligned_main_logp * (full_aligned_counts > 0)
    ).mean() / (full_aligned_counts > 0).mean()
    loss_args.scalar_report["t_max_p"] = (
        jnp.where(
            full_aligned_counts > 0,
            t_full_aligned_main_logp,
            utils.get_large_negative_number(t_full_aligned_main_logp.dtype),
        )
    ).max()
    loss_args.scalar_report["s_min_p"] = (
        s_full_aligned_main_logp * (full_aligned_counts > 0)
    ).min()
    loss_args.scalar_report["s_mean_p"] = (
        s_full_aligned_main_logp * (full_aligned_counts > 0)
    ).mean() / (full_aligned_counts > 0).mean()
    loss_args.scalar_report["s_max_p"] = (
        jnp.where(
            full_aligned_counts > 0,
            s_full_aligned_main_logp,
            utils.get_large_negative_number(s_full_aligned_main_logp.dtype),
        )
    ).max()

    if args.distill_main_path_numerator == "token_count":
        numerator = full_aligned_counts
    elif args.distill_main_path_numerator == "chunk_count":
        numerator = full_aligned_counts > 0
    elif args.distill_main_path_numerator == "log1p_token_count":
        numerator = jnp.log1p(full_aligned_counts)

    if args.distill_main_path_denominator == "token_count":
        denominator = global_full_aligned_counts.mean()
    elif args.distill_main_path_denominator == "chunk_count":
        denominator = (global_full_aligned_counts > 0).mean()

    elementwise_loss = (
        diff_fn(t_full_aligned_main_logp, s_full_aligned_main_logp)
        * numerator
        / denominator
    )

    distill_main_path_loss = elementwise_loss.mean() / len(args.distill_chunk_sizes)

    return distill_main_path_loss


def compute_baseline_dskd_loss(args, loss_args, epsilon=1e-8):
    s_target_embeds = loss_args.student_out.hidden_states[0][:, 1:]
    t_target_embeds = loss_args.teacher_out.hidden_states[0][:, 1:]
    s_hiddens = loss_args.student_out.hidden_states[-1][:, :-1]
    t_hiddens = loss_args.teacher_out.hidden_states[-1][:, :-1]

    s_index_embeds = jnp.concatenate(
        [loss_args.student_out.hidden_states[0][:, :-1], s_target_embeds], -1
    )
    t_index_embeds = jnp.concatenate(
        [loss_args.teacher_out.hidden_states[0][:, :-1], t_target_embeds], -1
    )
    # std across batch and dimensions is weird but consistent with https://github.dev/songmzhang/DSKD
    norm_t_index_embeds = t_index_embeds / t_index_embeds.std()
    norm_s_index_embeds = s_index_embeds / s_index_embeds.std()
    norm_t_target_embeds = t_target_embeds / t_target_embeds.std()
    norm_t_hiddens = t_hiddens / t_hiddens.std()

    projector_query_params = loss_args.params["model"]["projector_query"]
    projector_s2t_params = loss_args.params["model"]["projector_s2t"]
    projector_t2s_params = loss_args.params["model"]["projector_t2s"]

    s_q_hiddens = (
        s_index_embeds @ projector_query_params["kernel"]
        + projector_query_params["bias"]
    ).astype(jnp.float32)
    t_k_hiddens = norm_t_index_embeds.astype(jnp.float32)

    s_v_hiddens = (
        s_hiddens @ projector_s2t_params["kernel"] + projector_s2t_params["bias"]
    ).astype(jnp.float32)
    t_v_hiddens = (
        (norm_t_hiddens + norm_t_target_embeds) @ projector_t2s_params["kernel"]
        + projector_t2s_params["bias"]
    ).astype(jnp.float32)

    align = s_q_hiddens @ jnp.swapaxes(t_k_hiddens, -1, -2)
    align = align / jnp.sqrt(2 * t_hiddens.shape[-1])

    t2s_align_mask = (
        loss_args.batch["attention_mask_new"][:, 1:, None]
        * loss_args.batch["attention_mask_original"][:, None, 1:]
    )
    s2t_align_mask = (
        loss_args.batch["attention_mask_original"][:, 1:, None]
        * loss_args.batch["attention_mask_new"][:, None, 1:]
    )

    t2s_weight = jax.nn.softmax(
        jnp.where(t2s_align_mask, align, utils.get_large_negative_number(align.dtype)),
        -1,
    )
    s2t_weight = jax.nn.softmax(
        jnp.where(
            s2t_align_mask,
            jnp.swapaxes(align, -1, -2),
            utils.get_large_negative_number(align.dtype),
        ),
        -1,
    )

    t2s_hiddens = (t2s_weight @ t_v_hiddens).astype(s_hiddens.dtype)
    t2s_logits = t2s_hiddens @ jax.lax.stop_gradient(
        loss_args.predicted_embeddings[:, -1, :].T
    )
    t2s_ce_loss = cross_entropy(
        t2s_logits,
        loss_args.batch["input_ids_new"],
        loss_args.batch["loss_mask_new"],
        logit_mask=loss_args.logit_mask_new,
        logits_already_shifted=True,
        denom=loss_args.global_batch["loss_mask_new"][:, 1:].mean(),
    )
    t2s_acc_mask = jnp.argmax(t2s_logits, -1) == loss_args.batch["input_ids_new"][:, 1:]
    t2s_acc = (
        t2s_acc_mask * loss_args.batch["loss_mask_new"][:, 1:]
    ).mean() / loss_args.batch["loss_mask_new"][:, 1:].mean()
    loss_args.scalar_report["t2s_acc"] = t2s_acc

    if args.baseline.divergence == "akl":
        t2s_div_func = partial(
            baseline_utils.compute_adaptive_kl_divergence,
            alpha=args.baseline.adaptive_kl_alpha,
        )
    elif args.baseline.divergence == "skl":
        t2s_div_func = partial(
            baseline_utils.compute_skewed_kl_divergence,
            skew_lambda=args.baseline.skew_lambda,
        )
    elif args.baseline.divergence == "srkl":
        t2s_div_func = partial(
            baseline_utils.compute_skewed_reverse_kl_divergence,
            skew_lambda=args.baseline.skew_lambda,
        )
    elif args.baseline.divergence == "kl":
        t2s_div_func = baseline_utils.compute_forward_kl_divergence
    elif args.baseline.divergence == "rkl":
        t2s_div_func = baseline_utils.compute_reverse_kl_divergence

    t2s_kd_loss = t2s_div_func(
        logits=loss_args.student_logits[:, :-1],
        teacher_logits=jax.lax.stop_gradient(t2s_logits),
        target=loss_args.batch["input_ids_new"][:, 1:],
        kd_temp=args.baseline.kd_temp,
        padding_id=loss_args.tokenizer_new.pad_token_id,
        tea_temp=args.baseline.teacher_temperature,
        reduction="none",
        use_tea_temp=True,
    )
    t2s_kd_loss = (
        t2s_kd_loss * loss_args.batch["loss_mask_new"][:, 1:] * t2s_acc_mask
    ).mean() / (
        # can not use global batch since we do not have a global t2s_acc_mask
        (loss_args.batch["loss_mask_new"][:, 1:] * t2s_acc_mask).mean()
        + epsilon
    )

    s2t_hiddens = (s2t_weight @ s_v_hiddens).astype(s_hiddens.dtype)
    s2t_logits = s2t_hiddens @ loss_args.params["teacher_embeddings"][:, -1, :].T
    s2t_kd_loss = baseline_utils.compute_forward_kl_divergence(
        logits=s2t_logits,
        teacher_logits=loss_args.teacher_logits[:, :-1],
        target=loss_args.batch["input_ids_original"][:, 1:],
        kd_temp=args.baseline.kd_temp,
        padding_id=loss_args.tokenizer_teacher.pad_token_id,
        reduction="none",
    )
    s2t_kd_loss = (
        s2t_kd_loss * loss_args.batch["loss_mask_original"][:, 1:]
    ).mean() / loss_args.global_batch["loss_mask_original"][:, 1:].mean()
    loss_args.scalar_report["t2s_ce_loss"] = t2s_ce_loss
    loss_args.scalar_report["t2s_kd_loss"] = t2s_kd_loss
    loss_args.scalar_report["s2t_kd_loss"] = s2t_kd_loss

    dskd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
    return dskd_loss


def compute_baseline_uld_loss(args, loss_args):
    sorted_student_probs = jnp.sort(loss_args.student_probs, descending=True, axis=-1)
    sorted_teacher_probs = jnp.sort(loss_args.teacher_probs, descending=True, axis=-1)

    vocab_size_gap = (
        loss_args.new_config.vocab_size - loss_args.teacher_config.vocab_size
    )
    if vocab_size_gap > 0:
        sorted_teacher_probs = jnp.pad(
            sorted_teacher_probs,
            ((0, 0), (0, 0), (0, vocab_size_gap)),
            constant_values=0.0,
        )
    elif vocab_size_gap < 0:
        sorted_student_probs = jnp.pad(
            sorted_student_probs,
            ((0, 0), (0, 0), (0, -vocab_size_gap)),
            constant_values=0.0,
        )

    uld_loss = jnp.abs(sorted_student_probs - sorted_teacher_probs).sum(-1)
    uld_loss = (
        uld_loss * loss_args.batch["attention_mask_new"]
    ).mean() / loss_args.batch["attention_mask_new"].mean()
    return uld_loss


def compute_baseline_mined_loss(mined_mapping, args, loss_args):
    alignment_matrix_a = (
        loss_args.batch["alignment_matrix_a_unconstrained"]
        .at[:, :-1]
        .set(
            loss_args.batch["alignment_matrix_a_unconstrained"][:, :-1]
            & loss_args.batch["loss_mask_new"][:, 1:, None]
        )
    )
    alignment_matrix_b = (
        loss_args.batch["alignment_matrix_b_unconstrained"]
        .at[:, :-1]
        .set(
            loss_args.batch["alignment_matrix_b_unconstrained"][:, :-1]
            & loss_args.batch["loss_mask_original"][:, 1:, None]
        )
    )
    global_alignment_matrix_a = (
        loss_args.global_batch["alignment_matrix_a_unconstrained"]
        .at[:, :-1]
        .set(
            loss_args.global_batch["alignment_matrix_a_unconstrained"][:, :-1]
            & loss_args.global_batch["loss_mask_new"][:, 1:, None]
        )
    )
    global_alignment_matrix_b = (
        loss_args.global_batch["alignment_matrix_b_unconstrained"]
        .at[:, :-1]
        .set(
            loss_args.global_batch["alignment_matrix_b_unconstrained"][:, :-1]
            & loss_args.global_batch["loss_mask_original"][:, 1:, None]
        )
    )

    alignment_matrix_b_last_only_index, _ = get_last_index_per_column(
        alignment_matrix_b
    )
    alignment_matrix_a_last_only_index, _ = get_last_index_per_column(
        alignment_matrix_a
    )

    one_to_one_mask = (alignment_matrix_b[:, 1:].sum(-2) == 1) & (
        alignment_matrix_a[:, 1:].sum(-2) == 1
    )
    global_one_to_one_mask = global_alignment_matrix_b[:, 1:].sum(-2) == 1 & (
        global_alignment_matrix_a[:, 1:].sum(-2) == 1
    )
    # the two sources of signal: mined-transformed teacher logits and one-hot logits
    mined_teacher_logits = (
        loss_args.teacher_logits[
            ...,
            jnp.pad(
                mined_mapping, (0, loss_args.new_config.vocab_size - len(mined_mapping))
            ),
        ]
        + loss_args.logit_mask_new[None, None]
    )
    onehot_probs = jax.nn.one_hot(
        loss_args.batch["input_ids_new"][:, 1:], loss_args.new_config.vocab_size
    )
    # use arbitrary high / low numbers as the logits of the one hot encoding
    # specific numbers from https://github.com/songmzhang/DSKD/blob/main/code/criterions/min_edit_dis_kld.py
    onehot_logits = (
        onehot_probs * 100
        + (1 - onehot_probs) * -100_000
        + loss_args.logit_mask_new[None, None]
    )

    mined_teacher_logits = jax.lax.with_sharding_constraint(
        mined_teacher_logits,
        NamedSharding(loss_args.new_config.mesh, P("data", None, "model")),
    )
    onehot_logits = jax.lax.with_sharding_constraint(
        onehot_logits,
        NamedSharding(loss_args.new_config.mesh, P("data", None, "model")),
    )

    aligned_teacher_logits = jnp.take_along_axis(
        mined_teacher_logits,
        alignment_matrix_b_last_only_index[..., None],
        axis=1,
    )
    aligned_student_kl_logits = jnp.take_along_axis(
        loss_args.student_logits,
        alignment_matrix_a_last_only_index[..., None],
        axis=1,
    )

    not_one_to_one_alignments = alignment_matrix_a[:, 1:] * ~one_to_one_mask[:, None, :]
    global_not_one_to_one_alignments = (
        global_alignment_matrix_a[:, 1:] * ~global_one_to_one_mask[:, None, :]
    )
    # TODO: check loss mask here / correctness of max(-1)
    onehot_index = (
        not_one_to_one_alignments
        * jnp.arange(alignment_matrix_a.shape[1] - 1)[None, :, None]
    ).max(-1)
    onehot_mask = not_one_to_one_alignments.sum(-1) != 0
    global_onehot_mask = global_not_one_to_one_alignments.sum(-1) != 0

    aligned_onehot_logits = jnp.take_along_axis(
        onehot_logits,
        onehot_index[..., None],
        axis=1,
    )
    aligned_student_onehot_logits = jnp.take_along_axis(
        loss_args.student_logits,
        onehot_index[..., None],
        axis=1,
    )

    elementwise_mined_teacher_kl_loss = baseline_utils.compute_forward_kl_divergence(
        aligned_student_kl_logits,
        aligned_teacher_logits,
        target=None,
        kd_temp=args.baseline.kd_temp,
        padding_id=None,
        reduction="none",
    )
    elementwise_onehot_kl_loss = baseline_utils.compute_forward_kl_divergence(
        aligned_student_onehot_logits,
        aligned_onehot_logits,
        target=None,
        kd_temp=args.baseline.kd_temp,
        padding_id=None,
        reduction="none",
    )

    loss_args.scalar_report["mined_one_to_one_mask_sum"] = one_to_one_mask.sum()
    loss_args.scalar_report["mined_onehot_mask_sum"] = onehot_mask.sum()

    mined_kl_loss = (
        (elementwise_mined_teacher_kl_loss * one_to_one_mask).sum()
        + (elementwise_onehot_kl_loss * onehot_mask).sum()
    ) / (global_one_to_one_mask.sum() + global_onehot_mask.sum()).astype(jnp.float32)

    return mined_kl_loss
