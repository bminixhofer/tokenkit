"""
LM evaluation harness compatible evaluation interface for JAX models.

IMPORTANT NOTES:
- Log-likelihood scoring sorts examples by length and processes them with a decreasing schedule of maximum lengths (e.g. 2048, 1024, 512, ...).
    - This requires a recompile at every length change, causing progress to freeze for a few seconds.
    - This wastes quite a bit of tokens on padding, it would be better to pack, but this would be more complex to implement.
- Generation currently does not reshard. It uses FSDP which is fairly catastrophic for generation performance, so this part is really only suited to few examples.
    - We need to figure out how to reshard/distribute generation across accelerators.
"""

import copy
import logging
import math
from functools import partial
from pathlib import Path
from pprint import pprint

import jax
import jax.numpy as jnp
import lm_eval
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from lm_eval.loggers import EvaluationTracker
from tqdm.auto import tqdm

from tokenkit import utils
from tokenkit.eval import generate
from tokenkit.models import param

logger = logging.getLogger(__name__)

ATOL = 1e-2


@partial(
    jax.jit,
    static_argnames=("model_fn", "atol"),
    out_shardings=(None, None),
)
def score(
    model_fn,
    params,
    model_args,
    labels,
    suffix_mask,
    space_mask,
    logit_mask=None,
    atol=ATOL,
):
    logits = model_fn(*model_args, params=params).logits.astype(jnp.float32)
    if logit_mask is not None:
        logit_bias = jnp.full(
            len(logit_mask),
            fill_value=utils.get_large_negative_number(logits.dtype),
            dtype=logits.dtype,
        )
        logit_bias = logit_bias * ~logit_mask
        logits = logits + logit_bias[None, None, :]

    logprobs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(logprobs)

    shift_logprobs = logprobs[:, :-1]
    shift_labels = labels[:, 1:]
    shift_suffix_mask = suffix_mask[:, 1:]

    sequence_logprobs = jnp.take_along_axis(
        shift_logprobs, shift_labels[:, :, None], axis=-1
    ).squeeze(-1)
    max_logprobs = jnp.max(shift_logprobs, axis=-1)

    sequence_logprobs = (sequence_logprobs * shift_suffix_mask).sum(axis=-1)
    max_logprobs = (max_logprobs * shift_suffix_mask).sum(axis=-1)

    is_greedy = jnp.isclose(sequence_logprobs, max_logprobs, rtol=0.0, atol=atol)

    return sequence_logprobs, is_greedy


class JaxLM(lm_eval.api.model.LM):
    def __init__(
        self,
        model,
        config,
        params,
        tokenizer,
        lengths,
        tokens_per_batch,
        add_bos,
        chat_template_mode,
        logit_mask=None,
        score_fn=score,
        precompile=True,
        expand_input_ids=False,
        expand_input_ids_vocab=None,
    ):
        self.model = model
        self.model_fn = model.__call__
        self.config = config
        self.params = params
        self.tokenizer = tokenizer
        self.lengths = lengths
        self.tokens_per_batch = tokens_per_batch
        self.logit_mask = logit_mask
        self.score_fn = score_fn
        self.add_bos = add_bos
        self.chat_template_mode = chat_template_mode
        self.precompile = precompile

        self.expand_input_ids = expand_input_ids
        self.expand_input_ids_vocab = expand_input_ids_vocab

        for length in list(lengths):
            if length > self.max_length:
                logger.warning(
                    "Ignoring length %d as it exceeds maximum sequence length %d",
                    length,
                    self.max_length,
                )
                lengths.remove(length)

        for length in self.lengths:
            assert self.tokens_per_batch % length == 0

        self.max_batch_size = self.tokens_per_batch // self.lengths[0]

        super().__init__()

    # see https://github.com/EleutherAI/lm-evaluation-harness/blob/3fa4fd725c8a428710109f1d6c14eda37e95baea/lm_eval/models/huggingface.py#L368C1-L380C40
    @property
    def max_length(self):
        seqlen_config_attrs = (
            "n_positions",
            "max_position_embeddings",
            "n_ctx",
        )
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self.lengths[-1]
            return self.tokenizer.model_max_length
        return self.lengths[-1]

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        prefixes = [x.args[0] for x in requests]
        suffixes = [x.args[1] for x in requests]

        prefix_tokens = self.tokenizer(prefixes, add_special_tokens=False)["input_ids"]
        suffix_tokens = self.tokenizer(suffixes, add_special_tokens=False)["input_ids"]

        total_lengths = np.array(
            [
                len(prefix) + len(suffix)
                for prefix, suffix in zip(prefix_tokens, suffix_tokens)
            ]
        )

        permutation = np.argsort(total_lengths)[::-1]

        n_batches = math.ceil(len(prefix_tokens) / self.max_batch_size)

        input_ids = np.full(
            (self.max_batch_size, self.lengths[-1]),
            fill_value=self.tokenizer.pad_token_id,
            dtype=np.int32,
        )
        suffix_mask = np.zeros((self.max_batch_size, self.lengths[-1]), dtype=bool)

        output = [None for _ in range(len(prefix_tokens))]

        for batch_idx in tqdm(
            range(n_batches), desc="Running loglikelihood requests..."
        ):
            start, end = (
                batch_idx * self.max_batch_size,
                min((batch_idx + 1) * self.max_batch_size, len(prefix_tokens)),
            )

            batch_max_length = 0
            for idx, i in enumerate(permutation[start:end]):
                prefix = prefix_tokens[i]
                suffix = suffix_tokens[i]

                # best-effort truncation from the left
                while len(prefix) + len(suffix) > self.lengths[-1] - self.add_bos:
                    del prefix[0]

                if len(prefix) == 0:
                    raise ValueError(
                        f"Prefix is empty after truncation to length {self.lengths[-1]}"
                    )

                batch_max_length = max(
                    batch_max_length, len(prefix) + len(suffix) + self.add_bos
                )

                input_ids[idx] = self.tokenizer.pad_token_id
                suffix_mask[idx] = False

                offset = 0

                if self.add_bos:
                    assert self.tokenizer.bos_token_id is not None
                    input_ids[idx, 0] = self.tokenizer.bos_token_id
                    offset = 1

                input_ids[idx, offset : offset + len(prefix)] = prefix
                offset = offset + len(prefix)
                input_ids[idx, offset : offset + len(suffix)] = suffix
                offset = offset + len(suffix)

                suffix_mask[idx, offset - len(suffix) : offset] = True

            length_index = 0
            while self.lengths[length_index] < batch_max_length:
                length_index += 1

            ll = []
            is_greedy = []

            length = self.lengths[length_index]
            batch_size = self.tokens_per_batch // length
            for i in range(0, self.max_batch_size, batch_size):
                prev_length = self.config.max_length
                self.config.max_length = length
                ll_batch, is_greedy_batch = self.score_fn(
                    self.model_fn,
                    self.params,
                    (input_ids[i : i + batch_size, :length],),
                    input_ids[i : i + batch_size, :length],
                    suffix_mask[i : i + batch_size, :length],
                    None,
                    self.logit_mask,
                    ATOL,
                )
                self.config.max_length = prev_length
                ll_batch = process_allgather(ll_batch, tiled=True)
                is_greedy_batch = process_allgather(is_greedy_batch, tiled=True)

                ll.extend(ll_batch.tolist())
                is_greedy.extend(is_greedy_batch.tolist())

            for idx, i in enumerate(permutation[start:end]):
                output[i] = (
                    ll[idx],
                    is_greedy[idx],
                )

        return output

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def generate_until(self, ungrouped_requests):
        request_groups = {}
        generate_kwargs_groups = {}

        ungrouped_output_texts = [None for _ in range(len(ungrouped_requests))]

        for i, request in enumerate(ungrouped_requests):
            key = utils.make_hashable(request.args[1])
            if key not in request_groups:
                request_groups[key] = []
                generate_kwargs_groups[key] = copy.deepcopy(request.args[1])

            assert generate_kwargs_groups[key] == request.args[1]
            request_groups[key].append((i, request))

        for key, requests in request_groups.items():
            generate_kwargs = generate_kwargs_groups[key]
            print("Generating with kwargs:")
            pprint(generate_kwargs)
            print(f"Running {len(requests)} generate requests...")

            prompts = [
                utils.preprocess_prompt(x.args[0], self.chat_template_mode)
                for _, x in requests
            ]

            until = generate_kwargs.pop("until")

            max_new_tokens = generate_kwargs.pop("max_gen_toks", self.lengths[-1])
            sampling_config = generate.SamplingConfig(**generate_kwargs)

            generator = generate.Generator(
                mesh=self.config.mesh,
                model=self.model,
                params=self.params,
                tokenizer=self.tokenizer,
                sampling_config=sampling_config,
                logit_mask=self.logit_mask,
                until=until,
                # TODO: heuristic for batch size and lengths, clarify / improve
                batch_size=self.tokens_per_batch // self.lengths[-1],
                lengths=[self.lengths[-1]],
                max_new_tokens=max_new_tokens,
                precompile=self.precompile,
                expand_input_ids=self.expand_input_ids,
                expand_input_ids_vocab=self.expand_input_ids_vocab,
            )
            output_tokens = generator.generate(prompts)
            for request_index, example_output_tokens in zip(
                [i for i, _ in requests], output_tokens
            ):
                ungrouped_output_texts[request_index] = self.tokenizer.decode(
                    example_output_tokens
                ).strip()

            # original set deleted by Generator (this is not ideal :( )
            self.params = generator.params

        return ungrouped_output_texts


@partial(
    jax.jit,
    static_argnames=("model_fns", "combine_fn", "atol"),
    out_shardings=(None, None),
)
def score_lockstep(
    model_fns,
    combine_fn,
    params,
    combine_params,
    batch,
    atol=ATOL,
):
    all_unified_aligned_logits = []
    all_aligned_last_hidden_states = []
    all_output_embeddings = []

    for i in range(len(model_fns)):
        model_out = model_fns[i](
            input_ids=batch["input_ids"][:, i],
            attention_mask=batch["attention_mask"][:, i],
            params=params[i],
            dropout_rng=None,
            output_hidden_states=True,
            train=False,
        )
        unified_logits = model_out.logits[..., batch["unified_indices_" + str(i)]]
        unified_aligned_logits = jnp.take_along_axis(
            unified_logits,
            batch["inv_regular_token_ids"][:, i][:, :, None] - 1,
            axis=1,
        )
        aligned_last_hidden_states = jnp.take_along_axis(
            model_out.hidden_states[-1],
            batch["inv_regular_token_ids"][:, i][:, :, None] - 1,
            axis=1,
        )

        all_unified_aligned_logits.append(unified_aligned_logits)
        all_aligned_last_hidden_states.append(aligned_last_hidden_states)

        if model_fns[i].config.tie_word_embeddings:
            all_output_embeddings.append(
                param.get(
                    params[i],
                    param.get_input_embedding_path(model_fns[i].config.model_type),
                )[batch["unified_indices_" + str(i)]]
            )
        else:
            all_output_embeddings.append(
                param.get(
                    params["models"][str(i)],
                    param.get_output_embedding_path(model_fns[i].config.model_type),
                ).T[batch["unified_indices_" + str(i)]]
            )

    logits = combine_fn(
        all_aligned_last_hidden_states,
        all_unified_aligned_logits,
        combine_params,
        all_output_embeddings,
    )
    logits = jnp.where(
        batch["unified_indices_mask"][None, None, :],
        logits,
        utils.get_large_negative_number(logits.dtype),
    )
    logprobs = jax.nn.log_softmax(logits, axis=-1)

    unified_aligned_labels = jnp.take_along_axis(
        batch["unified_input_ids"],
        # collator ensures this is in bounds
        batch["inv_regular_token_ids"][:, 0],
        axis=1,
    )

    sequence_logprobs = jnp.take_along_axis(
        logprobs, unified_aligned_labels[:, :, None], axis=-1
    ).squeeze(-1)
    max_logprobs = jnp.max(logprobs, axis=-1)

    sequence_logprobs = (sequence_logprobs * batch["inv_regular_token_ids_mask"]).sum(
        axis=-1
    )
    max_logprobs = (max_logprobs * batch["inv_regular_token_ids_mask"]).sum(axis=-1)

    is_greedy = jnp.isclose(sequence_logprobs, max_logprobs, rtol=0.0, atol=atol)

    return sequence_logprobs, is_greedy


class LockstepJaxLM(lm_eval.api.model.LM):
    def __init__(
        self,
        models,
        configs,
        params,
        tokenizers,
        lengths,
        tokens_per_batch,
        add_bos,
        chat_template_mode,
        combine_fn,
        combine_params,
        logit_masks=None,
        score_fn=score_lockstep,
        precompile=True,
    ):
        self.models = models
        self.configs = configs
        self.params = params
        self.tokenizers = tokenizers
        self.lengths = lengths
        self.tokens_per_batch = tokens_per_batch
        self.logit_masks = logit_masks
        self.score_fn = score_fn
        self.add_bos = add_bos
        self.chat_template_mode = chat_template_mode
        self.combine_fn = combine_fn
        self.combine_params = combine_params
        self.precompile = precompile

        for length in list(lengths):
            if length > self.max_length:
                logger.warning(
                    "Ignoring length %d as it exceeds maximum sequence length %d",
                    length,
                    self.max_length,
                )
                lengths.remove(length)

        for length in self.lengths:
            assert self.tokens_per_batch % length == 0

        self.max_batch_size = self.tokens_per_batch // self.lengths[0]

        if not hasattr(configs[0], "mined_mapping"):
            shared_vocab_size = min(config.vocab_size for config in configs)

            regular_token_intersection = set.intersection(
                *[set(x.get_vocab().keys()) for x in tokenizers]
            )
            for tokenizer in tokenizers:
                regular_token_intersection -= set(tokenizer.all_special_tokens)
            regular_token_intersection = sorted(regular_token_intersection)

            self.unified_tokens = [
                regular_token_intersection
                + [tokenizer.model_kind_cls.replacements["<|<eot>|>"][0]]
                for tokenizer in self.tokenizers
            ]
            

            self.unified_indices = [
                np.pad(
                    np.array(
                        tokenizer.convert_tokens_to_ids(unified_tokens), dtype=np.int32
                    ),
                    (0, shared_vocab_size - len(unified_tokens)),
                    mode="constant",
                    constant_values=0,
                )
                for tokenizer, unified_tokens in zip(self.tokenizers, self.unified_tokens)
            ]

            self.unified_indices_mask = np.concatenate(
                [
                    np.ones(len(self.unified_tokens[0]), dtype=bool),
                    np.zeros(shared_vocab_size - len(self.unified_tokens[0]), dtype=bool),
                ]
            )
        else:
            shared_vocab_size = configs[0].vocab_size
            pivot_tokenizer = tokenizers[0]

            regular_pivot_tokens = sorted(set(pivot_tokenizer.get_vocab().keys()) - set(pivot_tokenizer.model_kind_cls.special_tokens))
            regular_pivot_indices = pivot_tokenizer.convert_tokens_to_ids(regular_pivot_tokens)
            self.unified_indices = [
                np.pad(
                    regular_pivot_indices + [pivot_tokenizer.convert_tokens_to_ids(pivot_tokenizer.model_kind_cls.replacements["<|<eot>|>"][0])],
                    (0, shared_vocab_size - len(regular_pivot_indices) - 1),
                    mode="constant",
                    constant_values=0,
                )
            ]

            for extra_idx, extra_tokenizer in enumerate(tokenizers[1:]):
                extra_indices = [configs[extra_idx + 1].mined_mapping[i] for i in regular_pivot_indices]
                self.unified_indices.append(np.pad(
                    extra_indices + [extra_tokenizer.convert_tokens_to_ids(extra_tokenizer.model_kind_cls.replacements["<|<eot>|>"][0])],
                    (0, shared_vocab_size - len(extra_indices) - 1),
                    mode="constant",
                    constant_values=0,
                ))

            self.unified_indices_mask = np.concatenate(
                [
                    np.ones(len(regular_pivot_tokens) + 1, dtype=bool),
                    np.zeros(shared_vocab_size - len(regular_pivot_tokens) - 1, dtype=bool),
                ]
            )

        self.inv_unified_indices = []
        for i in range(len(self.unified_indices)):
            current_inv_unified_indices = np.full(
                (len(self.tokenizers[i]),), fill_value=-1, dtype=np.int32
            )
            current_inv_unified_indices[self.unified_indices[i]] = np.arange(
                len(self.unified_indices[i])
            )
            self.inv_unified_indices.append(current_inv_unified_indices)

        super().__init__()

    def _model_max_length(self, config, tokenizer):
        seqlen_config_attrs = (
            "n_positions",
            "max_position_embeddings",
            "n_ctx",
        )
        for attr in seqlen_config_attrs:
            if hasattr(config, attr):
                return getattr(config, attr)
        if hasattr(tokenizer, "model_max_length"):
            if tokenizer.model_max_length == 1000000000000000019884624838656:
                return self.lengths[-1]
            return tokenizer.model_max_length
        return self.lengths[-1]

    @property
    def max_length(self):
        return min(
            self._model_max_length(config, tokenizer)
            for config, tokenizer in zip(self.configs, self.tokenizers)
        )

    def _encode_batch(
        self,
        all_prefix_token_ids,
        all_suffix_token_ids,
        all_regular_token_ids,
        max_length,
    ):
        def _encode_for_model(model_idx):
            tokenizer = self.tokenizers[model_idx]

            input_ids = np.full(
                (len(all_prefix_token_ids), max_length),
                fill_value=tokenizer.pad_token_id,
                dtype=np.int32,
            )
            attention_mask = np.zeros(
                (len(all_prefix_token_ids), max_length), dtype=np.int32
            )
            regular_token_ids = np.full(
                (len(all_prefix_token_ids), max_length), fill_value=-1, dtype=np.int32
            )

            for i in range(len(all_prefix_token_ids)):
                current_prefix_token_ids = all_prefix_token_ids[i][model_idx]
                current_suffix_token_ids = all_suffix_token_ids[i][model_idx]
                current_regular_token_ids = all_regular_token_ids[i][model_idx]

                input_ids[i, : len(current_prefix_token_ids)] = current_prefix_token_ids
                input_ids[
                    i,
                    len(current_prefix_token_ids) : len(current_prefix_token_ids)
                    + len(current_suffix_token_ids),
                ] = current_suffix_token_ids
                regular_token_ids[i, : len(current_regular_token_ids)] = (
                    current_regular_token_ids
                )
                attention_mask[
                    i, : len(current_prefix_token_ids) + len(current_suffix_token_ids)
                ] = 1

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "regular_token_ids": regular_token_ids,
            }

        encodings = []
        for model_idx in range(len(self.models)):
            encoding = _encode_for_model(model_idx)
            encodings.append(encoding)

        stacked_encodings = {
            key: np.stack([encoding[key] for encoding in encodings], axis=1)
            for key in encodings[0].keys()
        }

        stacked_encodings["regular_token_ids"] = np.where(
            stacked_encodings["regular_token_ids"]
            <= (stacked_encodings["regular_token_ids"].max(-1, keepdims=True)).min(
                1, keepdims=True
            ),
            stacked_encodings["regular_token_ids"],
            -1,
        )

        inv_regular_token_ids = np.full(
            stacked_encodings["regular_token_ids"].shape, fill_value=0, dtype=np.int32
        )
        inv_regular_token_ids_mask = np.zeros_like(inv_regular_token_ids).astype(bool)

        for model_idx in range(len(self.models)):
            for example_idx in range(len(all_prefix_token_ids)):
                current_inv_ids = np.where(
                    stacked_encodings["regular_token_ids"][example_idx, model_idx] != -1
                )[0]
                inv_regular_token_ids[
                    example_idx, model_idx, : len(current_inv_ids)
                ] = current_inv_ids
                inv_regular_token_ids_mask[
                    example_idx, model_idx, : len(current_inv_ids)
                ] = True

        for model_idx in range(len(self.models)):
            assert (
                inv_regular_token_ids_mask[:, 0]
                == inv_regular_token_ids_mask[:, model_idx]
            ).all()

        inv_regular_token_ids_mask = inv_regular_token_ids_mask[:, 0]

        batch = {
            "input_ids": stacked_encodings["input_ids"],
            "unified_input_ids": self.inv_unified_indices[0][
                stacked_encodings["input_ids"][:, 0]
            ],
            "attention_mask": stacked_encodings["attention_mask"],
            "regular_token_ids": stacked_encodings["regular_token_ids"],
            "inv_regular_token_ids": inv_regular_token_ids,
            "inv_regular_token_ids_mask": inv_regular_token_ids_mask,
            "unified_indices_mask": self.unified_indices_mask,
        }

        for i in range(len(self.unified_indices)):
            batch["unified_indices_" + str(i)] = self.unified_indices[i]

        return batch

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        prefixes = [x.args[0] for x in requests]
        suffixes = [x.args[1] for x in requests]

        all_prefix_token_ids = []
        all_suffix_token_ids = []
        all_regular_token_ids = []
        all_max_total_lengths = []

        for i, (prefix, suffix) in tqdm(
            enumerate(zip(prefixes, suffixes)),
            total=len(prefixes),
            desc="Encoding prompts...",
        ):
            current_prefix_token_ids = []
            current_suffix_token_ids = []
            current_regular_token_ids = []

            for model_idx in range(len(self.models)):
                prefix_token_ids, prefix_regular_token_ids = utils.encode_prompt(
                    utils.preprocess_prompt(prefix, "direct_encode_no_force_eos"),
                    self.tokenizers[model_idx],
                    max_length=self.lengths[-1],
                )
                suffix_token_ids, suffix_regular_token_ids = utils.encode_prompt(
                    utils.preprocess_prompt(
                        suffix, "direct_encode_no_force_bos_no_force_eos"
                    ),
                    self.tokenizers[model_idx],
                    max_length=self.lengths[-1],
                )

                # best-effort truncation from the left
                while len(prefix_token_ids) + len(suffix_token_ids) > self.lengths[-1]:
                    del prefix_token_ids[0]

                if len(prefix_token_ids) == 0:
                    raise ValueError(
                        f"Prefix is empty after truncation to length {self.lengths[-1]}"
                    )

                current_prefix_token_ids.append(prefix_token_ids)
                current_suffix_token_ids.append(suffix_token_ids)
                current_regular_token_ids.append(
                    [-1] * len(prefix_regular_token_ids) + suffix_regular_token_ids
                )

            all_prefix_token_ids.append(current_prefix_token_ids)
            all_suffix_token_ids.append(current_suffix_token_ids)
            all_regular_token_ids.append(current_regular_token_ids)
            all_max_total_lengths.append(
                max(
                    len(prefix) + len(suffix)
                    for prefix, suffix in zip(
                        current_prefix_token_ids, current_suffix_token_ids
                    )
                )
            )

        permutation = np.argsort(all_max_total_lengths)[::-1]
        n_batches = math.ceil(len(all_prefix_token_ids) / self.max_batch_size)

        output = [None for _ in range(len(all_prefix_token_ids))]

        for batch_idx in tqdm(
            range(n_batches), desc="Running loglikelihood requests..."
        ):
            start, end = (
                batch_idx * self.max_batch_size,
                min((batch_idx + 1) * self.max_batch_size, len(all_prefix_token_ids)),
            )

            batch_max_length = max(
                all_max_total_lengths[i] for i in permutation[start:end]
            )
            assert batch_max_length <= self.lengths[-1]

            length_index = 0
            while self.lengths[length_index] < batch_max_length:
                length_index += 1

            ll = []
            is_greedy = []

            length = self.lengths[length_index]
            batch_size = self.tokens_per_batch // length
            for i in range(0, self.max_batch_size, batch_size):
                batch_indices = permutation[start:end][i : i + batch_size]
                batch = self._encode_batch(
                    [all_prefix_token_ids[j] for j in batch_indices],
                    [all_suffix_token_ids[j] for j in batch_indices],
                    [all_regular_token_ids[j] for j in batch_indices],
                    length,
                )

                prev_max_lengths = []
                for config in self.configs:
                    prev_max_lengths.append(config.max_length)
                    config.max_length = self.lengths[-1]

                ll_batch, is_greedy_batch = self.score_fn(
                    tuple(self.models),
                    self.combine_fn,
                    self.params,
                    self.combine_params,
                    batch,
                    ATOL,
                )

                for config, prev_max_length in zip(self.configs, prev_max_lengths):
                    config.max_length = prev_max_length

                ll_batch = process_allgather(ll_batch, tiled=True)
                is_greedy_batch = process_allgather(is_greedy_batch, tiled=True)

                ll.extend(ll_batch.tolist())
                is_greedy.extend(is_greedy_batch.tolist())

            for idx, i in enumerate(permutation[start:end]):
                output[i] = (
                    ll[idx],
                    is_greedy[idx],
                )

        return output

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def generate_until(self, ungrouped_requests):
        request_groups = {}
        generate_kwargs_groups = {}

        ungrouped_output_texts = [None for _ in range(len(ungrouped_requests))]

        for i, request in enumerate(ungrouped_requests):
            key = utils.make_hashable(request.args[1])
            if key not in request_groups:
                request_groups[key] = []
                generate_kwargs_groups[key] = copy.deepcopy(request.args[1])

            assert generate_kwargs_groups[key] == request.args[1]
            request_groups[key].append((i, request))

        for key, requests in request_groups.items():
            generate_kwargs = generate_kwargs_groups[key]
            print("Generating with kwargs:")
            pprint(generate_kwargs)
            print(f"Running {len(requests)} generate requests...")

            prompts = [
                utils.preprocess_prompt(x.args[0], self.chat_template_mode)
                for _, x in requests
            ]

            until = generate_kwargs.pop("until")

            max_new_tokens = generate_kwargs.pop("max_gen_toks", self.lengths[-1])
            sampling_config = generate.SamplingConfig(**generate_kwargs)

            prev_max_lengths = []
            for config in self.configs:
                prev_max_lengths.append(config.max_length)
                config.max_length = self.lengths[-1]

            lockstep_generator = generate.LockstepGenerator(
                mesh=self.configs[0].mesh,
                all_models=self.models,
                all_configs=self.configs,
                all_params=self.params,
                all_tokenizers=self.tokenizers,
                sampling_config=sampling_config,
                max_new_tokens=max_new_tokens,
                combine_fn=self.combine_fn,
                combine_params=self.combine_params,
                dtype=None,
                all_logit_masks=self.logit_masks,
                precompile=self.precompile,
                batch_size=self.tokens_per_batch // self.lengths[-1],
                until=until,
                lengths=[self.lengths[-1]],
            )

            output_tokens = lockstep_generator.generate(prompts)

            for i, config in enumerate(self.configs):
                config.max_length = prev_max_lengths[i]

            for request_index, example_output_tokens in zip(
                [i for i, _ in requests], output_tokens
            ):
                ungrouped_output_texts[request_index] = (
                    self.tokenizers[-1].decode(example_output_tokens).strip()
                )

            # original set deleted by LockstepGenerator (this is not ideal :( )
            self.params = [
                generator.params for generator in lockstep_generator.generators
            ]
            self.combine_params = lockstep_generator.combine_params

        return ungrouped_output_texts


def evaluate(
    model,
    config,
    params,
    tokenizer,
    tasks,
    lengths,
    tokens_per_batch,
    logit_mask=None,
    output=None,
    add_bos=True,
    chat_template_mode="surround_instruct",
    cache_requests=True,
    jaxlm_kwargs=None,
    **kwargs,
):
    if output is not None:
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None

    lm_eval_model = JaxLM(
        model=model,
        config=config,
        params=params,
        tokenizer=tokenizer,
        lengths=lengths,
        tokens_per_batch=tokens_per_batch,
        add_bos=add_bos,
        chat_template_mode=chat_template_mode,
        logit_mask=logit_mask,
        **(jaxlm_kwargs or {}),
    )

    if output_dir is not None:
        evaluation_tracker = EvaluationTracker(output_path=output_dir)
    else:
        evaluation_tracker = None

    results = lm_eval.simple_evaluate(
        lm_eval_model,
        model_args="",
        tasks=tasks,
        evaluation_tracker=evaluation_tracker,
        cache_requests=cache_requests,
        **kwargs,
    )
    if evaluation_tracker is not None:
        # this is usually the `JaxLM` class which causes a memory leak
        evaluation_tracker.general_config_tracker.model_source = "jaxlm"

    if output_dir is not None:
        # compare __main__ in lm_eval
        evaluation_tracker.save_results_aggregated(
            results=results, samples=results["samples"]
        )
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=results["samples"][task_name]
            )

    return results["results"], lm_eval_model.params


def evaluate_lockstep(
    models,
    configs,
    params,
    tokenizers,
    tasks,
    lengths,
    tokens_per_batch,
    add_bos,
    combine_fn,
    combine_params,
    logit_masks=None,
    output=None,
    chat_template_mode="surround_instruct",
    cache_requests=True,
    jaxlm_kwargs=None,
    **kwargs,
):
    if output is not None:
        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None

    lm_eval_model = LockstepJaxLM(
        models=models,
        configs=configs,
        params=params,
        tokenizers=tokenizers,
        lengths=lengths,
        tokens_per_batch=tokens_per_batch,
        add_bos=add_bos,
        chat_template_mode=chat_template_mode,
        combine_fn=combine_fn,
        combine_params=combine_params,
        logit_masks=logit_masks,
        **(jaxlm_kwargs or {}),
    )

    if output_dir is not None:
        evaluation_tracker = EvaluationTracker(output_path=output_dir)
    else:
        evaluation_tracker = None

    results = lm_eval.simple_evaluate(
        lm_eval_model,
        model_args="",
        tasks=tasks,
        evaluation_tracker=evaluation_tracker,
        cache_requests=cache_requests,
        **kwargs,
    )
    if evaluation_tracker is not None:
        # this is usually the `JaxLM` class which causes a memory leak
        evaluation_tracker.general_config_tracker.model_source = "jaxlm"

    if output_dir is not None:
        # compare __main__ in lm_eval
        evaluation_tracker.save_results_aggregated(
            results=results, samples=results["samples"]
        )
        for task_name, config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=results["samples"][task_name]
            )

    return results["results"], (lm_eval_model.params, lm_eval_model.combine_params)
