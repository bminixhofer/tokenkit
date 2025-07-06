import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from pprint import pformat
import copy

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import regex as re
from flax import traverse_util
from scipy import sparse
from tqdm.auto import tqdm as raw_tqdm
from transformers import AutoTokenizer

import wandb
from tokenkit import constants, model_kinds
from tokenkit.byteify import ByteifyTokenizer, load_byteify_tokenizer

tqdm = partial(raw_tqdm, dynamic_ncols=True, disable=jax.process_index() != 0)

logger = logging.getLogger(__name__)


def log(data, step, **kwargs):
    # wandb does not support nested panels, so replace all except the last / with a _
    def replace_slashes(key):
        if "/" in key:
            parts = key.split("/")
            return "_".join(parts[:-1]) + "/" + parts[-1]
        return key

    data = {replace_slashes(k): v for k, v in data.items()}
    logger.info(pformat({**data, "_step": step}))
    if jax.process_index() == 0:
        wandb.log(data, step=step, **kwargs)


def keystr(x):
    if hasattr(x, "name"):
        return x.name
    elif hasattr(x, "key"):
        return x.key
    elif hasattr(x, "idx"):
        return x.idx

    assert isinstance(x, str)
    return x


# from praxis
def get_large_negative_number(dtype: jnp.dtype | np.dtype, module=jnp) -> jax.Array:
    """Returns a large negative value for the given dtype."""
    # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
    if module.issubdtype(dtype, module.inexact):
        dtype_max = module.finfo(dtype).max
    elif module.issubdtype(dtype, module.integer):
        dtype_max = module.iinfo(dtype).max
    else:
        raise ValueError("Unsupported dtype for inputs.")

    return module.asarray(-0.7 * dtype_max, dtype=dtype)


def get_space_mask(tokenizer, space_mask_mode):
    space_mask = np.zeros(len(tokenizer), dtype=bool)
    tokens = tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer)))
    special_token_ids = set(tokenizer.all_special_ids)

    space_mask_modes = set(space_mask_mode.split("+"))

    for i, token in enumerate(tokens):
        if len(token) == 0:
            continue

        if "space" in space_mask_modes and token[0] == "Ġ":
            space_mask[i] = True

        if "tab" in space_mask_modes and token[0] == "Ċ":
            space_mask[i] = True

        if "newline" in space_mask_modes and token[0] == "ĉ":
            space_mask[i] = True

        if "special" in space_mask_modes and i in special_token_ids:
            space_mask[i] = True

    return space_mask


def get_expand_input_ids_matrix(
    tokenizer,
    expand_input_ids_vocab,
    max_length=constants.EXPAND_INPUT_IDS_MAX_LENGTH,
    module=np,
):
    expansion_data = []
    expansion_indices = []

    vocab = tokenizer.get_vocab()

    for key, value in expand_input_ids_vocab.items():
        if key in vocab:
            indices = [vocab[key] + 1]
        else:
            indices = [
                # unsafe=True is OK because `tokenizer` must be byte-converted via conversion=byte, and thus has the correct vocab
                x + 1
                for x in tokenizer.convert_tokens_to_ids(
                    tokenizer.backend_tokenize(key, unsafe=True)
                )
            ][::-1][:max_length]
        while len(indices) < max_length:
            indices.append(0)

        expansion_data.append(1 + value)
        expansion_indices.append(indices)

    expansion_data.insert(0, -1)
    expansion_indices.insert(0, [0] * max_length)

    return (
        module.array(expansion_data, dtype=module.int32),
        module.array(expansion_indices, dtype=module.int32),
    )


def get_expand_input_ids_dict(
    tokenizer, expand_input_ids_vocab, max_length=constants.EXPAND_INPUT_IDS_MAX_LENGTH
):
    expansion_data, expansion_indices = get_expand_input_ids_matrix(
        tokenizer, expand_input_ids_vocab, max_length
    )

    return (
        {
            tuple(i for i in indices if i != 0): data
            for indices, data in zip(expansion_indices, expansion_data)
        },
        set(tokenizer.all_special_ids),
    )


def np_expand_input_ids(
    input_ids,
    expand_input_ids_dict,
    last_only=False,
    maxlen=constants.EXPAND_INPUT_IDS_MAX_LENGTH,
):
    expanded_input_ids = np.zeros_like(input_ids)

    for example_idx in range(len(input_ids)):
        last_maxlen_ids = []

        for i in range(len(input_ids[example_idx])):
            last_maxlen_ids.insert(0, input_ids[example_idx][i] + 1)
            if len(last_maxlen_ids) > maxlen:
                last_maxlen_ids.pop()

            if last_only and i < len(input_ids[example_idx]) - 1:
                continue

            if last_maxlen_ids[0] in expand_input_ids_dict[1]:
                expanded_input_ids[example_idx][i] = (
                    expand_input_ids_dict[0][(last_maxlen_ids[0],)] - 1
                )
            else:
                found = False
                last_maxlen_up_to = len(last_maxlen_ids)

                while not found and last_maxlen_up_to > 0:
                    try:
                        expanded_input_ids[example_idx][i] = (
                            expand_input_ids_dict[0][
                                tuple(last_maxlen_ids[:last_maxlen_up_to])
                            ]
                            - 1
                        )
                        found = True
                    except KeyError:
                        last_maxlen_up_to -= 1

    return expanded_input_ids


def jax_expand_input_ids(
    input_ids,
    expand_input_ids_matrix,
    last_only=False,
    maxlen=constants.EXPAND_INPUT_IDS_MAX_LENGTH,
):
    @partial(jax.jit, static_argnums=(1,))
    @partial(jax.vmap, in_axes=(0, None))
    def moving_window(a, size: int):
        padded_a = jnp.pad(a, ((0, size - 1),), mode="constant", constant_values=0)
        starts = jnp.arange(len(a))
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(padded_a, (start,), (size,))
        )(starts)

    input_ids_window = moving_window(input_ids[:, ::-1], maxlen) + 1
    input_ids_window_repeated = jnp.tril(
        jnp.tile(
            input_ids_window[:, :, None],
            (1, 1, maxlen, 1),
        )
    )[:, ::-1, ::-1, :]

    if last_only:
        input_ids_window_repeated = input_ids_window_repeated[:, -1:, :, :]

    def inner_fn(lookup_tokens):
        return (
            (expand_input_ids_matrix[1] == lookup_tokens[None, :]).all(axis=-1).argmax()
        )

    expanded_indices = jax.vmap(inner_fn)(
        input_ids_window_repeated.reshape(-1, maxlen)
    ).reshape(input_ids_window_repeated.shape[:-1])
    expanded_input_ids = (
        expand_input_ids_matrix[0][
            jnp.take_along_axis(
                expanded_indices,
                (expanded_indices > 0).argmax(-1, keepdims=True),
                axis=-1,
            )[:, :, 0]
        ]
        - 1
    )

    return expanded_input_ids


def fvt(
    source_tokenizer: ByteifyTokenizer,
    target_tokenizer: ByteifyTokenizer,
    source_embeddings,
    fallback_mode="random",
    verbose=True,
    allow_exact_match=True,
):
    # assumes both tokenizers are byte-level
    source_vocab = source_tokenizer.get_vocab()

    original_to_new_indices = np.zeros(len(target_tokenizer), dtype=int)
    diff_indices = []
    diff_embeddings = []

    stats = {
        "special_token_exact_match": 0,
        "exact_match": 0,
        "averaged": 0,
        "fallback": 0,
    }

    source_mean = source_embeddings.mean(0)
    source_std = source_embeddings.std(0)

    one_to_one_special_tokens_map = {}
    # special keys are ordered by importance (e.g., <bos> at the start)
    # so we copy them in reverse order to ensure that the most important one wins
    # in case of duplicates. This was a problem e.g. with transfer of Gemma3 to Qwen3:
    # ```
    # Copying special token <bos> -> <|endoftext|>
    # Copying special token <pad> -> <|endoftext|>
    # ```
    # (before reversal <pad> overwrote <bos>)
    for k in model_kinds.BaseModelKind.SPECIAL_KEYS[::-1]:
        v1 = source_tokenizer.model_kind_cls.replacements[k]
        v2 = target_tokenizer.model_kind_cls.replacements[k]

        if v1 is not None and v2 is not None:
            one_to_one_special_tokens_map[v2[0]] = v1[0]
            logger.info(f"Copying special token {v1[0]} -> {v2[0]}")
        elif v2 is not None:
            logger.warning(
                f"Special token {k} has no replacements in source tokenizer: {v1}. Not copying special token embedding."
            )

    for i in tqdm(
        range(len(target_tokenizer)), desc="Applying FVT..", disable=not verbose
    ):
        token = target_tokenizer.convert_ids_to_tokens(i)

        if token in one_to_one_special_tokens_map:
            stats["special_token_exact_match"] += 1
            original_to_new_indices[i] = source_vocab[
                one_to_one_special_tokens_map[token]
            ]
        elif (
            token in source_vocab
            and source_vocab[token] < len(source_embeddings)
            and allow_exact_match
        ):
            stats["exact_match"] += 1
            original_to_new_indices[i] = source_vocab[token]
        else:
            original_to_new_indices[i] = (
                0  # will be overwritten by setting diff_indices
            )
            diff_indices.append(i)

            if token in source_vocab:
                if source_vocab[token] < len(source_embeddings):
                    constituent_idx = np.array([source_vocab[token]])
                else:
                    constituent_idx = np.array([])
            else:
                try:
                    decomposed = source_tokenizer.convert_tokens_to_ids(
                        source_tokenizer.backend_tokenize(token)
                    )
                except UnicodeDecodeError:
                    decomposed = []
                constituent_idx = np.array(
                    [x for x in decomposed if x < len(source_embeddings)]
                )

            if len(constituent_idx) > 0:
                diff_embeddings.append(source_embeddings[constituent_idx].mean(0))
                stats["averaged"] += 1
            else:
                if fallback_mode == "random":
                    fallback_embedding = np.random.normal(
                        loc=source_mean,
                        scale=source_std,
                    )
                else:
                    fallback_embedding = source_embeddings[
                        source_tokenizer.unk_token_id
                    ]

                diff_embeddings.append(fallback_embedding)
                stats["fallback"] += 1

    logger.info(f"FVT exact match: {stats['exact_match']}")
    logger.info(f"FVT averaged: {stats['averaged']}")
    logger.info(f"FVT fallback: {stats['fallback']}")
    logger.info(f"FVT special token exact match: {stats['special_token_exact_match']}")

    diff_indices = np.array(diff_indices, dtype=int)
    diff_embeddings = np.array(diff_embeddings, dtype=np.float32)

    return diff_embeddings, original_to_new_indices, diff_indices


def label_by_prefix(pytree, label_maps, default=None):
    flat_pytree = traverse_util.flatten_dict(pytree)
    labels = {}

    for k in flat_pytree:
        for prefix, label in label_maps:
            is_match = (
                isinstance(prefix, str)
                and re.match(prefix, ".".join(k))
                or isinstance(prefix, tuple)
                and k[: len(prefix)] == prefix
            )

            if is_match:
                labels[k] = label
                break

        if k not in labels:
            if default is None:
                raise ValueError(f"No label found for key: {k}")
            else:
                labels[k] = default

    return traverse_util.unflatten_dict(labels)


def remove_none(pytree):
    flat_pytree = traverse_util.flatten_dict(pytree)
    for k in flat_pytree:
        if flat_pytree[k] is None:
            del flat_pytree[k]
    return traverse_util.unflatten_dict(flat_pytree)


def get_n_pad(n, pad_to_multiple_of):
    n_overflow = n % pad_to_multiple_of
    if n_overflow > 0:
        n_pad = pad_to_multiple_of - n_overflow
    else:
        n_pad = 0

    return n_pad


def print_example_alignments(teacher_tokens_to_chunks, student_tokens_to_chunks, teacher_tokens, student_tokens):
    n_chunks = teacher_tokens_to_chunks.shape[1]
    print(f"Example teacher tokens (n={teacher_tokens_to_chunks.sum()}):")
    for chunk_idx in range(n_chunks):
        chunk_tokens = [teacher_tokens[i] for i in np.where(teacher_tokens_to_chunks[:, chunk_idx])[0]]

        bg_color = (40 + chunk_idx % 8) # cycle through colors black (40) to white (47)
        print(f"\033[{bg_color}m" + " ".join(chunk_tokens) + "\033[0m", end=" ")

    print(f"\nExample student tokens (n={student_tokens_to_chunks.sum()}):")
    for chunk_idx in range(n_chunks):
        chunk_tokens = [student_tokens[i] for i in np.where(student_tokens_to_chunks[:, chunk_idx])[0]]
        
        bg_color = (40 + chunk_idx % 8) # cycle through colors black (40) to white (47)
        print(f"\033[{bg_color}m" + " ".join(chunk_tokens) + "\033[0m", end=" ")


def param_report(params, train_mask):
    # TODO: update with LoRA support
    return

    for key, value in params.items():

        @dataclass
        class ParamInfo:
            size: int
            trainable: bool

        def count_params(acc, info):
            total_count, trainable_count = acc

            return (
                total_count + info.size,
                trainable_count + info.size if info.trainable else trainable_count,
            )

        param_info = jax.tree.map(
            lambda x, trainable: ParamInfo(size=x.size, trainable=trainable),
            value,
            train_mask[key],
        )
        if not isinstance(param_info, dict):
            # make sure reduce works
            param_info = {"dummy": param_info}

        num_params, num_trainable_params = jax.tree.reduce(
            count_params,
            param_info,
            initializer=(0, 0),
        )

        # TODO: get rid of prints, and probably make return arg instead
        print(f"Num {key} params: {num_params}")
        print(f"Num {key} trainable params: {num_trainable_params}")


def get_surface_form_matrix(
    tokenizer_or_tokens,
    maxlen,
    hn_tokenizer: ByteifyTokenizer,
    padding=0,
    verbose=False,
):
    # tokens are expected to be byte encoded
    if isinstance(tokenizer_or_tokens, list):
        tokens = tokenizer_or_tokens
    else:
        tokenizer = tokenizer_or_tokens
        tokens = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))

    vocab_size = len(tokens)
    surface_form_matrix = np.full(
        (vocab_size + padding, maxlen),
        hn_tokenizer.pad_token_id if hn_tokenizer is not None else 0,
        dtype=np.int32,
    )

    n_truncated = 0

    for i, token in tqdm(enumerate(tokens), total=vocab_size, disable=not verbose):
        if token in hn_tokenizer.all_special_tokens:
            surface_form_matrix[i, 0] = hn_tokenizer.convert_tokens_to_ids(token)
            continue

        ids = hn_tokenizer.backend_tokenize_with_byte_fallback(token, unsafe="auto")

        if len(ids) > maxlen:
            ids = ids[:maxlen]
            n_truncated += 1

        surface_form_matrix[i, : len(ids)] = ids

    return surface_form_matrix, n_truncated


def preprocess_messages(messages):
    # convert messages format to prompt with chat template
    prompt = "<|<bos>|>"
    for message in messages:
        role_tag = {
            "user": "<|<user_name>|>",
            "assistant": "<|<assistant_name>|>",
            "system": "<|<system_name>|>",
        }[message["role"]]

        prompt += (
            f"<|<start_header>|>{role_tag}<|<end_header>|>{message['content']}<|<eot>|>"
        )
    return prompt


def preprocess_prompt(prompt, chat_template_mode):
    if chat_template_mode == "surround_instruct":
        prompt = f"<|<bos>|><|<start_header>|><|<user_name>|><|<end_header>|>{prompt}<|<eot>|><|<start_header>|><|<assistant_name>|><|<end_header>|>"
    elif chat_template_mode == "direct_encode":
        if not prompt.startswith("<|<bos>|>"):
            prompt = "<|<bos>|>" + prompt
        if not (prompt.endswith("<|<eot>|>") or prompt.endswith("<|<eos>|>")):
            prompt = prompt + "<|<eos>|>"
    elif chat_template_mode == "direct_encode_no_force_eos":
        if not prompt.startswith("<|<bos>|>"):
            prompt = "<|<bos>|>" + prompt
    elif chat_template_mode == "direct_encode_no_force_bos":
        if not (prompt.endswith("<|<eot>|>") or prompt.endswith("<|<eos>|>")):
            prompt = prompt + "<|<eos>|>"
    elif chat_template_mode == "direct_encode_no_force_bos_no_force_eos":
        pass
    else:
        raise ValueError(f"Unknown chat template mode: {chat_template_mode}")

    return prompt


def encode_prompt(prompt, tokenizer, max_length=None):
    token_ids = []
    regular_token_indices = []

    if max_length is not None:
        prompt = prompt[: constants.MAX_CHARS_PER_TOKEN * max_length]

    added_token_starts = set(x[0] for x in tokenizer.added_tokens_encoder.keys())

    def process_chunk(chunk):
        if chunk in tokenizer.added_tokens_encoder:
            token_ids.append(
                tokenizer.convert_tokens_to_ids(
                    tokenizer.model_kind_cls.byte_fallback_fn(chunk)
                )
            )
            regular_token_indices.append(-1)
        elif chunk in tokenizer.model_kind_cls.replacements:
            if tokenizer.model_kind_cls.replacements[chunk] is not None:
                token_ids.extend(
                    tokenizer.convert_tokens_to_ids(
                        tokenizer.model_kind_cls.replacements[chunk]
                    )
                )
                regular_token_indices.extend(
                    [-1] * len(tokenizer.model_kind_cls.replacements[chunk])
                )
        else:
            chunk_token_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            token_ids.extend(chunk_token_ids)

            try:
                regular_token_start = next(
                    i for i in regular_token_indices[::-1] if i != -1
                )
            except StopIteration:
                regular_token_start = -1

            regular_token_indices.extend(
                [regular_token_start + 1 + i for i in range(len(chunk_token_ids))]
            )

    start_i = 0
    i = 0

    while i < len(prompt):
        try:
            key = next(
                key
                for key in tokenizer.model_kind_cls.replacements.keys()
                if prompt[i:].startswith(key)
            )
        except StopIteration:
            key = None

        if key is None:
            if prompt[i] in added_token_starts:
                try:
                    key = next(
                        key
                        for key in tokenizer.added_tokens_encoder.keys()
                        if prompt[i:].startswith(key)
                    )
                except StopIteration:
                    key = None

        if key is not None:
            if start_i < i:
                chunk = prompt[start_i:i]
                process_chunk(chunk)
                start_i = i

            chunk = prompt[start_i : i + len(key)]
            process_chunk(chunk)
            start_i = i + len(key)
            i = start_i

            if max_length is not None and len(token_ids) >= max_length:
                return token_ids[:max_length], regular_token_indices[:max_length]
        else:
            i += 1

    if start_i < len(prompt):
        chunk = prompt[start_i:]
        process_chunk(chunk)

    if max_length is not None:
        return token_ids[:max_length], regular_token_indices[:max_length]
    else:
        return token_ids, regular_token_indices


def make_hashable(obj):
    """Recursively convert lists to tuples so they become hashable."""
    if isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(
            (k, make_hashable(v)) for k, v in sorted(obj.items())
        )  # Sort keys for consistency
    return obj


def init_linear(seed, in_shape, out_shape, dtype, **kwargs):
    return jax.device_get(
        flax.linen.Dense(features=out_shape, dtype=dtype, **kwargs).init(
            jax.random.PRNGKey(seed),
            jnp.ones((1, in_shape), dtype=dtype, device=jax.devices("cpu")[0]),
        )["params"]
    )


def compute_unigram_probabilities(tokenizer: ByteifyTokenizer, counts, additive_smoothing_constant=1e-9):
    counts = {int(k): v for k, v in counts.items()}

    # assign highest weight to special tokens
    max_value = max(counts.values())
    for special_token_id in tokenizer.all_special_ids:
        logger.info(f"Assigning max count {max_value} to special token {special_token_id}")
        counts[special_token_id] = max_value

    counts_sum = sum(counts.values())
    probs = np.array(
        [
            counts.get(token_id, 0) + additive_smoothing_constant * counts_sum
            for token_id in range(len(tokenizer))
        ],
        dtype=np.float32,
    )
    probs /= probs.sum()

    return probs


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "google/gemma-2-2b-it:source=Gemma2",
        "Qwen/Qwen2-1.5B-Instruct:source=Qwen2",
        "meta-llama/Llama-3.1-8B-Instruct:source=Llama3",
    ],
)
def test_encode_prompt(tokenizer_name):
    tokenizer = load_byteify_tokenizer(tokenizer_name)
    comparison_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name.split(":")[0])

    messages = [
        {"role": "user", "content": "Hello, world!"},
        {"role": "assistant", "content": "Hello, user!"},
    ]

    tokens, _ = encode_prompt(preprocess_messages(messages), tokenizer)
    comparison_token_ids = comparison_tokenizer.apply_chat_template(
        messages, use_system_prompt=False
    )

    tokens = comparison_tokenizer.convert_ids_to_tokens(
        tokenizer.convert_tokens_to_ids(tokens)
    )
    comparison_tokens = comparison_tokenizer.convert_ids_to_tokens(comparison_token_ids)

    # apply_chat_template may inject an (undesired) system prompt, so the best we can do is to check the suffix (and skip first token since it may be bos)
    assert " ".join(comparison_tokens).endswith(" ".join(tokens[1:]))


def test_expand_input_ids():
    tokenizer = load_byteify_tokenizer("google/gemma-2-2b-it:source=Gemma2")
    byte_tokenizer = load_byteify_tokenizer(
        "google/gemma-2-2b-it:source=Gemma2:conversion=byte"
    )

    expand_vocab = tokenizer.get_vocab()

    byte_input_ids = np.array(
        [byte_tokenizer.encode("Hello, world! How are you today?")]
    )
    expanded_input_ids_np = np_expand_input_ids(
        byte_input_ids, get_expand_input_ids_dict(byte_tokenizer, expand_vocab)
    )
    expanded_input_ids_jax = jax_expand_input_ids(
        byte_input_ids, get_expand_input_ids_matrix(byte_tokenizer, expand_vocab)
    )

    assert np.all(expanded_input_ids_np == expanded_input_ids_jax)
