"""Flax TPU LLaMA model."""

import math
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as pallas_flash_attention,
)
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_call_sample_docstring,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_tpu_llama import TPULlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TPULlamaConfig"
_CHECKPOINT_FOR_DOC = "afmck/testing-llama-tiny"
_REAL_CHECKPOINT_FOR_DOC = "openlm-research/open_llama_3b_v2"

LLAMA_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`LlamaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16`, or
            `jax.numpy.bfloat16`.

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

remat = nn_partitioning.remat

# adapted from modeling_rope_utils
def _compute_default_rope_parameters(
    config=None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
):
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.int32).astype(jnp.float32) / dim))
    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config, seq_len: Optional[int] = None, **rope_kwargs
):
    # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
    # No need to keep BC with longrope, unreleased when this new pattern was created.
    if len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` should be unset in `_compute_longrope_parameters`, got "
            f"{rope_kwargs}"
        )

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]
    short_factor = config.rope_scaling["short_factor"]
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if hasattr(config, "original_max_position_embeddings"):
        if seq_len and seq_len < config.original_max_position_embeddings:
            expanded_max_position_embeddings = config.original_max_position_embeddings
        else:
            expanded_max_position_embeddings = config.max_position_embeddings
        max_position_embeddings = config.original_max_position_embeddings
        factor = expanded_max_position_embeddings / max_position_embeddings
    else:
        max_position_embeddings = config.max_position_embeddings
        expanded_max_position_embeddings = max_position_embeddings * factor

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if expanded_max_position_embeddings > max_position_embeddings:
        ext_factors = jnp.array(long_factor, dtype=jnp.float32)
    else:
        ext_factors = jnp.array(short_factor, dtype=jnp.float32)
    inv_freq_shape = jnp.arange(0, dim, 2, dtype=jnp.int64).astype(jnp.float32) / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


def _compute_llama3_parameters(config, seq_len: Optional[int] = None, **rope_kwargs):
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len, **rope_kwargs)

    factor = config.rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
    "longrope": _compute_longrope_parameters,
}


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    freqs = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")

    emb = np.concatenate((freqs, freqs), axis=-1)
    out = np.concatenate((np.sin(emb)[:, None, :], np.cos(emb)[:, None, :]), axis=-1)
    return jnp.array(out[:, :, :num_pos])


# Copied from transformers.models.llama.modeling_flax_llama.rotate_half
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# Adapted from transformers.models.llama.modeling_flax_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos[:, :, None, :]) + (rotate_half(tensor) * sin_pos[:, :, None, :])


class FlaxTPULlamaRMSNorm(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32
    override_dim: int = None

    def setup(self):
        if self.override_dim is not None:
            dim = self.override_dim
        else:
            dim = self.config.hidden_size

        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), dim)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxTPULlamaRotaryEmbedding(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.rope_kwargs = {}

        if self.config.rope_scaling is not None:
            self.rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = self.config.max_position_embeddings
        self.original_max_seq_len = self.config.max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, **self.rope_kwargs)
        self.inv_freq = self.original_inv_freq = inv_freq

    def __call__(self, x, position_ids):
        inv_freq_expanded = jnp.tile(
            self.inv_freq[None, :, None].astype(jnp.float32),
            (position_ids.shape[0], 1, 1),
        )
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)

        freqs = jnp.swapaxes(jnp.matmul(inv_freq_expanded, position_ids_expanded), 1, 2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


class FlaxTPULlamaAttention(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.embed_dim // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        dense = partial(
            nn.Dense,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.q_proj = dense(self.num_heads * self.head_dim)
        self.k_proj = dense(self.num_key_value_heads * self.head_dim)
        self.v_proj = dense(self.num_key_value_heads * self.head_dim)
        self.o_proj = dense(self.embed_dim)

        if self.config.add_qk_norm:
            self.q_norm = FlaxTPULlamaRMSNorm(self.config, dtype=self.dtype, override_dim=self.head_dim)
            self.k_norm = FlaxTPULlamaRMSNorm(self.config, dtype=self.dtype, override_dim=self.head_dim)

        self.causal_mask = make_causal_mask(
            jnp.ones(
                (1, getattr(config, "max_length", config.max_position_embeddings)),
                dtype="bool",
            ),
            dtype="bool",
        )

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads * self.head_dim,))

    @nn.compact
    # Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoSelfAttention._concatenate_to_cache
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        raw_query = self.q_proj(hidden_states)
        raw_key = self.k_proj(hidden_states)
        raw_value = self.v_proj(hidden_states)

        query = self._split_heads(raw_query, self.num_heads)
        key = self._split_heads(raw_key, self.num_key_value_heads)
        value = self._split_heads(raw_value, self.num_key_value_heads)

        if self.config.add_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        cos, sin = position_embeddings
        query = apply_rotary_pos_emb(query, sin, cos)
        key = apply_rotary_pos_emb(key, sin, cos)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        else:
            assert attention_mask.ndim == 4

        attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # usual dot product attention
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output, self.num_heads)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, (raw_query, raw_key, raw_value)) if output_attentions else (attn_output,)
        return outputs


class FlaxTPULlamaFlashAttention(FlaxTPULlamaAttention):
    def setup(self):
        super().setup()

        if self.num_heads % len(jax.devices()) != 0:
            # TODO: warn or pad attention heads or neither or both?
            shard_across_model = False
        else:
            shard_across_model = True

        model_partition = "model" if shard_across_model else None
        data_partition = "data"

        self.flash_attn_fn = shard_map(
            partial(
                pallas_flash_attention,
                sm_scale=1 / math.sqrt(self.head_dim),
                causal=True,
            ),
            mesh=getattr(self.config, "mesh"),
            in_specs=(
                # bnlh
                P(data_partition, model_partition, None, None),
                P(data_partition, model_partition, None, None),
                P(data_partition, model_partition, None, None),
                # P(),
            ),
            # bnlh
            out_specs=P(data_partition, model_partition, None, None),
            check_rep=False,
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        raw_query = self.q_proj(hidden_states)
        raw_key = self.k_proj(hidden_states)
        raw_value = self.v_proj(hidden_states)

        query = self._split_heads(raw_query, self.num_heads)
        key = self._split_heads(raw_key, self.num_key_value_heads)
        value = self._split_heads(raw_value, self.num_key_value_heads)

        cos, sin = position_embeddings
        query = apply_rotary_pos_emb(query, sin, cos)
        key = apply_rotary_pos_emb(key, sin, cos)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask,
                (0, 0, mask_shift, 0),
                (1, 1, query_length, max_decoder_length),
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        else:
            assert attention_mask.ndim == 4

        attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)

        # transform boolean mask into float mask
        # attention_bias = lax.select(
        #     attention_mask > 0,
        #     jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
        #     jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
        #         self.dtype
        #     ),
        # )

        query = jnp.swapaxes(query, 1, 2)
        key = jnp.swapaxes(key, 1, 2)
        value = jnp.swapaxes(value, 1, 2)

        # TODO: revisit attention_bias when implementing packing
        # attention_bias = jnp.broadcast_to(
        #     attention_bias, (batch_size, self.num_heads, query_length, key_length)
        # )

        # flash attn needs fp32
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)

        # usual dot product attention
        attn_output = self.flash_attn_fn(
            query,
            key,
            value,
        ).astype(hidden_states.dtype)
        attn_output = jnp.swapaxes(attn_output, 1, 2)
        attn_output = self._merge_heads(attn_output, self.num_heads)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, (raw_query, raw_key, raw_value)) if output_attentions else (attn_output,)
        return outputs


class FlaxTPULlamaMLP(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = ACT2FN[self.config.hidden_act]

        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


LLAMA_ATTENTION_CLASSES = {
    "eager": FlaxTPULlamaAttention,
    "pallas_flash_attention": FlaxTPULlamaFlashAttention,
}


class FlaxTPULlamaDecoderLayer(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxTPULlamaRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = LLAMA_ATTENTION_CLASSES[self.config._attn_implementation](self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxTPULlamaRMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxTPULlamaMLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.NamedSharding(getattr(self.config, "mesh"), P("data", None, "model"))
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # residual connection
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.NamedSharding(getattr(self.config, "mesh"), P("data", None, "model"))
        )

        mlp_output = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + mlp_output

        return (hidden_states, attn_output, mlp_output)


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Llama, GPT_NEO->LLAMA, transformer->model
class FlaxTPULlamaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TPULlamaConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: TPULlamaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        module = self.module_class(
            config=config,
            dtype=dtype,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(
            config=self.config,
            dtype=self.dtype,
            gradient_checkpointing=True,
        )

    @classmethod
    def can_generate(cls) -> bool:
        # disable generation, handled separately
        # this is convenient since GenerationConfig.from_model_config(config) needs a pickleable config
        return False

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, None, attention_mask, position_ids, return_dict=False)[
            "params"
        ]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_ids,
            None,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Need to provide either input_ids or inputs_embeds (and not both)")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape
        else:
            batch_size, sequence_length, _ = inputs_embeds.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxTPULlamaAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4") if input_ids is not None else None,
            inputs_embeds if inputs_embeds is not None else None,
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxTPULlamaLayerCollection(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.rotary_emb = FlaxTPULlamaRotaryEmbedding(self.config, dtype=self.dtype)

        if self.gradient_checkpointing:
            FlaxTPULlamaDecoderCheckpointLayer = remat(FlaxTPULlamaDecoderLayer, static_argnums=(4, 5, 6))
            self.blocks = [
                FlaxTPULlamaDecoderCheckpointLayer(self.config, dtype=self.dtype, name=str(i))
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            self.blocks = [
                FlaxTPULlamaDecoderLayer(self.config, dtype=self.dtype, name=str(i))
                for i in range(self.config.num_hidden_layers)
            ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = [(), ()] if output_hidden_states else None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if output_hidden_states:
            all_hidden_states[0] += (hidden_states,)
            all_hidden_states[1] += (hidden_states,)

        for block_idx, block in enumerate(self.blocks):
            layer_outputs = block(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_hidden_states:
                if block_idx != len(self.blocks) - 1:
                    all_hidden_states[0] += (hidden_states,)
                all_hidden_states[1] += layer_outputs[1:]

            if output_attentions:
                raise NotImplementedError("Attention outputs are not implemented for TPULLama (with projections).")

        # this contains possible `None` values - `FlaxTPULlamaModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxTPULlamaModule(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        self.layers = FlaxTPULlamaLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.norm = FlaxTPULlamaRMSNorm(self.config, dtype=self.dtype)

    def embed(
        self,
        input_ids,
    ):
        return self.embed_tokens(input_ids.astype("i4"))

    def __call__(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        outputs = self.layers(
            inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if not self.config.skip_out_norm:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1]
            all_hidden_states[0] += (hidden_states,)

            # different from the HF default, `hidden_states` stores residual and non-residual representations
            # at this point. we return only the first one, which is the residual representation, for consistency
            outputs = (hidden_states, all_hidden_states[0]) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


@add_start_docstrings(
    "The bare Llama Model transformer outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class FlaxTPULlamaModel(FlaxTPULlamaPreTrainedModel):
    module_class = FlaxTPULlamaModule


append_call_sample_docstring(
    FlaxTPULlamaModel,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


class FlaxTPULlamaForCausalLMModule(nn.Module):
    config: TPULlamaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.model = FlaxTPULlamaModule(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def embed(self, input_ids):
        return self.model.embed(input_ids)

    def __call__(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        lm_logits = jax.lax.with_sharding_constraint(
            lm_logits,
            jax.sharding.NamedSharding(getattr(self.config, "mesh"), P("data", None, "model")),
        )

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a language modeling head (linear layer) on top.
    """,
    LLAMA_START_DOCSTRING,
)
# Copied from transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM with GPTJ->Llama
class FlaxTPULlamaForCausalLM(FlaxTPULlamaPreTrainedModel):
    module_class = FlaxTPULlamaForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Llama uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


append_call_sample_docstring(
    FlaxTPULlamaForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)