"""Flax TPU Gemma3 model."""

from typing import Optional, Tuple
import copy

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
from jax.sharding import PartitionSpec as P

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_tpu_gemma3 import TPUGemma3Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TPUGemma3Config"
_CHECKPOINT_FOR_DOC = "google/gemma-2-2b"
_REAL_CHECKPOINT_FOR_DOC = "openlm-research/open_llama_3b_v2"

TPU_GEMMA3_START_DOCSTRING = r"""

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
        config ([`GemmaConfig`]): Model configuration class with all the parameters of the model.
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

TPU_GEMMA3_INPUTS_DOCSTRING = r"""
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

def _compute_linear_scaling_rope_parameters(
    config=None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
):
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_linear_scaling_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        factor = rope_kwargs["factor"]
    elif config is not None:
        factor = config.rope_scaling["factor"]

    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len, **rope_kwargs)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    inv_freq /= factor
    return inv_freq, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
}

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


class FlaxTPUGemma3RMSNorm(nn.Module):
    config: TPUGemma3Config
    dim_override: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps

        if self.dim_override is not None:
            self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.dim_override)
        else:
            self.weight = self.param("weight", lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        if self.add_in_projection:
            hidden_states = hidden_states @ self.in_projection

        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)

        hidden_states = (1 + self.weight) * jnp.asarray(hidden_states, dtype=self.dtype)

        if self.add_out_projection:
            hidden_states = hidden_states @ self.out_projection

        return hidden_states


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaRotaryEmbedding with Llama->Gemma3
class FlaxTPUGemma3RotaryEmbedding(nn.Module):
    config: TPUGemma3Config
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


class FlaxTPUGemma3Attention(nn.Module):
    config: TPUGemma3Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        self.is_sliding = bool((self.layer_idx + 1) % self.config.sliding_window_pattern)
        self.sliding_window = self.config.sliding_window if self.is_sliding else None

        config = self.config
        self.embed_dim = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # otherwise we would manually have to scale attn weights
        assert config.query_pre_attn_scalar == config.head_dim

        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        kernel = jax.nn.initializers.normal(self.config.initializer_range)
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel,
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=kernel,
        )
        self.q_norm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype, dim_override=self.head_dim)
        self.k_norm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype, dim_override=self.head_dim)

        self.o_proj = nn.Dense(self.embed_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel)

        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads * self.head_dim,))

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

        query = self.q_norm(query)
        key = self.k_norm(key)

        cos, sin = position_embeddings

        key = jnp.asarray(apply_rotary_pos_emb(key, sin, cos), dtype=self.dtype)
        query = jnp.asarray(apply_rotary_pos_emb(query, sin, cos), dtype=self.dtype)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        if self.sliding_window is not None:
            min_dtype = jnp.finfo(hidden_states.dtype).min
            sliding_window_mask = jnp.tril(
                jnp.ones_like(attention_mask, dtype=bool), k=-self.sliding_window
            )
            attention_mask = jnp.where(sliding_window_mask, min_dtype, attention_mask)
            if attention_mask.shape[-1] <= 1:  # when decoding
                attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        key = jnp.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, repeats=self.num_key_value_groups, axis=2)

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

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = jnp.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output, (raw_query, raw_key, raw_value)) if output_attentions else (attn_output,)
        return outputs


class FlaxTPUGemma3MLP(nn.Module):
    config: TPUGemma3Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        if self.config.hidden_activation is None:
            logger.warning_once(
                "Gemma3's activation function should be approximate GeLU and not exact GeLU. "
                "Changing the activation function to `gelu_pytorch_tanh`."
                f"if you want to use the legacy `{self.config.hidden_act}`, "
                f"edit the `model.config` to set `hidden_activation={self.config.hidden_act}` "
                "  instead of `hidden_act`. See https://github.com/huggingface/transformers/pull/29402 for more details."
            )
            hidden_activation = "gelu_pytorch_tanh"
        else:
            hidden_activation = self.config.hidden_activation
        self.act = ACT2FN[hidden_activation]

        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaDecoderLayer with Llama->Gemma3
class FlaxTPUGemma3DecoderLayer(nn.Module):
    config: TPUGemma3Config
    layer_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxTPUGemma3Attention(self.config, self.layer_idx, dtype=self.dtype)
        self.pre_feedforward_layernorm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype)
        self.post_feedforward_layernorm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxTPUGemma3MLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        mesh = getattr(self.config, "mesh", None)
        if mesh is not None:
            hidden_states = jax.lax.with_sharding_constraint(
                hidden_states, jax.sharding.NamedSharding(mesh, P("data", None, "model"))
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

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
        attn_output = self.post_attention_layernorm(outputs[0])
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        mlp_output = self.post_feedforward_layernorm(hidden_states)
        # residual connection
        hidden_states = residual + mlp_output

        return (hidden_states, attn_output, mlp_output)


# Copied from transformers.models.gpt_neo.modeling_flax_gpt_neo.FlaxGPTNeoPreTrainedModel with GPTNeo->Gemma3, GPT_NEO->Gemma3, transformer->model
class FlaxTPUGemma3PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TPUGemma3Config
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: TPUGemma3Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

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

        random_params = self.module.init(rngs, input_ids, None, attention_mask, position_ids, return_dict=False)["params"]

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
            jax.random.PRNGKey(0), input_ids, None, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    @add_start_docstrings_to_model_forward(TPU_GEMMA3_INPUTS_DOCSTRING)
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

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGemma3Attention module
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


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaLayerCollection with Llama->Gemma3
class FlaxTPUGemma3LayerCollection(nn.Module):
    config: TPUGemma3Config
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.rotary_emb = FlaxTPUGemma3RotaryEmbedding(config=self.config)

        mesh = getattr(self.config, "mesh", None)
        del self.config.mesh
        local_config = copy.deepcopy(self.config)
        if mesh is not None:
            self.config.mesh = mesh

        local_config.rope_theta = self.config.rope_local_base_freq
        local_config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = FlaxTPUGemma3RotaryEmbedding(config=local_config)

        if self.gradient_checkpointing:
            FlaxTPUGemma3DecoderCheckpointLayer = remat(FlaxTPUGemma3DecoderLayer, static_argnums=(4, 5, 6))
            self.blocks = [
                FlaxTPUGemma3DecoderCheckpointLayer(self.config, layer_idx, dtype=self.dtype, name=str(layer_idx))
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        else:
            self.blocks = [
                FlaxTPUGemma3DecoderLayer(self.config, layer_idx, dtype=self.dtype, name=str(layer_idx))
                for layer_idx in range(self.config.num_hidden_layers)
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

        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        if output_hidden_states:
            all_hidden_states[0] += (hidden_states,)
            all_hidden_states[1] += (hidden_states,)

        for block_idx, block in enumerate(self.blocks):
            layer_outputs = block(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_hidden_states:
                # last block is followed by norm - added later
                if block_idx != len(self.blocks) - 1:
                    all_hidden_states[0] += (hidden_states,)

                all_hidden_states[1] += layer_outputs[1:]

            if output_attentions:
                raise NotImplementedError("Attention outputs are not implemented for TPUGemma3 (with projections).")

        # this contains possible `None` values - `FlaxGemma3Module` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaModule with Llama->Gemma3
class FlaxTPUGemma3Module(nn.Module):
    config: TPUGemma3Config
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
        self.layers = FlaxTPUGemma3LayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.norm = FlaxTPUGemma3RMSNorm(self.config, dtype=self.dtype)

    def embed(
        self,
        input_ids,
    ):
        inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        scaler = self.config.hidden_size ** 0.5
        inputs_embeds = inputs_embeds * scaler
        inputs_embeds = inputs_embeds @ self.embedding_projection

        return inputs_embeds

    # Ignore copy
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
    "The bare Gemma3 Model transformer outputting raw hidden-states without any specific head on top.",
    TPU_GEMMA3_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaModel with Llama->Gemma3
class FlaxTPUGemma3Model(FlaxTPUGemma3PreTrainedModel):
    module_class = FlaxTPUGemma3Module


append_call_sample_docstring(
    FlaxTPUGemma3Model,
    _CHECKPOINT_FOR_DOC,
    FlaxBaseModelOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)


# Copied from transformers.models.llama.modeling_flax_llama.FlaxLlamaForCausalLMModule with Llama->Gemma3
class FlaxTPUGemma3ForCausalLMModule(nn.Module):
    config: TPUGemma3Config
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.model = FlaxTPUGemma3Module(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def embed(self, input_ids):
        return self.model.embed(input_ids)

    # Ignore copy
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
        # should be skipped automatically in this case (since unused), but check if JIT actually does this
        if not self.config.skip_out_norm:
            if self.config.tie_word_embeddings:
                shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
                lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
            else:
                lm_logits = self.lm_head(hidden_states)

            lm_logits = jax.lax.with_sharding_constraint(
                lm_logits,
                jax.sharding.NamedSharding(getattr(self.config, "mesh"), P("data", None, "model")),
            )

            if self.config.final_logit_softcapping is not None:
                lm_logits = lm_logits / self.config.final_logit_softcapping
                lm_logits = jnp.tanh(lm_logits)
                lm_logits = lm_logits * self.config.final_logit_softcapping
        else:
            lm_logits = None

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@add_start_docstrings(
    """
    The Gemma3 Model transformer with a language modeling head (linear layer) on top.
    """,
    TPU_GEMMA3_START_DOCSTRING,
)
# Copied from transformers.models.gptj.modeling_flax_gptj.FlaxGPTJForCausalLM with GPTJ->Gemma3
class FlaxTPUGemma3ForCausalLM(FlaxTPUGemma3PreTrainedModel):
    module_class = FlaxTPUGemma3ForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Gemma3 uses a causal mask, those positions are masked anyways.
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
    FlaxTPUGemma3ForCausalLM,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)