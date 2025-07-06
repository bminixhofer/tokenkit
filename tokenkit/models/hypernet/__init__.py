import math
from pprint import pprint
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import copy

from transformers import PretrainedConfig

from tokenkit.models import param

EPSILON = 1e-8

class HypernetConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int,
        max_seq_length: int,
        num_embeddings: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_attention: bool = True,
        multiply_hidden_dim_by_num_embeddings: bool = True,
        hidden_expansion_factor: int = 2,
        num_layers: int = 3,
        num_heads: int = 16,
        residual: bool = True,
        residual_alpha: float = 8.0,
        shared: bool = True,
        use_attention_mask: bool = True,
        pooling: str = "first",  # "first", "mean"
        residual_pooling: str = "first",  # "first", "mean"
        architecture: str = "transformer",  # 'transformer', 'linear', 'identity'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.num_embeddings = num_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_attention = use_attention
        self.multiply_hidden_dim_by_num_embeddings = multiply_hidden_dim_by_num_embeddings
        self.hidden_expansion_factor = hidden_expansion_factor
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.residual = residual
        self.residual_alpha = residual_alpha
        self.shared = shared
        self.use_attention_mask = use_attention_mask
        self.pooling = pooling
        self.residual_pooling = residual_pooling
        self.architecture = architecture


class EmbeddingRescaler(nn.Module):
    shape: tuple[int] = ()
    axes: tuple[int] = (0,)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.w = self.param(
            "w",
            jax.nn.initializers.constant(1),
            (1,) * len(self.axes) + self.shape,
            self.dtype,
        )
        self.b = self.param(
            "b",
            jax.nn.initializers.constant(0.0),
            (1,) * len(self.axes) + self.shape,
            self.dtype,
        )

    def __call__(self, x):
        return x * self.w + self.b

    def scale_to(x, target=None, target_means=None, target_stds=None, axes=(0,)):
        if target_stds is None:
            target_stds = target.std(axis=axes)
        if target_means is None:
            target_means = target.mean(axis=0)

        w = (target_stds / (x.std(axis=axes) + EPSILON))[None]
        b = (target_means - (x * w).mean(axis=axes))[None]

        return w, b


class HypernetSelfAttention(nn.Module):
    config: HypernetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        attention_bias=None,
        init_cache: bool = False,
        deterministic=True,
        output_attentions: bool = False,
    ):
        assert (attention_bias is None) or (attention_mask is None) # can not have both

        batch_size = hidden_states.shape[0]

        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # combine masks if needed
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if attention_bias is not None:
            attention_bias = jnp.expand_dims(attention_bias, axis=(-3, -2))

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_bias = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class HypernetMLP(nn.Module):
    config: HypernetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.intermediate = nn.Dense(
            self.config.hidden_size * self.config.hidden_expansion_factor,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.output = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states


class HypernetTransformerLayer(nn.Module):
    config: HypernetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.use_attention:
            self.self_attention = HypernetSelfAttention(
                config=self.config, dtype=self.dtype
            )
        self.mlp = HypernetMLP(config=self.config, dtype=self.dtype)

        self.post_attention_layernorm = nn.LayerNorm(
            dtype=self.dtype, epsilon=self.config.layer_norm_eps
        )
        self.post_mlp_layernorm = nn.LayerNorm(
            dtype=self.dtype, epsilon=self.config.layer_norm_eps
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
    ):
        if self.config.use_attention:
            attention_output, _ = self.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                deterministic=True,
                output_attentions=False,
            )
            hidden_states = self.post_attention_layernorm(
                hidden_states + attention_output
            )

        mlp_output = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states + mlp_output)

        return hidden_states


class HypernetTransformerStack(nn.Module):
    config: HypernetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.position_embeddings = nn.Embed(
            num_embeddings=self.config.max_seq_length,
            features=self.config.hidden_size,
        )

        self.input_layernorm = nn.LayerNorm(
            dtype=self.dtype, epsilon=self.config.layer_norm_eps
        )

        self.layers = [
            HypernetTransformerLayer(config=self.config, dtype=self.dtype)
            for _ in range(self.config.num_layers)
        ]

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]))

        if position_ids is None:
            position_ids = jnp.arange(hidden_states.shape[1], dtype=jnp.int32)
            position_ids = jnp.expand_dims(position_ids, axis=0)

        # Add position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Hypernet(nn.Module):
    config: HypernetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.in_rescaler = EmbeddingRescaler(
            shape=(self.config.hidden_size,),
            axes=(0, 1),
            dtype=self.dtype,
        )
        self.out_rescaler = EmbeddingRescaler(
            shape=(self.config.hidden_size,),
            axes=(0,),
            dtype=self.dtype,
        )

        if not self.config.shared:
            raise NotImplementedError("Non-shared hypernet not implemented.")

        if self.config.architecture == "transformer":
            if self.config.multiply_hidden_dim_by_num_embeddings:
                hidden_size = self.config.hidden_size * self.config.num_embeddings
                transformer_config = copy.deepcopy(self.config)
                transformer_config.hidden_size = hidden_size
            else:
                transformer_config = self.config

            if not self.config.multiply_hidden_dim_by_num_embeddings:
                self.input_linear = nn.Dense(
                    self.config.hidden_size,
                    dtype=self.dtype,
                    kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                )

            self.output_linear = nn.Dense(
                self.config.hidden_size * self.config.num_embeddings,
                kernel_init=jax.nn.initializers.zeros, # residual
            )
            self.transformer = HypernetTransformerStack(
                config=transformer_config, dtype=self.dtype
            )
        elif self.config.architecture == "linear":
            self.linears = [
                nn.Dense(
                    self.config.hidden_size,
                    dtype=self.dtype,
                    kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                )
            ]
        elif self.config.architecture == "identity":
            pass
        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")

    def __call__(self, embeddings, attention_mask):
        # input (embeddings): [vocab_size, seq_length, num_embeddings, hidden_size]
        # output: [vocab_size, num_embeddings, hidden_size]
        if self.config.architecture == "identity":
            return embeddings[:, 0, :, :]

        if not self.config.use_attention_mask:
            attention_mask = jnp.ones_like(attention_mask)

        vocab_size, seq_length, _, _ = embeddings.shape
        embeddings = self.in_rescaler(embeddings)

        if self.config.architecture == "transformer":
            x = jnp.reshape(
                embeddings,
                (vocab_size, seq_length, self.config.hidden_size * self.config.num_embeddings),
            )
            if not self.config.multiply_hidden_dim_by_num_embeddings:
                x = self.input_linear(x)

            # TODO: impl packing
            x = self.transformer(x, attention_mask=attention_mask)

            # take embedding of the first token in the sequence as the pooled prediction
            if self.config.pooling == "first":
                x = x[:, 0, :]
            elif self.config.pooling == "mean":
                x = x.mean(axis=1, where=attention_mask[:, :, None] > 0)
            else:
                raise ValueError(f"Unknown pooling method: {self.config.pooling}")

            x = self.output_linear(x)
            x = jnp.reshape(x, (vocab_size, self.config.num_embeddings, self.config.hidden_size))

            if self.config.residual:
                residual_weight = self.config.residual_alpha / math.sqrt(self.config.hidden_size)
                if self.config.residual_pooling == "first":
                    non_residual = embeddings[:, 0, :, :]
                elif self.config.residual_pooling == "mean":
                    non_residual = embeddings.mean(axis=1, where=attention_mask[:, :, None, None] > 0)
                else:
                    raise ValueError(
                        f"Unknown pooling method: {self.residual_pooling}"
                    )

                predicted_embeddings = non_residual + residual_weight * x
            else:
                predicted_embeddings = x
        elif self.config.architecture == "linear":
            raise NotImplementedError("Linear architecture not implemented")

        return self.out_rescaler(predicted_embeddings)

    def init(self, rngs, embeddings, attention_mask):
        params = super().init(
            rngs, embeddings, attention_mask
        )

        # somewhat arbitrary, use ~Xavier normal
        in_std = math.sqrt(2.0 / self.config.hidden_size)

        in_w, in_b = EmbeddingRescaler.scale_to(
            embeddings, target_means=0, target_stds=in_std, axes=(0, 1)
        )

        params = param.put(params, "in_rescaler.w", in_w)
        params = param.put(params, "in_rescaler.b", in_b)

        preds = self.apply(params, embeddings, attention_mask)

        out_w, out_b = EmbeddingRescaler.scale_to(
            preds, target=embeddings[:, 0], axes=(0,)
        )

        params = param.put(params, "out_rescaler.w", out_w)
        params = param.put(params, "out_rescaler.b", out_b)

        return params


if __name__ == "__main__":
    config = HypernetConfig(
        hidden_size=768,
        architecture="transformer",
        use_attention=False,
        max_seq_length=8,
    )

    model = Hypernet(config=config, dtype=jnp.float32)
    x = np.random.randn(128, 4, 2, 768)
    attention_mask = np.ones((128, 4), dtype=np.float32)
    params = model.init(jax.random.PRNGKey(0), x, attention_mask)

    preds = model.apply(params, x, attention_mask)

    pprint(jax.tree.map(jnp.shape, params))
