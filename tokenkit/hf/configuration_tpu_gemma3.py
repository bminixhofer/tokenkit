"""TPU Gemma3 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class TPUGemma3Config(PretrainedConfig):
    model_type = "tpu_gemma3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        cache_implementation="hybrid",
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=6,
        expand_input_ids=False, # Transformers-native PyTorch generation support
        expand_input_ids_maxlen=None,
        expand_input_ids_vocab_size=None,
        expand_input_ids_dict=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation

        self.rope_local_base_freq = rope_local_base_freq
        # For configuring HybridCache to work with 5:1 attention pattern
        self.sliding_window_pattern = sliding_window_pattern
        self.rope_scaling = rope_scaling
        rope_config_validation(self)

        self.expand_input_ids = expand_input_ids
        self.expand_input_ids_maxlen = expand_input_ids_maxlen
        self.expand_input_ids_vocab_size = expand_input_ids_vocab_size
        self.expand_input_ids_dict = expand_input_ids_dict