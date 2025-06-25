from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from tokenkit.constants import BYTES_TO_CHARS, CHARS_TO_BYTES

BYTE_FALLBACK_MAP = {f"<0x{num:02X}>": num for num in range(256)}
INV_BYTE_FALLBACK_MAP = {v: k for k, v in BYTE_FALLBACK_MAP.items()}


def sentencepiece_byte_fallback_byte_fn(token: str) -> str:
    if token in BYTE_FALLBACK_MAP:
        return BYTES_TO_CHARS[BYTE_FALLBACK_MAP[token]]
    else:
        return "".join(
            BYTES_TO_CHARS[b] for b in token.replace("▁", " ").encode("utf-8")
        )


def sentencepiece_byte_fallback_precedence_fn(token: str) -> int:
    if token in BYTE_FALLBACK_MAP:
        return 0
    else:
        return 1


def identity_byte_fn(token: str) -> str:
    return token


class BaseModelKind(ABC):
    SPECIAL_KEYS = [
        "<|<bos>|>",
        "<|<pad>|>",
        "<|<start_header>|>",
        "<|<end_header>|>",
        "<|<eot>|>",
        "<|<system_name>|>",
        "<|<user_name>|>",
        "<|<assistant_name>|>",
    ]

    def __init__(self):
        self._byte_fallback_fn = identity_byte_fn
        self._byte_fallback_precedence_fn = lambda x: 0

    @property
    @abstractmethod
    def special_tokens(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        pass

    @property
    def byte_fallback_fn(self) -> Callable[[str], str]:
        return self._byte_fallback_fn

    @byte_fallback_fn.setter
    def byte_fallback_fn(self, value: Callable[[str], str]):
        self._byte_fallback_fn = value

    @property
    def byte_fallback_precedence_fn(self) -> Callable[[str], int]:
        return self._byte_fallback_precedence_fn

    @byte_fallback_precedence_fn.setter
    def byte_fallback_precedence_fn(self, value: Callable[[str], int]):
        self._byte_fallback_precedence_fn = value


class Qwen2ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": ["<|im_start|>"],
            "<|<end_header>|>": ["Ċ"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<eot>|>": ["<|im_end|>", "Ċ"],
            "<|<system_name>|>": ["system"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["assistant"],
        }

# Qwen3 does not require a bos token, but we add one for easier compatibility with other models
# this is supported per https://github.com/QwenLM/Qwen3/issues/486#issuecomment-2153775104
# the only difference between Qwen2 and Qwen3 model kind is that Qwen3 sets bos
class Qwen3ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<|endoftext|>"],
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": ["<|im_start|>"],
            "<|<end_header>|>": ["Ċ"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<eot>|>": ["<|im_end|>", "Ċ"],
            "<|<system_name>|>": ["system"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["assistant"],
        }


class Llama3ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<|begin_of_text|>"],
            "<|<pad>|>": ["<|end_of_text|>"],
            "<|<start_header>|>": ["<|start_header_id|>"],
            "<|<end_header>|>": ["<|end_header_id|>", "ĊĊ"],
            # give eot precedence over eos - not ideal but should work for chat templates
            "<|<eos>|>": ["<|eot_id|>"],
            "<|<eot>|>": ["<|eot_id|>"],
            "<|<system_name>|>": ["system"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["assistant"],
        }


class Gemma2ModelKind(BaseModelKind):
    def __init__(self):
        super().__init__()
        self._byte_fallback_fn = sentencepiece_byte_fallback_byte_fn
        self._byte_fallback_precedence_fn = sentencepiece_byte_fallback_precedence_fn

    @property
    def special_tokens(self) -> List[str]:
        return ["<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<bos>"],
            "<|<pad>|>": ["<pad>"],
            "<|<start_header>|>": ["<start_of_turn>"],
            "<|<end_header>|>": ["Ċ"],
            "<|<eos>|>": ["<eos>"],
            "<|<eot>|>": ["<end_of_turn>", "Ċ"],
            "<|<system_name>|>": ["user"],
            "<|<user_name>|>": ["user"],
            "<|<assistant_name>|>": ["model"],
        }


class Gemma3ModelKind(Gemma2ModelKind):
    pass


class Phi3ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": None,
            "<|<end_header>|>": ["Ċ"],
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<eot>|>": ["<|end|>", "Ċ"],
            "<|<system_name>|>": ["<|user|>"],
            "<|<user_name>|>": ["<|user|>"],
            "<|<assistant_name>|>": ["<|assistant|>"],
        }


class GPT2ModelKind(BaseModelKind):
    @property
    def special_tokens(self) -> List[str]:
        return ["<|endoftext|>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": None,
            "<|<pad>|>": ["<|endoftext|>"],
            "<|<start_header>|>": None,
            "<|<end_header>|>": None,
            "<|<eos>|>": ["<|endoftext|>"],
            "<|<eot>|>": ["<|endoftext|>"],
            "<|<system_name>|>": None,
            "<|<user_name>|>": None,
            "<|<assistant_name>|>": None,
        }


class TinyLlamaModelKind(BaseModelKind):
    def __init__(self):
        super().__init__()
        self._byte_fallback_fn = sentencepiece_byte_fallback_byte_fn
        self._byte_fallback_precedence_fn = sentencepiece_byte_fallback_precedence_fn

    @property
    def special_tokens(self) -> List[str]:
        return ["<s>", "</s>", "<unk>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<s>"],
            "<|<pad>|>": ["</s>"],
            "<|<start_header>|>": None, # chat template exists but not supported for TinyLlama
            "<|<end_header>|>": None,
            "<|<eos>|>": ["</s>"],
            "<|<eot>|>": ["</s>"],
            "<|<system_name>|>": None,
            "<|<user_name>|>": None,
            "<|<assistant_name>|>": None,
        }


class MistralModelKind(BaseModelKind):
    def __init__(self):
        super().__init__()

    @property
    def special_tokens(self) -> List[str]:
        return ["<s>", "</s>", "<unk>"]

    @property
    def replacements(self) -> Dict[str, Optional[List[str]]]:
        return {
            "<|<bos>|>": ["<s>"],
            "<|<pad>|>": ["</s>"],
            "<|<start_header>|>": None, # chat template exists but not supported for Mistral
            "<|<end_header>|>": None,
            "<|<eos>|>": ["</s>"],
            "<|<eot>|>": ["</s>"],
            "<|<system_name>|>": None,
            "<|<user_name>|>": None,
            "<|<assistant_name>|>": None,
        }

# Model kind registry
def get_model_kind_cls(model_kind: str) -> BaseModelKind:
    return {
        "Qwen2": Qwen2ModelKind(),
        "Qwen3": Qwen3ModelKind(),
        "Llama3": Llama3ModelKind(),
        "Gemma2": Gemma2ModelKind(),
        "Gemma3": Gemma3ModelKind(),
        "Phi3": Phi3ModelKind(),
        "GPT2": GPT2ModelKind(),
        "TinyLlama": TinyLlamaModelKind(),
        "Mistral": MistralModelKind(),
    }[model_kind]
