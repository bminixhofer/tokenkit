from dataclasses import dataclass, fields, is_dataclass
from transformers import HfArgumentParser
from argparse import ArgumentParser
import yaml


@dataclass
class HypernetArgs:
    architecture: str
    num_layers: int
    residual: bool
    residual_alpha: float
    use_attention: bool
    use_attention_mask: bool = False
    num_heads: int = 16
    shared: bool = True
    multiply_hidden_dim_by_num_embeddings: bool = True

@dataclass
class EvalArgs:
    tasks: list[str]
    lengths: list[int]
    tokens_per_batch: int
    add_bos: bool
    chat_template_mode: str
    confirm_run_unsafe_code: bool
    limit: int | None = None

@dataclass
class ModelArgs:
    pretrained_model_name_or_path: str
    tokenizer_name: str
    revision: str | None = None

def restore_dataclasses(args, cls):
    for field in fields(cls):
        if is_dataclass(field.type):
            setattr(
                args,
                field.name,
                restore_dataclasses(getattr(args, field.name), field.type),
            )
        elif isinstance(field.type, list) and is_dataclass(field.type.__args__[0]):
            setattr(
                args,
                field.name,
                [
                    restore_dataclasses(item, field.type.__args__[0])
                    for item in getattr(args, field.name)
                ],
            )
        elif isinstance(field.type, dict):
            setattr(
                args,
                field.name,
                {
                    k: restore_dataclasses(v, field.type.__args__[1])
                    for k, v in getattr(args, field.name).items()
                },
            )

    if not isinstance(args, cls):
        return cls(**args) if args is not None else None

    return args


def parse_args(cls):
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", type=str, nargs="*")
    meta_args = parser.parse_args()

    (args,) = HfArgumentParser([cls]).parse_yaml_file(meta_args.config)

    for overrides in (meta_args.overrides or []):
        for override in overrides.split():
            first_equals = override.find("=")
            key = override[:first_equals].split(".")
            try:
                value = yaml.safe_load(override[first_equals + 1 :])
            except yaml.YAMLError:
                raise ValueError(f"Invalid YAML: {override[first_equals + 1 :]}")

            current = args
            for k in key[:-1]:
                if isinstance(current, list):
                    current = current[int(k)]
                elif isinstance(current, dict):
                    current = current[k]
                else:
                    current = getattr(current, k)

            if isinstance(current, list):
                if int(key[-1]) >= len(current):
                    raise ValueError(f"Invalid key: {key[-1]}")
                current[int(key[-1])] = value
            elif isinstance(current, dict):
                current[key[-1]] = value
            else:
                if not hasattr(current, key[-1]):
                    raise ValueError(f"Invalid key: {key[-1]}")
                setattr(current, key[-1], value)

    return restore_dataclasses(args, cls)
