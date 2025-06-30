<h1 align="center">tokenkit🔁</h1>
<h3 align="center">Tokenization Transfer for LLMs</h3>

<div align="center">
<img src="https://github.com/user-attachments/assets/976478c8-8994-4780-8d77-b429ec707932" width="600">
</div>

`tokenkit` is a toolkit implementing advanced methods to transfer *models* and *model knowledge* across tokenizers.

## News

- __2025-04-23__: A new guide on [implementing cross-tokenizer distillation via ALM from scratch in PyTorch](./docs/pytorch_alm_from_scratch.ipynb)! 🔥
- __2025-04-22__: New [Llama3-2-3B-IT-Byte](https://huggingface.co/benjamin/Llama3-2-3B-IT-Byte) and [Gemma2-2B-IT-Byte](https://huggingface.co/benjamin/Gemma2-2B-IT-Byte) checkpoints with native `transformers` support (plus, documentation on how to train them). Also, new guides for [running tokenizer transfer](./docs/tokenizer_transfer.md) and [byteification](./docs/byteification.md)!
- __2025-04-02__: The initial release of `tokenkit` with support for cross-tokenizer distillation via ALM and Zero-Shot Tokenizer Transfer via FVT!

## Contents
- [Why Transfer Across Tokenizers?](#why-transfer-across-tokenizers)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Guides](#guides)
    - [Tokenizer Transfer via tokenkit](./docs/tokenizer_transfer.md)
    - [Byteification: A Unified Interface to Tokenizers](./docs/byteification.md)
    - [Implementing ALM From Scratch in PyTorch](./docs/pytorch_alm_from_scratch.ipynb) (new! 🔥)
- [Features](#features)
    - [Cross-Tokenizer Distillation](#cross-tokenizer-distillation)
    - [Zero-Shot Tokenizer Transfer](#zero-shot-tokenizer-transfer)
    - [Token-Level Ensembling & Evaluating Transferred Models](#token-level-ensembling--evaluating-transferred-models)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Why Transfer Across Tokenizers?

LLMs are bound to the tokenizer they were pretrained with. This limits their adaptability, reusability and modularity. Tokenizer transfer can lift this limitation. For example:
- If we want to reuse an LLM trained primarily on English in another language, we might want to update its tokenizer to one that is more suitable for the new language.
- If we want to combine (e.g., token-level ensemble) two LLMs, we need to transfer them to a common tokenizer.
- If we want to experiment with better tokenization schemes (e.g., byte-level tokenization), we might want to transfer an existing LLM to this tokenizer instead of training a new one expensively from scratch.
- If we want to transfer knowledge from a large teacher model to a smaller student model (which uses another tokenizer), we might want to use *cross-tokenizer distillation* to directly transfer the teacher's knowledge to the student without the need to first transfer the teacher to the student's tokenizer.

This library aims to let you accomplish all of this.

## Installation

`tokenkit` is primarily implemented in Jax, using PyTorch for data loading (so your PyTorch installation does not need to support an accelerator). Recommended installation:

<details>
<summary><strong>TPU</strong></summary>

```bash
# Clone the repository & install the library
git clone https://github.com/bminixhofer/tokenkit

# Create a new virtual environment
# Currently, requires Python <=3.10, but we are working on this: https://github.com/bminixhofer/tokenkit/issues/4
python -m venv tokenkit_env
. tokenkit_env/bin/activate

# Install torch & jax 0.5.0
pip install torch jax[tpu]==0.5.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Currently, tokenkit relies on a fork of `lm_eval`
pip install git+https://github.com/bminixhofer/lm-evaluation-harness

# Install the library and the remaining dependencies
pip install -r requirements.txt
pip install -e .
```
</details>

<details>
<summary><strong>GPU</strong></summary>

```bash
# Clone the repository & install the library
git clone https://github.com/bminixhofer/tokenkit

# Create a new virtual environment
# Currently, requires Python <=3.10, but we are working on this: https://github.com/bminixhofer/tokenkit/issues/4
python -m venv tokenkit_env
. tokenkit_env/bin/activate

# Install torch & jax 0.5.0
# you may need to substitute cuda12 with the version of CUDA you are using:
pip install torch jax[cuda12]==0.5.0

# Currently, tokenkit relies on a fork of `lm_eval`
pip install git+https://github.com/bminixhofer/lm-evaluation-harness

# Install the library and the remaining dependencies
pip install -r requirements.txt
pip install -e .
```
</details>

## Quickstart

After installing the library, you can play around with the scripts in `examples/` to get started immediately. For example:

```
bash examples/llama3_to_byte_tokenizer_gpu.sh
```

If you're interested in reproducing or improving on a public model which has been trained via ALM, you can also take a look at the `tokenkit` command used to train that model, for example [in the Training section of the Llama3-2-3B-IT-Byte model card](https://huggingface.co/benjamin/Llama3-2-3B-IT-Byte#training).

## Guides

- [Tokenizer Transfer via tokenkit](./docs/tokenizer_transfer.md) (start here!)
- [Byteification: A Unified Interface to Tokenizers](./docs/byteification.md)
- [Implementing ALM From Scratch in PyTorch](./docs/pytorch_alm_from_scratch.ipynb) (interactive notebook)

## Features

### Cross-Tokenizer Distillation

`tokenkit` supports [Approximate Likelihood Matching (ALM)](https://arxiv.org/abs/2503.20083) for cross-tokenizer distillation. ALM usually performs best, but we have also implemented the following baselines:

- [Dual Space Knowledge Distillation (DSKD)](https://arxiv.org/abs/2406.17328)
- [Universal Logit Distillation (ULD)](https://arxiv.org/abs/2402.12030)
- [Minimum Edit Distance Logit Alignment (MinED)](https://arxiv.org/abs/2401.10491)

You can run cross-tokenizer distillation using the [`scripts/cross_tokenizer_distill.py`](scripts/cross_tokenizer_distill.py) script. See [`examples`](examples) for examples on transferring to different subword tokenizers and to byte-level tokenization.

### Zero-Shot Tokenizer Transfer

`tokenkit` supports Zero-Shot Tokenizer Transfer (ZeTT) via [Fast Vocabulary Transfer (FVT)](https://aclanthology.org/2022.emnlp-industry.41). Zero-Shot Tokenizer Transfer is usually used to obtain a good initialization for additional training, but can in some cases also be useful on its own. See our [ZeTT paper](https://arxiv.org/abs/2405.07883) for more details.

You can run Zero-Shot Tokenizer Transfer using the [`scripts/zett.py`](scripts/zett.py) script.

**🚧 We are working on implementing more ZeTT methods (including hypernetwork training introduced [here](https://arxiv.org/abs/2405.07883)).**

### Token-Level Ensembling & Evaluating Transferred Models

`tokenkit` supports autoregressive generation & loglikelihood scoring evaluation by implementing a Jax backend to the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Alongside generating from single models, you can also generate from *token-level ensembles* of models. There are some predefined ensembles in [`configs/models`](configs/models). For example, this evaluates a token-level ensemle of Llama and Qwen on MMLU: 

```bash
python3 scripts/eval_lockstep.py \
  models=llama_qwen \
  eval.tasks=[mmlu]
```

To evaluate pretrained byte-level models, you'll need to pass embeddings to expand the input ids with (i.e., to use as n-gram embeddings). For example:

```bash
python3 scripts/eval.py \
  model.pretrained_model_name_or_path=\'benjamin/Gemma2-2B-IT-Byte\' \
  model.tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2:conversion=byte\' \
  expand_model.pretrained_model_name_or_path=\'benjamin/gemma-2-2b-it-flax\' \
  expand_model.tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2\' \
  eval.tasks=[mmlu]
```

To evaluate any other model (e.g., subword-to-subword transferred models), use something like the following:

```bash
python3 scripts/eval.py \
  model.pretrained_model_name_or_path=\'benjamin/Gemma2-2B-IT-with-Qwen2-Tokenizer\' \
  model.tokenizer_name=\'benjamin/Gemma2-2B-IT-with-Qwen2-Tokenizer:source=Gemma2:conversion=prebyteified\' \
  eval.tasks=[mmlu] \
```

## Citation

To refer to this repository or to cite Approximate Likelihood Matching, please use this citation:

```
@article{alm,
  title={Cross-Tokenizer Distillation via Approximate Likelihood Matching},
  author={Minixhofer, Benjamin and Vuli{\'c}, Ivan and Ponti, Edoardo Maria},
  journal={arXiv preprint arXiv:2503.20083},
  year={2025}
}
```

Please use this citation for Zero-Shot Tokenizer Transfer:

```
@inproceedings{zett,
title={Zero-Shot Tokenizer Transfer},
author={Benjamin Minixhofer and Edoardo Ponti and Ivan Vuli{\'c}},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=RwBObRsIzC}
}
```

## Acknowledgments

Constituent projects (ALM, ZeTT) were supported by a Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137; 2022-) awarded to Ivan Vulić, by the Google Cloud Research Credits program with the award GCP329647813, and by Cloud TPUs from Google’s TPU Research Cloud (TRC). The name `tokenkit` and the README layout were inspired by [mergekit](https://github.com/arcee-ai/mergekit). [big_vision](https://github.com/google-research/big_vision) was extremely useful as a high-quality reference JAX training codebase.
