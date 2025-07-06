import numpy as np
from jax.sharding import PartitionSpec as P

from tokenkit import utils


class PlainCollator:
    def __init__(
        self,
        tokenizer,
        max_length,
        use_chat_template=False,
        chat_template_mode="direct_encode",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.chat_template_mode = chat_template_mode

    def _encode_with_chat_template(self, texts, tokenizer, max_length):
        input_ids = np.full(
            (len(texts), max_length), fill_value=tokenizer.pad_token_id, dtype=np.int32
        )
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int32)

        for i in range(len(texts)):
            current_input_ids, _ = utils.encode_prompt(
                utils.preprocess_prompt(texts[i], self.chat_template_mode),
                tokenizer,
                max_length=max_length,
            )
            input_ids[i, : len(current_input_ids)] = current_input_ids
            attention_mask[i, : len(current_input_ids)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __call__(self, examples):
        # batched internally
        examples = examples[0]

        texts = examples["text"]

        if self.use_chat_template:
            encoding = self._encode_with_chat_template(
                texts,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )
        else:
            encoding = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return batch

    def get_batch_pspecs(self):
        batch_specs = {
            "input_ids": P("data", None),
            "attention_mask": P("data", None),
        }

        return batch_specs
