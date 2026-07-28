# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.config.configs import ChatDataLoaderConfig


class ChatMLTokenizer:
    eos_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_bos=True, add_eos=False):
        tokens = [ord(character) for character in text]
        if add_bos:
            tokens = [200] + tokens
        if add_eos:
            tokens.append(self.eos_id)
        return tokens


def process_sample(sample):
    return [
        {"role": "user", "content": sample["q"]},
        {"role": "assistant", "content": sample["a"]},
    ]


DATASET_RECORDS = [
    {"q": "AAAA", "a": "BBBB"},
    {"q": "CCCC", "a": "DDDD"},
    {"q": "EEEE", "a": "FFFF"},
]
SEQ_LEN = 200
SAMPLE_CONTENT_TOKEN_IDS = {ord(character) for character in "ABCDEF"}


def make_chat_dataloader_config():
    return ChatDataLoaderConfig(
        dataset_path="mock",
        load_dataset_kwargs={},
        sample_processor=process_sample,
        infinite=False,
    )


def next_single_sequence(loader):
    """Consume one local-batch-size-one batch from a ChatDataLoader."""

    input_batch, label_batch = next(iter(loader))
    assert input_batch["input"].shape[0] == 1
    assert label_batch.shape[0] == 1
    return input_batch["input"][0].tolist(), label_batch[0].tolist()


def sample_content_text(input_ids):
    """Extract the example question/answer characters from token IDs."""

    return "".join(
        chr(token_id) for token_id in input_ids if token_id in SAMPLE_CONTENT_TOKEN_IDS
    )
