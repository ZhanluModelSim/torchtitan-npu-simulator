# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/hf_datasets/text_datasets.py

Unified patch for ChatDataset enhancements:
1. chat_encoder: configurable non-Jinja encoding (e.g. encoding_dsv4.py)
2. Multi-turn conversation support
3. Multi-turn label masking

Replaces the old tokenizer.py patch — encoder routing is now done here,
not on the tokenizer.
"""

import copy

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.text_datasets import ChatDataLoader, ChatDataset
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchtitan.hf_datasets import _mtp_seq_len_delta

_orig_loader_init = ChatDataLoader.__init__


def _prepare_upstream_loader_config(config):
    """Apply NPU CLI fields without mutating the registry-owned config."""

    original_load_kwargs = getattr(config, "load_dataset_kwargs", {})
    load_kwargs = dict(original_load_kwargs)

    dataset_split = getattr(config, "dataset_split", "")
    if dataset_split:
        load_kwargs["split"] = dataset_split

    data_files = getattr(config, "data_files", "")
    if data_files:
        load_kwargs["data_files"] = data_files

    dataset_config_name = getattr(config, "dataset_config_name", "")
    if dataset_config_name:
        load_kwargs["name"] = dataset_config_name

    cleaned_load_kwargs = {key: value for key, value in load_kwargs.items() if value not in (None, "")}

    original_sample_processor = getattr(config, "sample_processor", None)
    sample_processor = original_sample_processor
    chat_processor_name = getattr(config, "chat_processor", None)
    if chat_processor_name:
        from torchtitan_npu.hf_datasets.chat_processors import import_chat_processor

        sample_processor = import_chat_processor(chat_processor_name)

    if sample_processor is None:
        raise ValueError("ChatDataLoaderConfig requires either sample_processor or chat_processor import path.")

    if cleaned_load_kwargs == original_load_kwargs and sample_processor is original_sample_processor:
        return config

    prepared_config = copy.copy(config)
    prepared_config.load_dataset_kwargs = cleaned_load_kwargs
    prepared_config.sample_processor = sample_processor
    return prepared_config


def _patched_loader_init(
    self,
    config,
    *,
    dp_world_size,
    dp_rank,
    tokenizer,
    seq_len,
    local_batch_size,
    **kwargs,
):
    config = _prepare_upstream_loader_config(config)

    chat_encoder_cfg = getattr(config, "chat_encoder", None)
    chat_encoder = chat_encoder_cfg.build() if chat_encoder_cfg is not None else None

    seq_len = seq_len + _mtp_seq_len_delta()

    _orig_loader_init(
        self,
        config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        **kwargs,
    )

    if chat_encoder is not None:
        self.dataset._chat_encoder = chat_encoder
        logger.info(
            "[ChatDataset Patch] chat_encoder=%s installed",
            type(chat_encoder).__name__,
        )


ChatDataLoader.__init__ = _patched_loader_init


def _patched_validate_messages(messages):
    if len(messages) < 2:
        raise ValueError(f"Expected at least 2 messages, got {len(messages)}")

    _validate_message_roles(messages)
    if len(messages) == 2:
        _validate_two_message_chat(messages)
        return

    _validate_first_message(messages)
    start = 1 if messages[0].get("role") == "system" else 0
    remaining = messages[start:]

    if len(remaining) < 2:
        raise ValueError("Need at least one (non-assistant, assistant) pair after system")

    _validate_collapsed_turn_order(_collapse_non_assistant_turns(remaining, start))


def _validate_message_roles(messages):
    valid_roles = {"assistant", "system", "tool", "user"}
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in valid_roles:
            raise ValueError(f"Unknown role '{role}' at index {idx}")


def _validate_two_message_chat(messages):
    if messages[0].get("role") != "user":
        raise ValueError(f"First message must be 'user', got '{messages[0].get('role')}'")
    if messages[1].get("role") != "assistant":
        raise ValueError(f"Second message must be 'assistant', got '{messages[1].get('role')}'")


def _validate_first_message(messages):
    if messages[0].get("role") not in ("system", "user"):
        raise ValueError(f"First message must be 'user' or 'system', got '{messages[0].get('role')}'")


def _collapse_non_assistant_turns(remaining, start):
    # Collapse consecutive non-assistant messages (e.g. user + tool results)
    # into a single logical turn so the alternating check works with
    # OpenAI tool-call patterns: [assistant(tool_calls), tool, tool, ..., user, assistant, ...]
    collapsed = []  # (original_index, msg) pairs
    for idx, msg in enumerate(remaining):
        if msg.get("role") == "assistant":
            collapsed.append((start + idx, msg))
        else:
            if collapsed and collapsed[-1][1].get("role") != "assistant":
                collapsed[-1] = (start + idx, msg)
            else:
                collapsed.append((start + idx, msg))
    return collapsed


def _validate_collapsed_turn_order(collapsed):
    for i in range(0, len(collapsed), 2):
        orig_idx, msg = collapsed[i]
        if msg.get("role") == "assistant":
            raise ValueError(f"Expected non-assistant at position {orig_idx}, got 'assistant'")
        if i + 1 < len(collapsed):
            next_orig_idx, next_msg = collapsed[i + 1]
            if next_msg.get("role") != "assistant":
                raise ValueError(f"Expected 'assistant' at position {next_orig_idx}, got '{next_msg.get('role')}'")


ChatDataset._validate_messages = staticmethod(_patched_validate_messages)


def _patched_tokenize_sample(self, sample):
    messages = self._sample_processor(sample)
    self._validate_messages(messages)

    chat_encoder = getattr(self, "_chat_encoder", None)
    assistant_ranges = None
    if chat_encoder is not None:
        if not hasattr(chat_encoder, "encode_messages_with_assistant_ranges"):
            raise NotImplementedError(
                "ChatDataset only supports chat encoders that provide encode_messages_with_assistant_ranges()."
            )
        full_text, full_tokens, assistant_ranges = chat_encoder.encode_messages_with_assistant_ranges(
            messages,
            self._tokenizer,
            self._eos_id,
        )
    else:
        full_text = self._tokenizer.apply_chat_template(messages, tokenize=False)
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

    if not self._logged_first_sample:
        logger.info(
            "[ChatDataset Patch] First sample (total %d chars, %d tokens):\n"
            "--- HEAD (first 500 chars) ---\n"
            "%s\n"
            "--- TAIL (last 500 chars) ---\n"
            "%s",
            len(full_text),
            len(full_tokens),
            full_text[:500],
            full_text[-500:],
        )
        self._logged_first_sample = True

    if len(full_tokens) - 1 > self.seq_len:
        logger.debug(
            "Dropping sample %d: tokens %d exceeds seq_len %d",
            self._sample_idx,
            len(full_tokens) - 1,
            self.seq_len,
        )
        return None

    input_ids = full_tokens[:-1]
    label_ids = full_tokens[1:]

    if assistant_ranges is not None:
        label_ids = _labels_for_token_ranges(label_ids, assistant_ranges)
    else:
        assistant_ranges = _assistant_ranges_by_chatml_header_scan(self._tokenizer, full_tokens, self._eos_id)
        if not assistant_ranges:
            raise ValueError(
                "Unable to locate ChatML assistant ranges. Only ChatML tokenizer "
                "path and DSV4 segmented encoder path are supported."
            )
        label_ids = _labels_for_token_ranges(label_ids, assistant_ranges)

    if not _validate_label_ids(input_ids, label_ids):
        logger.warning(
            "Dropping sample %d: all labels are masked; check assistant spans and chat template.",
            self._sample_idx,
        )
        return None
    return input_ids, label_ids


def _assistant_ranges_by_chatml_header_scan(tokenizer, full_tokens, eos_id):
    assistant_header = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_bos=False,
        add_eos=False,
    )
    if not assistant_header:
        return []

    ranges = []
    header_len = len(assistant_header)
    pos = 0
    while pos < len(full_tokens):
        slice_end = pos + header_len
        if slice_end <= len(full_tokens) and full_tokens[pos:slice_end] == assistant_header:
            resp_start = slice_end
            resp_end = resp_start
            while resp_end < len(full_tokens) and full_tokens[resp_end] != eos_id:
                resp_end += 1
            ranges.append((resp_start, min(resp_end + 1, len(full_tokens))))
            pos = resp_end + 1
        else:
            pos += 1

    return ranges


def _labels_for_token_ranges(label_ids, token_ranges):
    masked = [IGNORE_INDEX] * len(label_ids)
    for start, end in token_ranges:
        # label_ids[i] is full_tokens[i + 1], so token ranges shift left by 1.
        label_start = max(start - 1, 0)
        label_end = min(end - 1, len(label_ids))
        for i in range(label_start, label_end):
            masked[i] = label_ids[i]

    return masked


def _validate_label_ids(input_ids, label_ids):
    if len(input_ids) != len(label_ids):
        raise ValueError(f"ChatDataset produced mismatched input/label lengths: {len(input_ids)} vs {len(label_ids)}")
    return not all(label == IGNORE_INDEX for label in label_ids)


ChatDataset._tokenize_sample = _patched_tokenize_sample
