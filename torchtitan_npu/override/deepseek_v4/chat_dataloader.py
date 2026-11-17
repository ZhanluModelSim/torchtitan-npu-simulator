# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select the DeepSeek-V4 chat dataloader without adding model recipes."""

from torchtitan.config import derive, override
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

from torchtitan_npu.models.deepseek_v4.chat_dataset import DeepSeekV4ChatDataLoader
from torchtitan_npu.models.deepseek_v4.chat_encoder import DSV4EncoderConfig, ReasoningEffort, ThinkingMode


@override(
    target=HuggingFaceTextDataLoader.Config,
    fqns=["dataloader"],
    exact=True,
    description="Use structured DeepSeek-V4 chat data for SFT",
)
def sft(
    cfg: HuggingFaceTextDataLoader.Config,
    *,
    thinking_mode: ThinkingMode = "thinking",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
) -> DeepSeekV4ChatDataLoader.Config:
    if not cfg.dataset_path or cfg.dataset_path == "json":
        raise ValueError("SFT requires a JSONL or JSON file via --dataloader.dataset-path")
    return derive(
        cfg,
        DeepSeekV4ChatDataLoader.Config,
        dataset_path="json",
        data_files=cfg.dataset_path,
        chat_encoder=DSV4EncoderConfig(
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
        ),
    )
