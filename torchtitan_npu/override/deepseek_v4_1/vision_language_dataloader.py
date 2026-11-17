# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select the DeepSeek-V4.1 vision-language dataloader without adding model recipes."""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import DeepSeekV41DataLoader
from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import DeepSeekV41VisionLanguageDataLoader
from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import ReasoningEffort, ThinkingMode


@override(
    target=DeepSeekV41DataLoader.Config,
    fqns=["dataloader"],
    exact=True,
    description="Use structured DeepSeek-V4.1 multimodal chat data for SFT",
)
def sft(
    cfg: DeepSeekV41DataLoader.Config,
    *,
    image_root: str = "",
    thinking_mode: ThinkingMode = "chat",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
) -> DeepSeekV41VisionLanguageDataLoader.Config:
    if not cfg.dataset_path:
        raise ValueError("SFT requires a JSON, JSONL, or Parquet dataset via --dataloader.dataset-path")
    return derive(
        cfg,
        DeepSeekV41VisionLanguageDataLoader.Config,
        image_root=image_root,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )
