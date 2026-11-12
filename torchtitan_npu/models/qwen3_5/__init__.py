# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from dataclasses import dataclass

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.models.qwen3_5 import build_model_config as _upstream_build_model_config
from torchtitan.models.qwen3_5.model import Qwen35Model

from torchtitan_npu.patches.torchtitan.models.qwen3_5 import (
    vision_encoder as _vision_patch,
)

_vision_patch.apply()


class NPUQwen35Model(Qwen35Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        text_only: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.text_only = config.text_only

    def forward(self, *args, **kwargs):
        if self.text_only:
            kwargs.setdefault("special_tokens", {"image_id": -1, "video_id": -1})
        return super().forward(*args, **kwargs)

    def parallelize(self, **kwargs):
        from torchtitan_npu.models.common.compile_policy import parallelize_with_model_compile

        return parallelize_with_model_compile(self, **kwargs)

    def apply_model_compile(self, model_compile):
        from torchtitan_npu.models.common.compile_policy import compile_transformer_blocks
        from torchtitan_npu.override.qwen3_5.parallelize import _qwen_aot_eager_compile_options

        with _qwen_aot_eager_compile_options(model_compile.backend):
            compile_transformer_blocks(self, model_compile)

    def _apply_fsdp(self, **kwargs):
        from torchtitan_npu.override.qwen3_5.parallelize import apply_qwen3_5_npu_fsdp

        apply_qwen3_5_npu_fsdp(self, **kwargs)


def model_registry(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> NPUQwen35Model.Config:
    """Build the NPU Qwen3.5 config with the upstream flavor parameters."""
    config = _upstream_build_model_config(
        flavor,
        parallelism=parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )
    return derive(config, NPUQwen35Model.Config)
