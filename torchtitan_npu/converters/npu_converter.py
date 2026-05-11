# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base_converter import BaseConverter


class NPUConverter(Configurable, ModelConverter):
    """NPU Converter wrapper that follows the Configurable pattern."""

    _patch_cls: type["BaseConverter"] | None = None
    _patch_name: str | None = None
    _supported_models: set[str] | None = None

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.parallel_dims = parallel_dims
        self.model_compile_enabled = model_compile_enabled
        self.model_name = "unknown"

    def convert(self, model: nn.Module) -> nn.Module:  # pyrefly: ignore [bad-override]
        self.model_name = self._infer_model_name(model)
        self._validate_compatibility()

        try:
            if self._patch_cls is None:
                raise RuntimeError("Missing patch class for NPUConverter")
            count = self._patch_cls.apply(model, self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply patch '{self._patch_name}' : {e}"
            ) from e
        if count > 0:
            logger.info(
                f"[NPU-CONVERTER] Applied '{self._patch_name}' : {count} replacements"
            )
        else:
            logger.warning(f"[NPU-CONVERTER] Applied no '{self._patch_name}' converter")
        return model

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        pass

    def _infer_model_name(self, model: nn.Module) -> str:
        module = model.__class__.__module__
        for name in ("deepseek_v32", "deepseek_v3", "qwen3", "llama4", "llama3"):
            if name in module:
                return name
        return "unknown"

    def _validate_compatibility(self):
        if self._patch_cls is None:
            raise RuntimeError("Missing patch class for NPUConverter")
        if not self._patch_cls.is_compatible(None, self.model_name):
            raise ValueError(
                f"Patch '{self._patch_name}' is NOT compatible with model '{self.model_name}' \n"
                f"Supported models: {self._supported_models}"
            )
