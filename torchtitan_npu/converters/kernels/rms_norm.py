# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu
from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.magi2_preview.norms import MultiModalityRMSNorm

logger = logging.getLogger(__name__)


def _get_eps(module: nn.Module) -> float | None:
    for attr_name in ["eps", "variance_epsilon", "epsilon"]:
        eps = getattr(module, attr_name, None)
        if eps is not None:
            return float(eps)
    return None


class NPURMSNorm(RMSNorm):
    def __init__(self, parent: RMSNorm):
        normalized_shape = parent.normalized_shape
        if len(normalized_shape) != 1:
            raise ValueError(
                "NPURMSNorm supports one-dimensional normalized_shape, got "
                f"{normalized_shape}"
            )
        super().__init__(
            RMSNorm.Config(
                normalized_shape=normalized_shape[0],
                eps=_get_eps(parent),
                elementwise_affine=parent.elementwise_affine,
                param_init=getattr(parent, "_param_init", None),
            )
        )
        if parent.weight is not None:
            self.register_parameter("weight", parent.weight)
        self.eps = _get_eps(parent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches the default implementation of nn.RMSNorm:
        # - Use user-provided eps if it exists.
        # - Otherwise, use the machine epsilon of the current input `x`.
        resolved_eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return torch_npu.npu_rms_norm(x, self.weight, resolved_eps)[0]

    def _init_self_parameters(self) -> None:
        if getattr(self, "_param_init", None) is not None:
            super()._init_self_parameters()
        else:
            self.reset_parameters()


class NpuMultiModalityRMSNorm(MultiModalityRMSNorm):
    """MAGI-2 MultiModalityRMSNorm backed by ``torch_npu.npu_rms_norm``.

    ``num_modality == 1`` maps to a single fused call with the gain
    (``weight + weight_bias``) as the kernel gamma. ``num_modality > 1``
    runs one fused call per modality segment (rows arrive modality-sorted
    and ``m_splits`` gives the per-modality row counts; the math is
    row-local, so segmenting is exact). With ``num_patterns > 1`` the gain
    is per-pattern and cannot fold into the kernel gamma, so the kernel
    runs with an all-ones gamma and the gain multiplies afterwards.
    """

    def __init__(self, parent: MultiModalityRMSNorm) -> None:
        nn.Module.__init__(self)
        self.dim = parent.dim
        self.eps = parent.eps
        self.num_modality = parent.num_modality
        self.num_patterns = parent.num_patterns
        self.out_dtype = parent.out_dtype
        self.weight_bias = parent.weight_bias
        self.register_parameter("weight", parent.weight)

    def _norm_with_gain(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        if self.num_patterns == 1:
            return torch_npu.npu_rms_norm(x, gain.reshape(-1), self.eps)[0]
        ones = torch.ones(self.dim, device=x.device, dtype=torch.float32)
        return torch_npu.npu_rms_norm(x, ones, self.eps)[0] * gain

    def forward(
        self, x: torch.Tensor, m_splits: list[int] | None = None
    ) -> torch.Tensor:
        out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
        if self.num_modality == 1:
            gain = self.weight.view(self.num_patterns, self.dim) + self.weight_bias
            return self._norm_with_gain(x, gain).to(out_dtype)

        if m_splits is None:
            raise ValueError("m_splits is required when num_modality > 1")
        splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else m_splits
        splits = [int(s) for s in splits]
        if len(splits) != self.num_modality:
            raise ValueError(
                f"Expected {self.num_modality} m_splits entries, got {len(splits)}"
            )

        scaled = []
        for chunk, w in zip(
            torch.split(x, splits, dim=0),
            self.weight.chunk(self.num_modality),
            strict=True,
        ):
            if chunk.shape[0] == 0:
                scaled.append(chunk)
                continue
            gain = w.view(self.num_patterns, self.dim) + self.weight_bias
            scaled.append(self._norm_with_gain(chunk, gain))
        return torch.cat(scaled, dim=0).to(out_dtype)


class NpuRMSNormConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in list(model.named_modules()):
            if type(module) is MultiModalityRMSNorm:
                replace_module_with_name(
                    model, name, NpuMultiModalityRMSNorm(module)
                )
            elif isinstance(module, RMSNorm):
                replace_module_with_name(model, name, NPURMSNorm(module))


@register_model_converter("npu_rms_norm")
class RMSNormModelConfig(ModelCustomConfig):
    model_converter = NpuRMSNormConverter
