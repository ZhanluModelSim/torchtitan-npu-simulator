# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.models.deepseek_v4.model import HcHead
from torchtitan_npu.ops.tilelang import mhc_head_compute_mix_tilelang

logger = logging.getLogger(__name__)


def _to_local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class NpuHcHeadComputeMixTilelang(HcHead):
    def __init__(self, parent: HcHead):
        self.__dict__.update(parent.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, DTensor):
            raise ValueError(
                "NpuHcHeadComputeMixTilelang expects local tensor input; apply HcHeadParallelStyle with local TP input."
            )
        hc_head_fn = _to_local_tensor(self.hc_head_fn)
        hc_head_base = _to_local_tensor(self.hc_head_base)
        hc_head_scale = _to_local_tensor(self.hc_head_scale)
        if isinstance(hc_head_fn, DTensor):
            hc_head_fn = hc_head_fn.to_local()
        if isinstance(hc_head_base, DTensor):
            hc_head_base = hc_head_base.to_local()
        if isinstance(hc_head_scale, DTensor):
            hc_head_scale = hc_head_scale.to_local()

        is_tnd = x.dim() == 3

        if is_tnd:
            x = x.flatten(1).unsqueeze(1)  # [T, N, D] -> [T, 1, N*D]
        elif x.dim() == 4:
            x = x.flatten(2)  # [B, S, N, D] -> [B, S, N*D]
        else:
            raise ValueError(
                f"NpuHcHeadComputeMixTilelang expects 3D [T, N, D] or 4D [B, S, N, D] tensor, "
                f"but got input with shape {x.shape}"
            )

        hc_mult = self.hc_head_fn.shape[0]
        y = mhc_head_compute_mix_tilelang(
            x,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            None,
            False,
            self.norm_eps,
            self.hc_eps,
            hc_mult,
        )

        if is_tnd:
            y = y.squeeze(1)  # [T, 1, out_features] -> [T, out_features]

        return y


class MHCHeadComputeMixTilelangConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in list(model.named_modules()):
            if not isinstance(module, HcHead):
                continue

            logger.info(f"Replacing {name} with NpuHcHeadComputeMixTilelang.")
            replace_module_with_name(model, name, NpuHcHeadComputeMixTilelang(module))


@register_model_converter("npu_mhc_head_compute_mix_tilelang")
class MHCHeadComputeMixTilelangConfig(ModelCustomConfig):
    model_converter = MHCHeadComputeMixTilelangConverter
