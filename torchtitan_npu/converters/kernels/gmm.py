# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/models/moe/moe.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor

from ..base_converter import BaseConverter
from ..convert_utils import replace_functions, replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)

# Calculate the number of experts and EP degree, which are used as parameters
# when invoking operators during Hifloat8 low-precision training.
group_size_params = {
    "num_experts": None,
    "expert_model_parallel_size": None,
    "g_size": None,
}


def npu_grouped_mm(x, weight, group_list):
    # This function is replaced at runtime by quantization converters
    # (e.g. HiF8 / MXFP8) that patch the reference to quantize inputs
    # before the grouped MM (see patches/quantization/quantize.py).
    return torch._grouped_mm(x, weight, group_list)


def _run_experts_grouped_mm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
) -> torch.Tensor:
    # pyrefly: ignore [missing-attribute]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)

    h = npu_grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets)
    h = torch_npu.npu_swiglu(h, dim=-1)
    out = npu_grouped_mm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)

    return out


def npu_grouped_experts_forward(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    is_tp = False
    if isinstance(self.w2, DTensor):
        w2 = self.w2.to_local()
        w13 = self.w13.to_local() if self.w13 is not None else None
        from torch.distributed.tensor.placement_types import Shard as _Shard

        for p in self.w2.placements:
            if isinstance(p, _Shard) and p.dim == 2:
                is_tp = True
                break
        tp_group = self.w2.device_mesh.get_group() if is_tp else None
        logged_attr = "_logged"
        if not hasattr(npu_grouped_experts_forward, logged_attr):
            setattr(npu_grouped_experts_forward, logged_attr, True)
            logger.info(
                f"[GMM-TP] w2 placements={self.w2.placements}, is_tp={is_tp}, "
                f"w2 local shape={w2.shape}, w13 local shape={w13.shape if w13 is not None else None}"
            )
    else:
        w2 = self.w2
        w13 = self.w13
        tp_group = None

    # pyrefly: ignore [bad-argument-type]
    out = _run_experts_grouped_mm(w13, w2, None, x, num_tokens_per_expert)

    if is_tp and tp_group is not None:
        import torch.distributed as dist

        pre_ar = out.mean().item()
        dist.all_reduce(out, group=tp_group)
        post_ar = out.mean().item()
        ar_logged_attr = "_ar_logged"
        if not hasattr(npu_grouped_experts_forward, ar_logged_attr):
            setattr(npu_grouped_experts_forward, ar_logged_attr, True)
            ratio = post_ar / pre_ar if pre_ar != 0 else float("inf")
            logger.info(
                "[GMM-TP] all-reduce: pre_mean=%.6f, post_mean=%.6f, ratio=%s",
                pre_ar,
                post_ar,
                ratio,
            )

    return out


def npu_grouped_experts_init_weights(self, init_std: float):
    for w in [self.w2, self.w13]:
        if w is not None:
            nn.init.normal_(w, mean=0.0, std=init_std)


class NpuGroupedExperts(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.w2 = module.w2
        self.w13 = module.w13

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return npu_grouped_experts_forward(self, x, num_tokens_per_expert)


@register_npu_converter("npu_gmm")
class GMMKernel(BaseConverter):

    TARGET_PACKAGE = "torchtitan.models.common.moe"
    TARGET_CLASS = "GroupedExperts"

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:

        replacement_counts = 0

        # 1. Replacing GroupedExperts methods
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="forward",
            new_method=npu_grouped_experts_forward,
            package=cls.TARGET_PACKAGE,
        )

        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="init_weights",
            new_method=npu_grouped_experts_init_weights,
            package=cls.TARGET_PACKAGE,
        )

        # 2. Replacing module function _run_experts_grouped_mm
        func_replacements = replace_functions(
            func_name="_run_experts_grouped_mm",
            new_func=_run_experts_grouped_mm,
            package=cls.TARGET_PACKAGE,
        )
        replacement_counts += func_replacements

        # Initialize w13
        cls._change_existing_instances(model)

        # pyrefly: ignore [bad-return]
        return replacement_counts

    @classmethod
    def _change_existing_instances(cls, model: nn.Module):
        """Traverse the model and convert w1+w3 of the existing GroupedExperts into w13."""
        for name, module in model.named_modules():
            class_name = type(module).__name__
            if (
                "GroupedExperts" not in class_name
                and cls.TARGET_CLASS not in class_name
            ):
                continue
            w1 = getattr(module, "w1", None)
            w3 = getattr(module, "w3", None)

            if w1 is not None and w3 is not None:
                try:
                    cls._create_w13_from_w1_w3(module, name)
                except Exception as e:
                    logger.warning(f"Failed to convert {name}: {e}")
            else:
                logger.warning(f"  {name}: Missing w1/w3, skipping")
        return

    @classmethod
    def _create_w13_from_w1_w3(cls, module: nn.Module, module_name: str):
        """Create parameter w13 from w1"""
        w1 = module.w1

        # pyrefly: ignore [bad-index]
        num_experts = w1.shape[0]
        # pyrefly: ignore [bad-index]
        hidden_dim = w1.shape[1]
        # pyrefly: ignore [bad-index]
        dim = w1.shape[2]

        # pyrefly: ignore [no-matching-overload]
        w13_data = torch.empty(
            num_experts, hidden_dim * 2, dim, dtype=w1.dtype, device=w1.device
        )
        module.register_parameter("w13", nn.Parameter(w13_data))
        # pyrefly: ignore [bad-argument-type]
        module.use_grouped_mm = True

        # pyrefly: ignore [bad-argument-type]
        module.w1 = None
        # pyrefly: ignore [bad-argument-type]
        module.w3 = None

        # Add w13 initializer to _param_init if it exists (new torchtitan config system)
        _param_init = getattr(module, "_param_init", None)
        if _param_init is not None:
            # Use w1's initializer for w13 (combined w1+w3)
            w1_init = _param_init.get("w1")
            if w1_init is not None:
                _param_init["w13"] = w1_init

        logger.info(f"  {module_name}: Created w13 [{w13_data.shape}]")
