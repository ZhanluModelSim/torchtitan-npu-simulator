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
from torchtitan.models.common.moe import GroupedExperts

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
    StateDictUpdater,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.tools.weight_utils import _split_w13_for_mapping, fuse_experts

logger = logging.getLogger(__name__)

# Calculate the number of experts and EP degree, which are used as parameters
# when invoking operators during Hifloat8 low-precision training.
group_size_params = {
    "num_experts": None,
    "expert_model_parallel_size": None,
    "g_size": None,
}


def _run_experts_grouped_mm(
    w13: torch.Tensor | None,
    w2: torch.Tensor,
    _w3: torch.Tensor | None,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)
    if w13 is None:
        raise ValueError("w13 cannot be None for grouped_mm experts")
    h = torch._grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offsets)
    # DSv4 injects ``swiglu_limit`` on its GroupedExperts to bound gate/up before
    # the swiglu activation. Without this clamp the bf16 grouped_mm output can
    # overflow during the first optimizer step under sqrtsoftplus routing and
    # produce NaNs.
    if swiglu_limit is not None:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
    h = torch_npu.npu_swiglu(h, dim=-1)
    if routed_scores is not None:
        h = h * routed_scores.to(h.dtype)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


def npu_grouped_experts_forward(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    routed_scores: torch.Tensor | None = None,
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
        # One-time diagnostic. Skip entirely under torch.compile: dynamo cannot
        # trace setattr on a function object, and this branch has no effect on
        # the computation.
        logged_attr = "_logged"
        if not torch.compiler.is_compiling() and not hasattr(npu_grouped_experts_forward, logged_attr):
            setattr(npu_grouped_experts_forward, logged_attr, True)
            logger.info(
                f"[GMM-TP] w2 placements={self.w2.placements}, is_tp={is_tp}, "
                f"w2 local shape={w2.shape}, w13 local shape={w13.shape if w13 is not None else None}"
            )
    else:
        w2 = self.w2
        w13 = self.w13
        tp_group = None

    # DSv4 sets ``self.swiglu_limit`` on its GroupedExperts instance (see
    # torchtitan_npu/models/deepseek_v4/moe.py). Other models leave this unset
    # and the clamp is skipped in ``_run_experts_grouped_mm``.
    swiglu_limit = getattr(self, "swiglu_limit", None)

    # Tensor subclasses (for example MXFP8 training weights) already own their
    # autograd behavior. Keep those paths untouched without depending on a
    # converter-specific class name or version.
    plain_weight_types = (torch.Tensor, nn.Parameter)
    needs_meta_autograd = (
        x.device.type == "meta"
        and type(w2) in plain_weight_types
        and type(w13) in plain_weight_types
    )
    if needs_meta_autograd:
        from torchtitan_npu.simulator.hardware_shims.grouped_experts_shim import (
            run_meta_grouped_experts,
        )

        out = run_meta_grouped_experts(
            _run_experts_grouped_mm,
            w13,
            w2,
            x,
            num_tokens_per_expert,
            swiglu_limit,
            routed_scores,
        )
    else:
        out = _run_experts_grouped_mm(
            w13,
            w2,
            None,
            x,
            num_tokens_per_expert,
            swiglu_limit,
            routed_scores,
        )

    if is_tp and tp_group is not None:
        import torch.distributed as dist

        # One-time diagnostic of the TP all-reduce. Skip under torch.compile
        # (dynamo cannot trace setattr on a function) and avoid the .item()
        # device syncs on every step once it has been logged.
        ar_logged_attr = "_ar_logged"
        log_ar = not torch.compiler.is_compiling() and not hasattr(npu_grouped_experts_forward, ar_logged_attr)
        pre_ar = out.mean().item() if log_ar else None

        dist.all_reduce(out, group=tp_group)

        if log_ar:
            setattr(npu_grouped_experts_forward, ar_logged_attr, True)
            post_ar = out.mean().item()
            ratio = (
                post_ar / pre_ar  # pyrefly: ignore [unsupported-operation]
                if pre_ar != 0
                else float("inf")
            )
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


class NpuGroupedExperts(GroupedExperts):
    def __init__(
        self,
        parent: GroupedExperts,
    ):
        self.__dict__.update(parent.__dict__)
        self.use_grouped_mm = True
        if self.w1 is not None and self.w3 is not None:
            w13_data = torch.empty(
                self.num_experts,
                self.w2.shape[2] * 2,
                self.w2.shape[1],
                dtype=self.w1.dtype,
                device=self.w1.device,
            )
            self.w13 = nn.Parameter(w13_data)
            # Add w13 initializer to _param_init if it exists (new torchtitan config system)
            _param_init = getattr(parent, "_param_init", None)
            if _param_init is not None:
                # Use w1's initializer for w13 (combined w1+w3)
                w1_init = _param_init.get("w1")
                if w1_init is not None:
                    _param_init["w13"] = w1_init

            # pyrefly: ignore [bad-assignment]
            self.w1 = None
            # pyrefly: ignore [bad-assignment]
            self.w3 = None
            # pyrefly: ignore [bad-assignment]
            parent.w1 = None
            # pyrefly: ignore [bad-assignment]
            parent.w3 = None
            logger.info(f"  NpuGroupedExperts: Created w13 [{w13_data.shape}]")

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        routed_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return npu_grouped_experts_forward(self, x, num_tokens_per_expert, routed_scores)

    def init_weights(self, init_std: float):
        npu_grouped_experts_init_weights(self, init_std)


class NpuGroupedExpertConverter(ModelCustomConverter):
    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if not isinstance(module, GroupedExperts):
                continue
            replace_module_with_name(model, name, NpuGroupedExperts(module))


class GMMStateDictUpdater(StateDictUpdater):
    @classmethod
    def to_hf(cls, state_dict):
        has_w13 = any(".moe.experts.w13" in k for k in state_dict)
        if has_w13:
            state_dict = _split_w13_for_mapping(state_dict)
        return state_dict

    @classmethod
    def from_hf(cls, state_dict):
        keys_to_remove = [k for k in state_dict if k.endswith(".weight_scale_inv")]
        for k in keys_to_remove:
            del state_dict[k]

        return fuse_experts(state_dict)


@register_model_converter("npu_gmm")
class GMMModelConfig(ModelCustomConfig):
    model_converter = NpuGroupedExpertConverter
    state_dict_updater = GMMStateDictUpdater
