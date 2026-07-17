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
from collections.abc import Callable

import torch
import torch_npu
from torch import nn
from torch._functorch.partitioners import default_partition
from torch._inductor.custom_graph_pass import CustomPartitionerFn
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

_GROUPED_MM_INT32_OFFSET_MAX = torch.iinfo(torch.int32).max
_ExpertActivationFn = Callable[[torch.Tensor, float | None, torch.Tensor | None], torch.Tensor]
_DYNAMO_ATTR = "_dynamo"
_MAYBE_MARK_DYNAMIC_ATTR = "maybe_mark_dynamic"
_EXPERT_ACTIVATION_FN_ATTR = "_expert_activation_fn"
_EXPERT_ACTIVATION_COMPILE_KEY_ATTR = "_expert_activation_compile_key"
_LOGGED_ATTR = "_logged"
_AR_LOGGED_ATTR = "_ar_logged"

# Calculate the number of experts and EP degree, which are used as parameters
# when invoking operators during Hifloat8 low-precision training.
group_size_params = {
    "num_experts": None,
    "expert_model_parallel_size": None,
    "g_size": None,
}


class _NpuGmmAotDefaultPartitioner(CustomPartitionerFn):
    def __call__(
        self,
        gm,
        joint_inputs,
        *,
        compiler=None,
        static_lifetime_input_indices=None,
        **kwargs,
    ):
        # AOTAutograd's default partition keeps clamp-gradient masks in backward;
        # Inductor's min-cut partitioner may pull them into the forward graph.
        return default_partition(
            gm,
            joint_inputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
            **kwargs,
        )

    def uuid(self):
        # Separate compile-cache entries from graphs built with another partitioner.
        return "npu_expert_activation_default_partition"


def _validate_grouped_mm_offsets_int32_range(x: torch.Tensor) -> None:
    num_routed_tokens = x.shape[0]
    if num_routed_tokens > _GROUPED_MM_INT32_OFFSET_MAX:
        raise ValueError(
            "npu_gmm requires int32 grouped_mm offsets; "
            f"this rank has {num_routed_tokens} routed rows in one GMM call, "
            f"above the int32 limit {_GROUPED_MM_INT32_OFFSET_MAX}. "
            "Reduce the per-rank MoE workload."
        )


def _expert_activation(
    h: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
) -> torch.Tensor:
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
    return h


def _compile_expert_activation(*, backend: str, dynamic_tokens: bool) -> _ExpertActivationFn:
    compiled_activation = torch.compile(
        _expert_activation,
        backend=backend,
        fullgraph=True,
        options={"custom_partitioner_fn": _NpuGmmAotDefaultPartitioner()},
    )
    maybe_mark_dynamic = getattr(getattr(torch, _DYNAMO_ATTR), _MAYBE_MARK_DYNAMIC_ATTR)

    def run_compiled_activation(
        h: torch.Tensor,
        swiglu_limit: float | None = None,
        routed_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dynamic_tokens:
            maybe_mark_dynamic(h, 0)
            if routed_scores is not None:
                maybe_mark_dynamic(routed_scores, 0)
        return compiled_activation(h, swiglu_limit, routed_scores)

    return run_compiled_activation


def _run_experts_grouped_mm(
    w13: torch.Tensor | None,
    w2: torch.Tensor,
    _w3: torch.Tensor | None,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None = None,
    routed_scores: torch.Tensor | None = None,
    *,
    activation_fn: _ExpertActivationFn = _expert_activation,
) -> torch.Tensor:
    _validate_grouped_mm_offsets_int32_range(x)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    if w13 is None:
        raise ValueError("w13 cannot be None for grouped_mm experts")
    h = torch._grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offsets)
    h = activation_fn(h, swiglu_limit, routed_scores)
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


def _all_reduce_tp_output(out: torch.Tensor, tp_group) -> None:
    import torch.distributed as dist

    log_ar = not torch.compiler.is_compiling() and not getattr(npu_grouped_experts_forward, _AR_LOGGED_ATTR, False)
    pre_ar = out.mean().item() if log_ar else 0.0
    dist.all_reduce(out, group=tp_group)
    if log_ar:
        setattr(npu_grouped_experts_forward, _AR_LOGGED_ATTR, True)
        post_ar = out.mean().item()
        ratio = post_ar / pre_ar if pre_ar != 0 else float("inf")
        logger.info(
            "[GMM-TP] all-reduce: pre_mean=%.6f, post_mean=%.6f, ratio=%s",
            pre_ar,
            post_ar,
            ratio,
        )


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
        # Skip the one-time diagnostic while tracing; setattr on a function is not traceable.
        if not torch.compiler.is_compiling() and not getattr(
            npu_grouped_experts_forward,
            _LOGGED_ATTR,
            False,
        ):
            setattr(npu_grouped_experts_forward, _LOGGED_ATTR, True)
            logger.info(
                "[GMM-TP] w2 placements=%s, is_tp=%s, w2 local shape=%s, w13 local shape=%s",
                self.w2.placements,
                is_tp,
                w2.shape,
                w13.shape if w13 is not None else None,
            )
    else:
        w2 = self.w2
        w13 = self.w13
        tp_group = None

    # DSv4 sets ``self.swiglu_limit`` on its GroupedExperts instance (see
    # torchtitan_npu/models/deepseek_v4/moe.py). Other models leave this unset
    # and the clamp is skipped in ``_expert_activation``.
    swiglu_limit = getattr(self, "swiglu_limit", None)

    out = _run_experts_grouped_mm(
        w13,
        w2,
        None,
        x,
        num_tokens_per_expert,
        swiglu_limit,
        routed_scores,
        activation_fn=getattr(self, _EXPERT_ACTIVATION_FN_ATTR),
    )

    if is_tp and tp_group is not None:
        _all_reduce_tp_output(out, tp_group)

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
        self._expert_activation_fn: _ExpertActivationFn = _expert_activation
        self._expert_activation_compile_key: tuple[str, bool] | None = None
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


def compile_expert_activation(
    model: nn.Module,
    *,
    backend: str,
    dynamic_tokens: bool,
) -> None:
    """Compile the activation bridge for grouped experts in ``model``."""
    experts = [module for module in model.modules() if isinstance(module, NpuGroupedExperts)]
    compile_key = (backend, dynamic_tokens)
    if not experts or all(getattr(module, _EXPERT_ACTIVATION_COMPILE_KEY_ATTR) == compile_key for module in experts):
        return

    compiled_activation = _compile_expert_activation(
        backend=backend,
        dynamic_tokens=dynamic_tokens,
    )
    for module in experts:
        setattr(module, _EXPERT_ACTIVATION_FN_ATTR, compiled_activation)
        setattr(module, _EXPERT_ACTIVATION_COMPILE_KEY_ATTR, compile_key)


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
