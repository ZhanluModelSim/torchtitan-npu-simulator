# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash feed-forward blocks built on the upstream MoE machinery.

``GlmMoE`` extends the shared ``MoE`` module (router → RoutedExperts
(dispatcher + inner experts) → shared experts) so the NPU overrides apply
unchanged: ``override.common.token_dispatcher.asc`` swaps in the fused
``npu_moe_token_permute``/``unpermute`` dispatcher and
``override.common.swiglu_group.asc`` swaps in ``cann_ops_nn.swiglu_group``
for the routed experts — both keyed on the upstream
``*.moe.routed_experts.inner_experts`` FQN. EP support comes from the same
``AllToAllTokenDispatcher`` (the NPU variant keeps the fused kernels for the
EP=1 local branch).

Expert weight layouts follow the torchtitan grouped-GEMM convention:
``w1_EFD``/``w3_EFD`` [E, I, H] (gate/up) and ``w2_EDF`` [E, H, I] (down),
which maps 1:1 onto the reference ``gate_up_proj [E, 2*I, H]`` /
``down_proj [E, H, I]`` (gate_up = cat(w1, w3)).
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import _grouped_mm, nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.config_utils import make_token_dispatcher_config
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)

from .common import WEIGHT_INIT, linear_config


def clamp_swiglu_eager(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Eager clamp-swiglu: fold the per-half clamp bounds then silu(gate)*up."""
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class GlmMLP(FeedForward):
    """Dense MLP with the clamped swiglu activation.

    ``swiglu_limit`` lives on the module so
    ``override.common.swiglu_group.asc_shared_experts`` (and any dense-FFN
    swiglu_group user) can read it; without that override the eager forward
    clamps explicitly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        dim: int = 4096
        hidden_dim: int = 12288
        swiglu_limit: float = 10.0
        w1: Linear.Config | None = None
        w2: Linear.Config | None = None
        w3: Linear.Config | None = None

        def __post_init__(self) -> None:
            # Guard on None: build() re-runs __post_init__ on dataclasses.replace(self),
            # which would swap in fresh configs and drop sharding stamps set on them.
            if self.w1 is None:
                self.w1 = linear_config(self.dim, self.hidden_dim)
            if self.w2 is None:
                self.w2 = linear_config(self.hidden_dim, self.dim)
            if self.w3 is None:
                self.w3 = linear_config(self.dim, self.hidden_dim)

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((self.w1(x), self.w3(x)), dim=-1)
        return self.w2(clamp_swiglu_eager(packed, self.swiglu_limit))


class TopkRouter(TokenChoiceTopKRouter):
    """Sigmoid + e_score_correction_bias router with n_group=1 (the group
    gate degenerates).

    Conforms to the upstream ``TokenChoiceTopKRouter`` contract so the
    shared ``MoE`` wrapper can drive it: returns
    ``(topk_scores_BLK, topk_expert_ids_BLK, scores_BLE)`` with the batch
    dims preserved. The GLM correction bias is a static checkpoint buffer,
    not the auxiliary-loss-free dynamic bias, so the ``expert_bias_E``
    argument is ignored and ``GlmMoE`` disables the aux-free hook
    (``load_balance_coeff=None``)."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        dim: int = 4096
        n_group: int = 1
        topk_group: int = 1
        routed_scaling_factor: float = 2.5
        norm_topk_prob: bool = True
        debug_force_load_balance: bool = False
        gate: Linear.Config | None = None

        def __post_init__(self) -> None:
            if self.gate is None:
                self.gate = Linear.Config(
                    in_features=self.dim,
                    out_features=self.num_experts,
                    bias=False,
                    param_init={"weight": nn.init.zeros_},
                )
            self.score_func = "sigmoid"
            self.route_norm = self.norm_topk_prob
            self.route_scale = self.routed_scaling_factor
            self._debug_force_load_balance = self.debug_force_load_balance

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.hidden_dim = config.dim
        self.debug_force_load_balance = config.debug_force_load_balance
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_experts))

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.e_score_correction_bias.device
        self.e_score_correction_bias = torch.zeros(self.num_experts, device=device)

    def _node_limited_scores(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        """Group-limited routing (MindSpeed noaux_tc n_group path)."""
        b, l, e = scores_for_choice.shape
        group_scores = scores_for_choice.view(b, l, self.num_group, e // self.num_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, -1, e // self.num_group).reshape(b, l, e)
        return torch.where(score_mask.bool(), scores_for_choice, float("-inf"))

    def forward(
        self,
        x_BLD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # GLM routing is score-only: the patched MoE threads ``input_ids`` /
        # ``image_mask`` for the DSV4 hash router; both stay unused here.
        del expert_bias_E, input_ids, image_mask

        if x_BLD.device.type == "meta":
            # Meta capture cannot enter autocast; the fp32 gate math is
            # shape-relevant only, so a plain upcast suffices.
            scores_BLE = self.gate(x_BLD).float()
        else:
            with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
                scores_BLE = self.gate(x_BLD)
        scores_BLE = torch.sigmoid(scores_BLE)

        scores_for_choice_BLE = scores_BLE + self.e_score_correction_bias
        if self.num_group > 1:
            scores_for_choice_BLE = self._node_limited_scores(scores_for_choice_BLE)

        if self._debug_force_load_balance:
            bs, slen, _ = scores_BLE.shape
            topk_expert_ids_BLK = (
                torch.arange(bs * slen * self.top_k, device=scores_BLE.device, dtype=torch.int64).reshape(
                    bs, slen, self.top_k
                )
                % self.num_experts
            )
        else:
            _, topk_expert_ids_BLK = torch.topk(scores_for_choice_BLE, k=self.top_k, dim=-1, sorted=False)
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        if self.route_norm:
            denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denominator
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        )


class GlmExperts(GroupedExperts):
    """Routed experts on the upstream grouped-GEMM contract.

    ``forward(x_RD, num_tokens_per_expert_E)`` runs the two grouped GEMMs
    around the clamped swiglu; the routing scores multiply at the dispatcher
    combine, not here (``routed_scores_R`` stays unused, matching the NPU
    ``swiglu_group`` override which receives them only when the dispatcher
    absorbs the scores). With ``override.common.swiglu_group.asc`` enabled
    this module is replaced by the fused ``cann_ops_nn.swiglu_group`` expert
    at config-build time; this body is the eager/decomposed fallback.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 10.0

        def __post_init__(self) -> None:
            self.param_init = dict.fromkeys(("w1_EFD", "w2_EDF", "w3_EFD"), WEIGHT_INIT)

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def _local_weight(self, name: str) -> torch.Tensor:
        weight = getattr(self, name)
        if isinstance(weight, DTensor):
            return weight.to_local()
        return weight

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        *,
        routed_scores_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del routed_scores_R  # scores multiply at the dispatcher combine

        w1 = self._local_weight("w1_EFD")
        w2 = self._local_weight("w2_EDF")
        w3 = self._local_weight("w3_EFD")

        offsets = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets, axis, src=spmd.P, dst=spmd.V)

        gate = _grouped_mm(x_RD.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
        up = _grouped_mm(x_RD.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
        hidden = clamp_swiglu_eager(torch.cat((gate, up), dim=-1), self.swiglu_limit)
        return _grouped_mm(hidden, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x_RD)


class GlmMoE(MoE):
    """Routed MoE + shared expert (reference Glm5NextTextMoE) on the shared
    ``MoE`` wrapper. The aux-free load-balancing hook stays off: GLM routing
    uses the static ``e_score_correction_bias``."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        dim: int = 4096
        top_k: int = 8
        moe_intermediate_size: int = 2048
        n_shared_experts: int = 1
        swiglu_limit: float = 10.0
        routed_scaling_factor: float = 2.5
        norm_topk_prob: bool = True
        n_group: int = 1
        topk_group: int = 1
        debug_force_load_balance: bool = False
        routed_experts: RoutedExperts.Config | None = None
        router: TopkRouter.Config | None = None
        shared_experts: GlmMLP.Config | None = None

        def __post_init__(self) -> None:
            if self.routed_experts is None:
                self.routed_experts = RoutedExperts.Config(
                    inner_experts=GlmExperts.Config(
                        dim=self.dim,
                        hidden_dim=self.moe_intermediate_size,
                        num_experts=self.num_experts,
                        swiglu_limit=self.swiglu_limit,
                    ),
                    token_dispatcher=make_token_dispatcher_config(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        comm_backend="standard",
                    ),
                )
            if self.router is None:
                self.router = TopkRouter.Config(
                    dim=self.dim,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                    routed_scaling_factor=self.routed_scaling_factor,
                    norm_topk_prob=self.norm_topk_prob,
                    debug_force_load_balance=self.debug_force_load_balance,
                )
            if self.shared_experts is None:
                self.shared_experts = GlmMLP.Config(
                    dim=self.dim,
                    hidden_dim=self.moe_intermediate_size * self.n_shared_experts,
                    swiglu_limit=self.swiglu_limit,
                )
            self.load_balance_coeff = None
