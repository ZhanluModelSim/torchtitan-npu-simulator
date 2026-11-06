# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: run DeepSeek-V4 score TopK with a compile-safe TileLang op."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router
from torchtitan_npu.ops.tilelang import tilelang_topk_gate


class TilelangRouter(TokenChoiceTopKRouter):
    """DeepSeek-V4 router using TileKernels TopK on non-hash layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def _select_experts(self, scores_TE: torch.Tensor, expert_bias_E=None, **router_kwargs) -> torch.Tensor:
        return _tilelang_select_experts(self, scores_TE, expert_bias_E)


def _tilelang_select_experts(router: TokenChoiceTopKRouter, scores_TE: torch.Tensor, expert_bias_E) -> torch.Tensor:
    scores_for_choice_TE = scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
    return _tilelang_topk(scores_for_choice_TE, router.top_k)


def _tilelang_topk(scores_for_choice_TE: torch.Tensor, top_k: int) -> torch.Tensor:
    scores_for_choice = (
        scores_for_choice_TE.to_local() if isinstance(scores_for_choice_TE, DTensor) else scores_for_choice_TE
    )
    if scores_for_choice.device.type != "npu":
        raise ValueError("TileKernels topk_gate requires NPU inputs.")
    if scores_for_choice.dtype != torch.float32:
        raise ValueError("TileKernels topk_gate requires float32 scores.")

    num_experts = scores_for_choice.size(-1)
    flat_scores = scores_for_choice.reshape(-1, num_experts).contiguous()
    flat_indices = tilelang_topk_gate(flat_scores, top_k)
    indices_TK = flat_indices.view(*scores_for_choice.shape[:-1], top_k)
    if isinstance(scores_for_choice_TE, DTensor):
        return DTensor.from_local(
            indices_TK, scores_for_choice_TE.device_mesh, scores_for_choice_TE.placements, run_check=False
        )
    return indices_TK


class TilelangV4Router(DeepSeekV4Router):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Router.Config):
        pass

    def _select_experts(self, scores_TE: torch.Tensor, expert_bias_E=None, *, input_ids_T=None, **router_kwargs):
        if self.hash:
            return super()._select_experts(scores_TE, expert_bias_E, input_ids_T=input_ids_T, **router_kwargs)
        return _tilelang_select_experts(self, scores_TE, expert_bias_E)


class TilelangV41Router(DeepSeekV41Router):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Router.Config):
        pass

    def _topk(self, scores_for_choice_TE: torch.Tensor) -> torch.Tensor:
        return _tilelang_topk(scores_for_choice_TE, self.top_k)


@override(target=TokenChoiceTopKRouter.Config, exact=True, description="Dynamic routing via TileKernels topk_gate")
def tilelang(cfg: TokenChoiceTopKRouter.Config) -> TilelangRouter.Config:
    return derive(cfg, TilelangRouter.Config)


@override(target=DeepSeekV4Router.Config, exact=True, description="DeepSeek-V4 routing via TileKernels topk_gate")
def tilelang_v4(cfg: DeepSeekV4Router.Config) -> TilelangV4Router.Config:
    return derive(cfg, TilelangV4Router.Config)


@override(target=DeepSeekV41Router.Config, exact=True, description="DeepSeek-V4.1 routing via TileKernels topk_gate")
def tilelang_v41(cfg: DeepSeekV41Router.Config) -> TilelangV41Router.Config:
    return derive(cfg, TilelangV41Router.Config)
