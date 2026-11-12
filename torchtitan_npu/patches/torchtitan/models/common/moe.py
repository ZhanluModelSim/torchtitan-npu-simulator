# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5543

"""Preserve pre-W2 router scores and SwiGLU clamping on typed expert linears."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_remat as remat
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.models.common.moe import RoutedExperts

from torchtitan_npu.models.common.activation import LimitedSwiGLU


class _RouterScoreAbsorbingRoutedExperts(RoutedExperts):
    """Use dispatcher-aligned scores on BF16 activations before the down projection."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask_T
        routed_input_rd, tokens_per_expert_e, metadata = self.token_dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        offsets_e = torch.cumsum(tokens_per_expert_e, dim=0, dtype=torch.int32)
        with maybe_set_sparse_mesh():
            remat.recompute_needs_tensor(routed_input_rd)
            gate_up_r2f = self.w13(routed_input_rd.bfloat16(), offsets_e)
            remat.recompute_needs_tensor(gate_up_r2f)
            hidden_rf = self.activation_fn(gate_up_r2f, offsets=offsets_e)
            routed_scores_r = getattr(metadata, "routed_scores_R", None)
            if routed_scores_r is not None:
                hidden_rf = (hidden_rf.float() * routed_scores_r.float().reshape(-1, 1)).to(hidden_rf.dtype)
            routed_output_rd = self.w2(hidden_rf, offsets_e)
            remat.recompute_needs_tensor(routed_output_rd)
            routed_output_rd = routed_output_rd.type_as(routed_input_rd)
            if self.output_postprocess is not None:
                routed_output_rd = self.output_postprocess(routed_output_rd)
        return self.token_dispatcher.combine(routed_output_rd, metadata, x_TD)


def _clamp_make_routed_experts_config(
    *,
    swiglu_limit: float | None = None,
    absorb_router_scores: bool = True,
    **kwargs,
):
    config = _original_make_routed_experts_config(**kwargs)
    dispatcher = config.token_dispatcher
    if "absorb_router_scores" in {field.name for field in dataclasses.fields(dispatcher)}:
        dispatcher = dataclasses.replace(dispatcher, absorb_router_scores=absorb_router_scores)
    elif absorb_router_scores:
        raise ValueError(f"Pre-W2 router scores are unsupported by dispatcher {type(dispatcher).__qualname__}")
    if swiglu_limit is not None:
        config = dataclasses.replace(config, activation_fn=LimitedSwiGLU.Config(swiglu_limit=swiglu_limit))
    return dataclasses.replace(config, token_dispatcher=dispatcher)


def _clamp_make_ffn_config(*, swiglu_limit: float | None = None, **kwargs):
    config = _original_make_ffn_config(**kwargs)
    if swiglu_limit is None:
        return config
    return dataclasses.replace(config, activation_fn=LimitedSwiGLU.Config(swiglu_limit=swiglu_limit))


_original_make_routed_experts_config: Callable[..., Any]
_original_make_ffn_config: Callable[..., Any]


def apply() -> None:
    import torchtitan.models.common.config_utils
    import torchtitan.models.common.moe
    import torchtitan.models.common.moe.moe

    torchtitan.models.common.moe.moe.RoutedExperts = _RouterScoreAbsorbingRoutedExperts
    torchtitan.models.common.moe.RoutedExperts = _RouterScoreAbsorbingRoutedExperts
    torchtitan.models.common.config_utils.RoutedExperts = _RouterScoreAbsorbingRoutedExperts
    global _original_make_routed_experts_config, _original_make_ffn_config
    _original_make_routed_experts_config = torchtitan.models.common.config_utils.make_routed_experts_config
    _original_make_ffn_config = torchtitan.models.common.config_utils.make_ffn_config
    torchtitan.models.common.config_utils.make_routed_experts_config = _clamp_make_routed_experts_config
    torchtitan.models.common.config_utils.make_ffn_config = _clamp_make_ffn_config


apply()
