# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: fuse local MoE token permutation on NPU.

Only the ``EP=1`` path changes; expert-parallel dispatch remains unchanged.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


@dataclass(frozen=True, kw_only=True)
class NPULocalDispatchMetadata:
    input_shape: torch.Size
    permuted_indices: torch.Tensor
    topk_scores_experts_sorted_N: torch.Tensor  # noqa: N815


class NPUAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Use fused NPU permutation for local dispatch and matching metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def dispatch(  # pyrefly: ignore [bad-override]
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        if self.ep_mesh is not None:
            return super().dispatch(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        routed_input_RD, permuted_indices = torch_npu.npu_moe_token_permute(
            x_TD,
            topk_expert_ids_TK,
        )
        routed_scores_N1, _ = torch_npu.npu_moe_token_permute(
            topk_scores_TK.reshape(-1, 1),
            topk_expert_ids_TK.reshape(-1, 1),
        )
        metadata = NPULocalDispatchMetadata(
            input_shape=x_TD.shape,
            permuted_indices=permuted_indices,
            topk_scores_experts_sorted_N=routed_scores_N1.reshape(-1),
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        if not isinstance(metadata, NPULocalDispatchMetadata):
            return super().combine(
                routed_output_RD,
                metadata,
                x_TD,
                num_local_tokens_after_padding=num_local_tokens_after_padding,
                local_seq_len_after_padding=local_seq_len_after_padding,
            )

        # The local path does not use expert-parallel padding metadata.
        del num_local_tokens_after_padding, local_seq_len_after_padding

        # Match the upstream FP32 routing-score calculation.
        routed_output_RD = (
            routed_output_RD.to(torch.float32)
            * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
        ).to(routed_output_RD.dtype)
        out_ND = torch_npu.npu_moe_token_unpermute(
            routed_output_RD,
            metadata.permuted_indices,
            None,
        )
        return out_ND.view(
            metadata.input_shape[0],
            self.top_k,
            metadata.input_shape[1],
        ).sum(dim=1)


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description=(
        "Use torch_npu MoE token permute/unpermute for local (EP=1) token dispatch"
    ),
)
def npu_token_dispatcher_override(
    cfg: AllToAllTokenDispatcher.Config,
) -> NPUAllToAllTokenDispatcher.Config:
    return derive(cfg, NPUAllToAllTokenDispatcher.Config)
