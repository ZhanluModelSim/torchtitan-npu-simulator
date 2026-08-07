# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

"""Backport router keyword forwarding from TorchTitan PR #3634.

The forward body retains this checkout's ``routed_experts`` naming. Remove the
backport after the TorchTitan dependency includes the PR.
"""

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

__all__ = ["RouterKwargsMoE"]


class RouterKwargsMoE(MoE):
    """Forward router-specific keyword arguments through ``MoE``."""

    def forward(self, x_BLD: torch.Tensor, **router_kwargs) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        sp_size = getattr(self.routed_experts.token_dispatcher, "sp_size", 1)
        if not isinstance(x_BLD, DTensor) and self.seq_dim_tp_sharded:
            seq_pad = 0
            seq_dim_pad_tokens = 0
            num_local_tokens_after_seq_dim_padding = B * L
        else:
            seq_pad = sp_size - L if sp_size > L else 0
            if seq_pad:
                x_BLD = F.pad(x_BLD, (0, 0, 0, seq_pad))
                L = L + seq_pad
            seq_dim_pad_tokens = (-L) % sp_size
            local_batch_size = (
                x_BLD._local_tensor.shape[0] if isinstance(x_BLD, DTensor) else B
            )
            num_local_tokens_after_seq_dim_padding = (
                local_batch_size * (L + seq_dim_pad_tokens) // sp_size
            )

        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(x_BLD, self.expert_bias_E, **router_kwargs)

        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_BLK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        with torch.no_grad():
            self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_BLD = self.routed_experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
            num_local_tokens_after_seq_dim_padding=(
                num_local_tokens_after_seq_dim_padding
            ),
        )

        shared_out_BLD = (
            self.shared_experts(x_BLD) if self.shared_experts is not None else None
        )

        if (
            isinstance(self.routed_experts.token_dispatcher, DeepEPTokenDispatcher)
            and self.routed_experts.token_dispatcher.sp_size == 1
        ):
            from torchtitan.distributed.deepep.deepep import sync_combine

            sync_combine()

        if seq_dim_pad_tokens:
            out_BLD = out_BLD[:, :L, :]

        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD

        if seq_pad:
            out_BLD = out_BLD[:, : L - seq_pad, :]

        return out_BLD
