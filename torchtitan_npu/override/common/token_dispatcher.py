# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NPU fused MoE token dispatcher.

Replaces the pure-PyTorch local token reordering (argsort + index_select) in
``AllToAllTokenDispatcher._local_reorder`` and the combine-time
score multiply + ``deterministic_scatter_add`` with the fused NPU kernels
``npu_moe_token_permute`` / ``npu_moe_token_unpermute``. For EP>1, the
rank-major -> expert-major reroute after all-to-all is replaced by
``npu_moe_re_routing``.

Semantics of the upstream metadata fields under this override:

- ``token_indices_experts_sorted_N`` stores the *inverse* permutation Q
  returned by ``npu_moe_token_permute`` (the double-argsorted forward
  permutation), NOT the forward argsort P of the upstream ``_local_reorder``.
  Q is exactly the ``sorted_indices`` input consumed by
  ``npu_moe_token_unpermute``.
- ``topk_scores_experts_sorted_N`` stores the original ``topk_scores_TK``
  ``(T, K)`` tensor, NOT the per-row gathered scores of the upstream
  ``_local_reorder``. ``npu_moe_token_unpermute`` expects ``probs`` shaped
  ``(T, K)``.
- ``routed_scores_R`` is populated when ``absorb_router_scores=True``. In that
  case scores are permuted alongside tokens, transported through EP, and
  consumed by grouped experts before the down projection; combine then skips
  the post-W2 multiply.

The EP=1 path no longer falls back to ``LocalTokenDispatcher``: dispatch is a
local permute and combine is a local unpermute, both fused into the NPU
kernels.

Precision: when pre-W2 absorption is disabled, ``npu_moe_token_unpermute``
multiplies the routing scores and sums over top-K internally in fp32
regardless of the input dtypes, and returns an output matching the first
argument's dtype. This is at least as precise as the upstream
fp32-score-multiply + bf16 ``scatter_add``.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import current_spmd_mesh, maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan_npu.ops.ascendc.moe_re_routing import npu_moe_re_routing
from torchtitan_npu.ops.ascendc.moe_token_unpermute import npu_moe_token_unpermute
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)


class AscAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher for EP>1 with NPU fused permute/unpermute kernels.

    Overrides only ``dispatch`` and ``combine``. The all-to-all exchange
    helpers (``_token_count_exchange``, ``_sync_token_count_exchange``,
    ``_dispatch_token_exchange``, ``_combine_token_exchange``) and
    ``_permute`` / ``_unpermute`` (rank-major <-> expert-major) are inherited
    from ``AllToAllTokenDispatcher``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, AllToAllDispatchMetadata | LocalDispatchMetadata]:
        """Reorder tokens, then all-to-all dispatch to expert-parallel ranks.

        ``npu_moe_token_permute`` replaces ``_local_reorder``: it fuses the
        argsort + ``index_select`` into one kernel and returns the inverse
        permutation Q instead of the forward argsort P.

        When ``ep_mesh`` is None (EP=1), returns the local permute result
        directly with ``LocalDispatchMetadata`` — no all-to-all.

        With SP, x_TD/topk_scores_TK/topk_expert_ids_TK are already
        the local SP shard (from DTensor Shard to_local via LocalMapConfig).

        Args:
            x_TD: ``(T, D)`` local token shard
            topk_scores_TK: ``(T, K)`` routing scores
            topk_expert_ids_TK: ``(T, K)`` expert indices
            num_local_tokens_per_expert_E: ``(E,)`` token counts for this local
                token shard

        Returns:
            routed_input_RD: ``[R = sum(num_tokens_per_local_expert_e), input_dim(D)]``.
                Tokens in expert-major order for local experts.
            num_tokens_per_local_expert_e: ``(num_local_experts,)`` token counts
            metadata: dispatch metadata for combine()
        """
        # Fused NPU permute: replaces _local_reorder (argsort + index_select).
        # sorted_indices_N is the inverse permutation Q; npu_moe_token_unpermute
        # consumes Q directly in combine.
        routed_input_ND, sorted_indices_N = torch_npu.npu_moe_token_permute(
            x_TD,
            topk_expert_ids_TK,
        )

        if self.absorb_router_scores:
            routed_scores_ND, _ = torch_npu.npu_moe_token_permute(
                topk_scores_TK.reshape(-1, 1),
                topk_expert_ids_TK.reshape(-1, 1),
            )
            routed_scores_ND = routed_scores_ND.reshape(-1)
        else:
            routed_scores_ND = None

        # EP=1: local dispatch — no all-to-all needed.
        if self.ep_mesh is None:
            return (
                routed_input_ND,
                num_local_tokens_per_expert_E,
                LocalDispatchMetadata(
                    token_indices_experts_sorted_N=sorted_indices_N,
                    topk_scores_experts_sorted_N=topk_scores_TK,
                    routed_scores_R=routed_scores_ND,
                ),
            )

        ep_size = self.ep_mesh.size()

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():  # sparse mesh reinterpret
            for axis in ["dp", "cp", "tp"]:
                spmd.mutate_type(num_local_tokens_per_expert_E, axis, src=spmd.P, dst=spmd.V)

        # generate the input splits and output splits for all-to-all
        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                num_local_tokens_per_expert_E = spmd.reinterpret_mesh(
                    num_local_tokens_per_expert_E, spmd.current_mesh()
                )
                routed_input_ND = spmd.reinterpret_mesh(routed_input_ND, spmd.current_mesh())
                if routed_scores_ND is not None:
                    routed_scores_ND = spmd.reinterpret_mesh(routed_scores_ND, spmd.current_mesh())

            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E,
                    pg,
                    ep_size,
                )
                (
                    num_global_tokens_per_local_expert_E,
                    input_splits_list,
                    output_splits_list,
                ) = self._sync_token_count_exchange(
                    num_local_tokens_per_expert_E,
                    num_global_tokens_per_local_expert_EP_e,
                    ep_size,
                )

            routed_input_RD = self._dispatch_token_exchange(
                routed_input_ND,
                pg,
                output_splits_list,
                input_splits_list,
            )
            routed_scores_rank_major_R = (
                self._dispatch_token_exchange(
                    routed_scores_ND.reshape(-1, 1),
                    pg,
                    output_splits_list,
                    input_splits_list,
                ).reshape(-1)
                if routed_scores_ND is not None
                else None
            )
            # Fused CANN reroute replaces inherited _permute. The returned
            # restore indices are consumed by the matching unpermute op in
            # combine and by the reroute autograd formula.
            expert_token_num_per_rank = num_global_tokens_per_local_expert_E.view(
                ep_size,
                -1,
            )
            (
                routed_input_RD,
                permuted_scales,
                restore_indices,
                num_global_tokens_per_local_expert_e,
            ) = npu_moe_re_routing(
                routed_input_RD,
                expert_token_num_per_rank,
                per_token_scales=routed_scores_rank_major_R,
            )
            routed_scores_R = permuted_scales if routed_scores_ND is not None else None

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=sorted_indices_N,
            topk_scores_experts_sorted_N=topk_scores_TK,
            input_shape=routed_input_RD.shape,
            permuted_indices=restore_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            routed_scores_R=routed_scores_R,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + fused unpermute.

        ``npu_moe_token_unpermute`` replaces the upstream score multiply +
        ``deterministic_scatter_add``: the kernel scatters each routed row
        back to its original token, multiplies by the routing score, and sums
        over top-K, accumulating internally in fp32. The output dtype follows
        the first argument.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens

        Returns:
            out_TD: ``(T, D)`` combined output.
        """
        probs = (
            torch.ones_like(metadata.topk_scores_experts_sorted_N)
            if self.absorb_router_scores and metadata.routed_scores_R is not None
            else metadata.topk_scores_experts_sorted_N
        )

        # EP=1: fused NPU unpermute — no all-to-all to reverse.
        if self.ep_mesh is None:
            return npu_moe_token_unpermute(
                routed_output_RD,
                metadata.token_indices_experts_sorted_N,
                probs=probs,
            )

        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            routed_output_RD = npu_moe_token_unpermute(
                routed_output_RD,
                metadata.permuted_indices,
                probs=None,
            )
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD,
                pg,
                metadata.input_splits,
                metadata.output_splits,
            )

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            # dense mesh reinterpret
            routed_output_RD = spmd.reinterpret_mesh(routed_output_RD, spmd.current_mesh())

        # Fused NPU unpermute: scatter + score x probs + sum over top-K.
        return npu_moe_token_unpermute(
            routed_output_RD,
            metadata.token_indices_experts_sorted_N,
            probs=probs,
        )


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="NPU fused npu_moe_token_permute/unpermute for MoE dispatch/combine",
)
def asc(
    cfg: AllToAllTokenDispatcher.Config,
) -> AscAllToAllTokenDispatcher.Config:
    return derive(cfg, AscAllToAllTokenDispatcher.Config)
