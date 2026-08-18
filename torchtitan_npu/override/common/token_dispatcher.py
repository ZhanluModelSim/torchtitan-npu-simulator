# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NPU fused MoE token dispatcher.

Replaces the pure-PyTorch local token reordering (argsort + index_select) in
``AllToAllTokenDispatcher._local_reorder`` and the combine-time
score multiply + ``deterministic_scatter_add`` with the fused NPU kernels
``npu_moe_token_permute`` / ``npu_moe_token_unpermute``.

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

The EP=1 path no longer falls back to ``LocalTokenDispatcher``: dispatch is a
local permute and combine is a local unpermute, both fused into the NPU
kernels.

Precision: ``npu_moe_token_unpermute`` multiplies the routing scores and
sums over top-K internally in fp32 regardless of the input dtypes, and
returns an output matching the first argument's dtype. This is at least as
precise as the upstream fp32-score-multiply + bf16 ``scatter_add``.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.distributed.spmd_types import current_spmd_mesh, maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
)


class NPUAllToAllTokenDispatcher(AllToAllTokenDispatcher):
    """Token dispatcher for EP>1 with NPU fused permute/unpermute kernels.

    Overrides only ``dispatch`` and ``combine``. The all-to-all exchange
    helpers (``_token_count_exchange``, ``_sync_token_count_exchange``,
    ``_dispatch_token_exchange``, ``_combine_token_exchange``),
    ``_permute`` / ``_unpermute`` (rank-major <-> expert-major), and the
    SP coordinate helpers are inherited from ``AllToAllTokenDispatcher``.
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
        routed_input_ND, sorted_indices_N = torch_npu.npu_moe_token_permute(x_TD, topk_expert_ids_TK)

        # EP=1: local dispatch — no all-to-all needed.
        if self.ep_mesh is None:
            return (
                routed_input_ND,
                num_local_tokens_per_expert_E,
                LocalDispatchMetadata(
                    token_indices_experts_sorted_N=sorted_indices_N,
                    topk_scores_experts_sorted_N=topk_scores_TK,
                ),
            )

        ep_size = self.ep_mesh.size()
        # EP all-to-all below produces (R, D) where R != N = T*K.

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
            # Reorder from rank-major to expert-major via _permute (inherited).
            (
                input_shape,
                routed_input_RD,
                permuted_indices,
                num_global_tokens_per_local_expert_e,
            ) = self._permute(
                routed_input_RD,
                num_global_tokens_per_local_expert_E,
            )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=sorted_indices_N,
            topk_scores_experts_sorted_N=topk_scores_TK,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    # pyrefly: ignore [bad-override]
    def combine(
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata,
        x_TD: torch.Tensor,
        *,
        num_local_tokens_after_padding: int,
        local_seq_len_after_padding: int,
    ) -> torch.Tensor:
        """Reverse the dispatch: unpermute + all-to-all + fused unpermute.

        ``npu_moe_token_unpermute`` replaces the upstream score multiply +
        ``deterministic_scatter_add``: the kernel scatters each routed row
        back to its original token, multiplies by the routing score, and sums
        over top-K, accumulating internally in fp32. The output dtype follows
        the first argument.

        When sp_size > 1, dispatch uses local token indices. The fused
        unpermute reconstructs the local ``(T, D)`` shard; combine then
        scatters it to global positions so the full SP view is correct.

        Args:
            routed_output_RD: ``(R, D)`` expert outputs in expert-major order
            metadata: AllToAllDispatchMetadata from dispatch()
            x_TD: ``(T, D)`` original input tokens
            num_local_tokens_after_padding: Local token count to use for the
                combined SP view after logical padding. MoE padding passes this
                count without materializing pad rows.
            local_seq_len_after_padding: Per-batch local sequence length after
                logical padding, used to map local token indices to global SP
                positions.

        Returns:
            out_TD: Combined output. With SP, shape is
                ``(num_local_tokens_after_padding * sp_size, D)``.
        """
        # EP=1: fused NPU unpermute — no all-to-all to reverse.
        if self.ep_mesh is None:
            return torch_npu.npu_moe_token_unpermute(
                routed_output_RD,
                metadata.token_indices_experts_sorted_N,
                probs=metadata.topk_scores_experts_sorted_N,  # topk_scores_TK
            )

        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            # Reverse expert-major reordering (inherited)
            routed_output_RD = self._unpermute(routed_output_RD, metadata.input_shape, metadata.permuted_indices)
            # All-to-all combine: returns AsyncCollectiveTensor — the a2a runs
            # on the HCCL stream and won't block until the tensor is accessed.
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
        local_out = torch_npu.npu_moe_token_unpermute(
            routed_output_RD,
            metadata.token_indices_experts_sorted_N,
            probs=metadata.topk_scores_experts_sorted_N,  # topk_scores_TK
        )

        # SP: scatter local [T, D] output to global [T*sp_size, D] buffer.
        if self.sp_size > 1:
            T = local_out.shape[0]
            out_TD = torch.zeros(
                num_local_tokens_after_padding * self.sp_size,
                local_out.shape[-1],
                device=local_out.device,
                dtype=local_out.dtype,
            )
            local_indices = torch.arange(T, device=local_out.device)
            global_indices = self._sp_global_token_indices(
                local_indices,
                local_seq_len_after_padding,
            )
            out_TD.scatter_(
                0,
                global_indices.unsqueeze(-1).expand(-1, local_out.shape[-1]),
                local_out,
            )
            return out_TD

        return local_out


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="NPU fused npu_moe_token_permute/unpermute for MoE dispatch/combine",
)
def npu_all_to_all_token_dispatcher(
    cfg: AllToAllTokenDispatcher.Config,
) -> NPUAllToAllTokenDispatcher.Config:
    return derive(cfg, NPUAllToAllTokenDispatcher.Config)
