# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4095

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport standard-dispatch support for pre-W2 router-score absorption.

The TorchTitan revision pinned by ``requirements.txt`` applies scores after
combine. This patch adds the opt-in score transport needed by standard EP1/EP2
dispatch while keeping the default path numerically identical. DeepEP keeps
its pinned dispatch/combine behavior for now, but exposes the same
``absorb_router_scores`` configuration knob for NPU dispatcher composition;
score absorption remains disabled by default. HybridEP, MinimalAsyncEP, and
TorchAO keep their pinned behavior and are not patched here. The dispatcher
contract follows the upstream design in
https://github.com/pytorch/torchtitan/pull/4095.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torchtitan.distributed.spmd_types import current_spmd_mesh, maybe_set_sparse_mesh
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata as TorchTitanAllToAllDispatchMetadata,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher as TorchTitanAllToAllTokenDispatcher,
)
from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher as TorchTitanDeepEPTokenDispatcher,
)
from torchtitan.models.common.token_dispatcher import (
    LocalDispatchMetadata as TorchTitanLocalDispatchMetadata,
)
from torchtitan.models.common.token_dispatcher import (
    LocalTokenDispatcher as TorchTitanLocalTokenDispatcher,
)
from torchtitan.ops.scatter_add import deterministic_scatter_add


@dataclass(frozen=True, kw_only=True)
class LocalDispatchMetadata(TorchTitanLocalDispatchMetadata):
    """Local dispatch metadata with optional pre-W2 router scores."""

    routed_scores_R: torch.Tensor | None = None  # noqa: N815


@dataclass(frozen=True, kw_only=True)
class AllToAllDispatchMetadata(TorchTitanAllToAllDispatchMetadata):
    """All-to-all metadata with optional pre-W2 router scores."""

    routed_scores_R: torch.Tensor | None = None  # noqa: N815


class LocalTokenDispatcher(TorchTitanLocalTokenDispatcher):
    """Local dispatcher with optional pre-W2 router-score absorption."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanLocalTokenDispatcher.Config):
        # Patch override: retain router scores for pre-W2 absorption.
        absorb_router_scores: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.absorb_router_scores = config.absorb_router_scores

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        (
            routed_input_RD,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        # Patch override: carry the expert-sorted scores in dispatch metadata
        # so grouped experts can apply them before the down projection.
        routed_scores_R = topk_scores_experts_sorted_N if self.absorb_router_scores else None
        # Patch override: extend the upstream metadata with routed scores.
        metadata = LocalDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            routed_scores_R=routed_scores_R,
        )
        return routed_input_RD, num_local_tokens_per_expert_E, metadata

    def combine(  # pyrefly: ignore [bad-override]
        self,
        routed_output_RD: torch.Tensor,
        metadata: LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        out_TD = torch.zeros_like(x_TD)
        # Patch override: the config is the single source of truth; skip the
        # multiply when grouped experts applied scores before down projection.
        if not self.absorb_router_scores:
            routed_output_RD = (
                routed_output_RD.to(torch.float32) * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)
        dim = x_TD.shape[-1]
        return deterministic_scatter_add(
            out_TD,
            metadata.token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, dim),
            routed_output_RD,
        )


class AllToAllTokenDispatcher(TorchTitanAllToAllTokenDispatcher):
    """Standard EP dispatcher with optional pre-W2 router-score absorption."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanAllToAllTokenDispatcher.Config):
        # Patch override: retain router scores for pre-W2 absorption.
        absorb_router_scores: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.absorb_router_scores = config.absorb_router_scores

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        if self.ep_mesh is None:
            return LocalTokenDispatcher.dispatch(
                self,  # pyrefly: ignore [bad-argument-type]
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        ep_size = self.ep_mesh.size()
        (
            routed_input_ND,
            token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N,
        ) = self._local_reorder(x_TD, topk_scores_TK, topk_expert_ids_TK)
        # Patch override: keep scores aligned with the rank-major token stream
        # so they can follow the same EP all-to-all as routed activations.
        routed_scores_N = topk_scores_experts_sorted_N if self.absorb_router_scores else None

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            for axis in ("dp", "cp", "tp"):
                spmd.mutate_type(num_local_tokens_per_expert_E, axis, src=spmd.P, dst=spmd.V)

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
                if routed_scores_N is not None:
                    routed_scores_N = spmd.reinterpret_mesh(routed_scores_N, spmd.current_mesh())

            with torch.no_grad():
                num_global_tokens_per_local_expert_EP_e = self._token_count_exchange(
                    num_local_tokens_per_expert_E, pg, ep_size
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

            routed_input_RD = self._dispatch_token_exchange(routed_input_ND, pg, output_splits_list, input_splits_list)
            # Patch override: transport router scores through EP in their
            # original dtype, alongside the routed activations.
            routed_scores_rank_major_R = (
                self._dispatch_token_exchange(
                    routed_scores_N.reshape(-1, 1), pg, output_splits_list, input_splits_list
                ).reshape(-1)
                if routed_scores_N is not None
                else None
            )
            (
                input_shape,
                routed_input_RD,
                permuted_indices,
                num_global_tokens_per_local_expert_e,
                routed_scores_R,
            ) = self._permute(
                routed_input_RD,
                num_global_tokens_per_local_expert_E,
                routed_scores_rank_major_R,
            )

        metadata = AllToAllDispatchMetadata(
            token_indices_experts_sorted_N=token_indices_experts_sorted_N,
            topk_scores_experts_sorted_N=topk_scores_experts_sorted_N,
            input_shape=input_shape,
            permuted_indices=permuted_indices,
            input_splits=input_splits_list,
            output_splits=output_splits_list,
            # Patch override: expose the EP-aligned scores to grouped experts.
            routed_scores_R=routed_scores_R,
        )
        return routed_input_RD, num_global_tokens_per_local_expert_e, metadata

    def _permute(  # pyrefly: ignore[bad-override]
        self,
        routed_input_RD: torch.Tensor,
        num_global_tokens_per_local_expert_E: torch.Tensor,
        routed_scores_rank_major_R: torch.Tensor | None = None,
    ):
        """Reorder rank-major tokens (and optional scores) to expert-major."""
        (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
        ) = super()._permute(
            routed_input_RD,
            num_global_tokens_per_local_expert_E,
        )
        # Patch override: scores must follow the same rank-major -> expert-major
        # index mapping as routed_input_RD.
        routed_scores_R = (
            routed_scores_rank_major_R[permuted_indices] if routed_scores_rank_major_R is not None else None
        )
        return (
            input_shape,
            routed_input_RD,
            permuted_indices,
            num_global_tokens_per_local_expert_e,
            routed_scores_R,
        )

    def combine(  # pyrefly: ignore [bad-override]
        self,
        routed_output_RD: torch.Tensor,
        metadata: AllToAllDispatchMetadata | LocalDispatchMetadata,
        x_TD: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_mesh is None:
            assert isinstance(metadata, LocalDispatchMetadata)
            return LocalTokenDispatcher.combine(
                self,  # pyrefly: ignore [bad-argument-type]
                routed_output_RD,
                metadata,
                x_TD,
            )

        assert isinstance(metadata, AllToAllDispatchMetadata)
        with maybe_set_sparse_mesh():
            pg = (
                current_spmd_mesh().get_group(  # pyrefly: ignore [missing-attribute]
                    "ep"
                )
                if get_spmd_backend() == "spmd_types"
                else self.ep_mesh.get_group()
            )
            routed_output_RD = self._unpermute(routed_output_RD, metadata.input_shape, metadata.permuted_indices)
            routed_output_RD = self._combine_token_exchange(
                routed_output_RD, pg, metadata.input_splits, metadata.output_splits
            )

        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            routed_output_RD = spmd.reinterpret_mesh(routed_output_RD, spmd.current_mesh())

        out_TD = torch.zeros_like(x_TD)
        # Patch override: use the dispatcher config instead of a per-call flag;
        # grouped experts may have absorbed scores before the down projection.
        if not self.absorb_router_scores:
            routed_output_RD = (
                routed_output_RD.to(torch.float32) * metadata.topk_scores_experts_sorted_N.reshape(-1, 1)
            ).to(routed_output_RD.dtype)

        token_indices_experts_sorted_N = metadata.token_indices_experts_sorted_N
        assert isinstance(token_indices_experts_sorted_N, torch.Tensor)
        return deterministic_scatter_add(
            out_TD,
            token_indices_experts_sorted_N.reshape(-1, 1).expand(-1, out_TD.shape[-1]),
            routed_output_RD,
        )


class DeepEPTokenDispatcher(TorchTitanDeepEPTokenDispatcher):
    """DeepEP dispatcher with a forward-compatible absorption config knob.

    DeepEP's score transport is owned by its backend-specific dispatch state,
    so this patch intentionally does not alter dispatch or combine. The
    ``absorb_router_scores`` field is exposed for dispatcher config composition
    and remains disabled by default until DeepEP pre-W2 absorption is wired.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TorchTitanDeepEPTokenDispatcher.Config):
        absorb_router_scores: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.absorb_router_scores = config.absorb_router_scores


def apply() -> None:
    import torchtitan.models.common.config_utils
    import torchtitan.models.common.token_dispatcher as token_dispatcher_module

    token_dispatcher_module.LocalTokenDispatcher = LocalTokenDispatcher  # pyrefly: ignore [bad-assignment]
    token_dispatcher_module.AllToAllTokenDispatcher = AllToAllTokenDispatcher  # pyrefly: ignore [bad-assignment]
    token_dispatcher_module.DeepEPTokenDispatcher = DeepEPTokenDispatcher  # pyrefly: ignore [bad-assignment]

    # Patch override: DeepSeek-V4's ``_make_moe_config()`` calls
    # ``config_utils.make_routed_experts_config()``, whose module-level
    # ``from`` imports cached the upstream dispatcher classes. Refresh those
    # bindings so its factory builds configs with the patched dispatcher
    # contracts.
    torchtitan.models.common.config_utils.LocalTokenDispatcher = (  # pyrefly: ignore [bad-assignment]
        LocalTokenDispatcher
    )
    torchtitan.models.common.config_utils.AllToAllTokenDispatcher = (  # pyrefly: ignore [bad-assignment]
        AllToAllTokenDispatcher
    )
    torchtitan.models.common.config_utils.DeepEPTokenDispatcher = (  # pyrefly: ignore [bad-assignment]
        DeepEPTokenDispatcher
    )


apply()
