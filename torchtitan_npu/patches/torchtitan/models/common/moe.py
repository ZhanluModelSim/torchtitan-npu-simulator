# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hash routing and SwiGLU clamping for the common MoE (DeepSeek-V4).

Adds ``hash`` / ``vocab_size`` to ``TokenChoiceTopKRouter.Config``, an
``input_ids`` argument to ``MoE.forward``, and the ``"sqrtsoftplus"`` score
function; adds an optional ``swiglu_limit`` to ``GroupedExperts`` (routed
experts) and ``FeedForward`` (shared experts) with the DeepSeek-V4 clamp
(``up`` in ``[-limit, limit]``, ``gate`` capped at ``limit`` — the same
clamp as transformers / the inference repo; 0 disables).  The config
factories gain a ``swiglu_limit`` passthrough.  Swaps the originals at
import time so downstream code keeps using the upstream names; non-hash
configs and the default 0.0 clamp take the byte-identical upstream path.
``config_utils`` is deliberately imported inside ``apply()``, after the
class swaps, so its module-level imports bind the patched classes — keep
the swap-then-import order.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher

__all__ = ["HashMoE", "HashRouter"]


def _build_hash_routing_table(vocab_size, num_experts, top_k, device=None, chunk_size=8192):
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = torch.rand((end - start, num_experts), device=device).topk(top_k, dim=-1).indices
    return tid2eid


class HashRouter(TokenChoiceTopKRouter):
    """TokenChoiceTopKRouter with optional DSV4 hash routing.

    When ``hash`` is set, tokens are routed by a fixed tid->expert table
    (``tid2eid``) instead of score top-k; ``expert_bias_E`` is not applied on
    hash layers.  ``vocab_size`` is required iff ``hash``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        hash: bool = False
        vocab_size: int | None = None
        score_func: Literal[  # pyrefly: ignore [bad-override]
            "softmax", "sigmoid", "sqrtsoftplus"
        ] = "sigmoid"

    def __init__(self, config: Config):
        super().__init__(config)
        self.hash = config.hash
        self.vocab_size = config.vocab_size
        if self.hash:
            if config.vocab_size is None:
                raise ValueError("hash routing requires vocab_size.")
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(self.vocab_size, self.num_experts, self.top_k),
                persistent=True,
            )

    def _init_self_buffers(self, *, buffer_device=None):
        if self.hash:
            if buffer_device is None:
                buffer_device = self.tid2eid.device
            with torch.device(buffer_device):  # pyrefly: ignore [no-matching-overload]
                self.tid2eid = _build_hash_routing_table(
                    self.vocab_size,
                    self.num_experts,
                    self.top_k,
                    device=buffer_device,
                )

    def forward(self, x_BLD, expert_bias_E=None, *, input_ids=None):
        # Compute gate in float32 to help stability of expert load balancing
        # (torchtitan TokenChoiceTopKRouter pattern).
        with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
            scores = self.gate(x_BLD)
        # scores is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        elif self.score_func == "sqrtsoftplus":
            # Use the baseline's ``F.softplus`` expression directly. Equivalent
            # formulas can round differently near top-k decision boundaries.
            scores = F.softplus(scores).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            if input_ids is None:
                raise ValueError("input_ids is required for DSV4 hash routing.")
            selected_experts_indices = self.tid2eid[input_ids]
        else:
            scores_for_choice = scores if expert_bias_E is None else scores + expert_bias_E
            # Apply node-limited routing if configured (upstream behavior).
            if self.num_expert_groups is not None:
                scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1, sorted=False)[1]

        top_scores = scores.gather(dim=-1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices, scores


class HashMoE(MoE):
    """MoE whose forward threads an optional ``input_ids`` to the router.

    ``Config`` is redefined so ``build()`` instantiates this class rather than
    the upstream ``MoE`` (the inherited alias would construct the base).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def forward(self, x_BLD: torch.Tensor, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Forward through the router (with optional ``input_ids``) and experts.

        The body mirrors upstream ``MoE.forward``; ``input_ids`` is only
        consumed by the router's hash path.
        """
        B, L, _D = x_BLD.shape
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
            local_batch_size = x_BLD._local_tensor.shape[0] if isinstance(x_BLD, DTensor) else B
            num_local_tokens_after_seq_dim_padding = local_batch_size * (L + seq_dim_pad_tokens) // sp_size

        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(x_BLD, self.expert_bias_E, input_ids=input_ids)

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
            num_local_tokens_after_seq_dim_padding=(num_local_tokens_after_seq_dim_padding),
        )

        shared_out_BLD = self.shared_experts(x_BLD) if self.shared_experts is not None else None

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


class _ClampGroupedExperts(GroupedExperts):
    """GroupedExperts with the optional DeepSeek-V4 SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        # SwiGLU limit on the gate/up activations; 0 disables the clamp.
        # DeepSeek-V4 uses 10.0 (transformers config default).
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation; the gate/up grouped-mms are clamped
        before the SiLU when ``swiglu_limit > 0``."""
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        g_RF = torch._grouped_mm(
            x_RD.bfloat16(),
            w1_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        u_RF = torch._grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        if self.swiglu_limit > 0:
            u_RF = torch.clamp(u_RF, min=-self.swiglu_limit, max=self.swiglu_limit)
            g_RF = torch.clamp(g_RF, max=self.swiglu_limit)
        h_RF = F.silu(g_RF) * u_RF
        return torch._grouped_mm(h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E).type_as(x_RD)


class _ClampFeedForward(FeedForward):
    """FeedForward (shared expert) with the optional SwiGLU clamp."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        swiglu_limit: float = 0.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.w2(F.silu(gate) * up)


def _clamp_make_routed_experts_config(*, swiglu_limit: float = 0.0, **kwargs):
    """``make_routed_experts_config`` with the clamp passthrough."""
    cfg = _original_make_routed_experts_config(**kwargs)
    return dataclasses.replace(
        cfg,
        inner_experts=dataclasses.replace(cfg.inner_experts, swiglu_limit=swiglu_limit),
    )


def _clamp_make_ffn_config(*, swiglu_limit: float = 0.0, **kwargs):
    """``make_ffn_config`` with the clamp passthrough."""
    cfg = _original_make_ffn_config(**kwargs)
    return dataclasses.replace(cfg, swiglu_limit=swiglu_limit)


# Assigned in ``apply()`` before the clamp factories are ever called; declared
# at module scope so type checkers see the names.
_original_make_routed_experts_config: Callable[..., Any] = lambda **kwargs: None
_original_make_ffn_config: Callable[..., Any] = lambda **kwargs: None


# Swap the originals so ``torchtitan.models.common.moe`` resolves to the
# extended classes (see the module docstring).  Modules imported after this
# patch binds the new classes; ``hash=False`` and the default 0.0 clamp keep
# the upstream behavior.


def apply() -> None:
    import torchtitan.models.common.feed_forward
    import torchtitan.models.common.moe

    # Swap the classes BEFORE ``config_utils`` is imported: its module-level
    # ``from torchtitan.models.common.moe import GroupedExperts`` (and
    # ``from ...feed_forward import FeedForward``) then binds the patched
    # classes, so the factories build the clamped configs.  Importing it
    # earlier would freeze the originals into its namespace and the
    # factories would keep building unpatched configs.
    torchtitan.models.common.moe.TokenChoiceTopKRouter = HashRouter
    torchtitan.models.common.moe.MoE = HashMoE
    torchtitan.models.common.moe.GroupedExperts = _ClampGroupedExperts
    torchtitan.models.common.feed_forward.FeedForward = _ClampFeedForward

    import torchtitan.models.common.config_utils

    global _original_make_routed_experts_config, _original_make_ffn_config
    _original_make_routed_experts_config = torchtitan.models.common.config_utils.make_routed_experts_config
    _original_make_ffn_config = torchtitan.models.common.config_utils.make_ffn_config
    torchtitan.models.common.config_utils.make_routed_experts_config = _clamp_make_routed_experts_config
    torchtitan.models.common.config_utils.make_ffn_config = _clamp_make_ffn_config


apply()
