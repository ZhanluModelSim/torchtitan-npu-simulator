# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 dense decoder: repo-local per-batch metadata protocol on upstream geometry.

The upstream ``torchtitan.models.qwen3`` implementation owns the model math.
This module only rebinds the configuration onto a subclass that provides the
``build_attention_masks`` entry used by the training and simulation runners;
geometry builders are reused from upstream instead of being copied.
"""

from dataclasses import dataclass, fields

from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.qwen3.model import Qwen3Model as _UpstreamQwen3Model

__all__ = ["Qwen3Model", "as_npu_config"]


class Qwen3Model(_UpstreamQwen3Model):
    """Upstream Qwen3 dense decoder plus the repo-local mask-building protocol."""

    @dataclass(kw_only=True, slots=True)
    class Config(_UpstreamQwen3Model.Config):
        @property
        def n_layers(self) -> int:
            # Runner summary protocol (shared with the kimi_k3/deepseek_v4
            # configs); derived from the layer list so reduced-depth
            # presets cannot drift from the actual tree.
            return len(self.layers)

    def build_attention_masks(
        self,
        inputs,
        labels,
        extra_kwargs,
        *,
        cp_mesh=None,
        load_balancer_type=None,
    ):
        """Derive per-batch varlen metadata on the host device.

        Dense Qwen3 keeps the upstream causal contract: document boundaries
        come from ``positions`` resets and every layer consumes the shared
        ``VarlenMetadata``.  Context parallelism is rejected explicitly: the
        upstream varlen backend raises for CP>1 and the dense recipe has no
        CP metadata plan, so silent chunking here would hide that boundary.
        """
        if cp_mesh is not None:
            raise NotImplementedError("Qwen3 dense training does not support context parallelism")
        if load_balancer_type is not None:
            raise ValueError("Qwen3 dense training does not use a context-parallel load balancer")
        positions = extra_kwargs.get("positions")
        if positions is None:
            raise ValueError("positions are required to build Qwen3 varlen metadata")
        masks = self.get_attention_masks(positions=positions)
        if not isinstance(masks, VarlenMetadata):
            raise TypeError(
                f"the varlen backend must produce VarlenMetadata for Qwen3, got {type(masks).__name__}"
            )
        extra_kwargs = dict(extra_kwargs)
        extra_kwargs["attention_masks"] = masks
        return inputs, labels, extra_kwargs


def as_npu_config(upstream: _UpstreamQwen3Model.Config) -> "Qwen3Model.Config":
    """Rebind a fresh upstream Qwen3Model.Config onto the repo-local class.

    The upstream config tree is built inside the calling factory, so the
    shallow field copy still yields an independent tree per call while
    tracking any future upstream fields without duplicating geometry code.
    """
    return Qwen3Model.Config(**{field.name: getattr(upstream, field.name) for field in fields(upstream)})
