# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention primitives specific to block-diffusion training."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import ScaledDotProductAttention


def build_prefix_canvas_attention_mask(
    seq_len: int,
    block_size: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the reference prefix-causal/canvas-bidirectional visibility mask.

    ``True`` means visible, matching PyTorch SDPA's boolean-mask convention.
    Production attention uses two SDPA calls instead of materializing this
    quadratic tensor; this helper makes the contract explicit and testable.
    """

    if block_size <= 0 or seq_len < block_size or seq_len % block_size != 0:
        raise ValueError(
            "seq_len must be a positive multiple of block_size and contain "
            f"at least one block, got seq_len={seq_len}, block_size={block_size}"
        )

    prefix_len = seq_len - block_size
    query = torch.arange(seq_len, device=device).unsqueeze(1)
    key = torch.arange(seq_len, device=device).unsqueeze(0)
    return torch.where(query < prefix_len, key <= query, key < seq_len)


class PrefixCanvasSDPA(ScaledDotProductAttention):
    """Causal clean prefix followed by one bidirectional denoising canvas.

    For a sequence of length ``S`` and canvas size ``B``, queries ``[0, S-B)``
    attend causally within the prefix. Queries ``[S-B, S)`` attend to every
    prefix and canvas key. Splitting the query range into two SDPA calls avoids
    allocating an ``S x S`` mask while preserving the exact visibility graph.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductAttention.Config):
        block_size: int

        def __post_init__(self) -> None:
            if self.block_size <= 0:
                raise ValueError("block_size must be greater than zero")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.block_size = config.block_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks=None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "PrefixCanvasSDPA constructs its visibility pattern internally; "
                "an external attention mask is not supported"
            )
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("PrefixCanvasSDPA requires equal Q/K/V sequence lengths")

        seq_len = q.shape[1]
        if seq_len < self.block_size or seq_len % self.block_size != 0:
            raise ValueError(
                "sequence length must be a multiple of block_size and contain "
                f"one canvas, got seq_len={seq_len}, block_size={self.block_size}"
            )

        prefix_len = seq_len - self.block_size
        outputs = []
        if prefix_len:
            outputs.append(
                super().forward(
                    q[:, :prefix_len],
                    k[:, :prefix_len],
                    v[:, :prefix_len],
                    scale=scale,
                    enable_gqa=enable_gqa,
                    is_causal=True,
                    **kwargs,
                )
            )
        outputs.append(
            super().forward(
                q[:, prefix_len:],
                k,
                v,
                scale=scale,
                enable_gqa=enable_gqa,
                is_causal=False,
                **kwargs,
            )
        )
        return torch.cat(outputs, dim=1) if prefix_len else outputs[0]
