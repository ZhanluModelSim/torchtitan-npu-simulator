# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom-op boundary for the TileKernels TopK gate.

The external ``tile_kernels.moe.topk_gate`` implementation is intentionally
loaded only by the runtime custom-op implementation.  During
``torch.compile``/AOT eager tracing, the registered fake implementation
provides output metadata and prevents the optional package from being
executed while tracing.
"""

import torch


@torch.library.custom_op(
    "torchtitan_npu::tilelang_topk_gate",
    mutates_args=(),
    device_types="npu",
)
def tilelang_topk_gate_op(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Run the validated TileKernels TopK gate on an NPU tensor."""

    try:
        from tile_kernels.moe import topk_gate as tile_kernels_topk_gate  # pyrefly: ignore [missing-import]
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileKernels topk_gate is unavailable; install the validated "
            "tile_kernels package and load the matching CANN environment."
        ) from error

    return tile_kernels_topk_gate(scores, num_topk)


@tilelang_topk_gate_op.register_fake
def _tilelang_topk_gate_fake(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Return shape/dtype metadata without loading or executing TileKernels."""

    return torch.empty(
        (scores.shape[0], num_topk),
        dtype=torch.int64,
        device=scores.device,
    )


def tilelang_topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """Public wrapper used by the DeepSeek-V4 router."""

    return tilelang_topk_gate_op(scores, num_topk)
