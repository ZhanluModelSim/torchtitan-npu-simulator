# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN Compressor entry point with symbolic-safe output shape inference."""

import importlib

import torch


def _compressor_forward_fake(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio,
    state_block_table=None,
    cu_seqlens: torch.Tensor | None = None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1,
    grad_enabled=False,
):
    # Match CANN's output capacity, including partial blocks in packed input.
    # Python min in older CANN packages guards on independent unbacked sizes.
    head_dim = wkv.shape[0] // coff
    if x.ndim == 3:
        prefix = (x.shape[0], (x.shape[1] + cmp_ratio - 1) // cmp_ratio)
    else:
        assert cu_seqlens is not None
        batch_size = cu_seqlens.shape[0] - 1
        tokens = x.shape[0]
        capacity = tokens // cmp_ratio + batch_size
        # min(a, b) for integer sizes, without Python guards or sym_min dispatch
        # (which can return NotImplemented in the compiled CANN fake call).
        prefix = ((tokens + capacity - abs(tokens - capacity)) // 2,)
    pooled_shape = (*prefix, head_dim)
    saved_shape = (*prefix, coff * cmp_ratio, head_dim)
    return (
        x.new_empty(pooled_shape),
        x.new_empty(saved_shape, dtype=torch.float32),
        x.new_empty(saved_shape, dtype=torch.float32),
    )


def load_compressor():
    """Load only for the selected override; preserve CANN kernels and autograd."""
    module = importlib.import_module("cann_ops_transformer.ops.compressor")
    forward_op = getattr(torch.ops.cann_ops_transformer, "_compressor_forward", None)
    if forward_op is not None:
        # Remove when supported CANN packages use symbolic-safe shape inference.
        torch.library.register_fake(forward_op.default, _compressor_forward_fake)
    compressor_op = getattr(torch.ops.cann_ops_transformer, "compressor", None)
    return compressor_op.default if compressor_op is not None else module.compressor
