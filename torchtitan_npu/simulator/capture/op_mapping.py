# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maps raw dispatcher op names (aten/npu) to canonical L0 op_type strings
(the canonical op set from spec/L0-OpNode.md, plus NPU-specific extensions
used by DeepSeek-V4: rms_norm, rope, swiglu, MoE dispatch, sparse-attention
family). Any raw op name absent from OP_MAPPING resolves to "unknown" --
OpCostModel (Task 6) already handles "unknown" op_type gracefully."""

from __future__ import annotations

OP_MAPPING: dict[str, str] = {
    # linear algebra
    "aten.addmm.default": "matmul",
    "aten.mm.default": "matmul",
    "aten.bmm.default": "bmm",
    "aten.matmul.default": "matmul",
    "aten.einsum.default": "einsum",
    # attention
    "aten.scaled_dot_product_attention.default": "sdpa",
    "aten._scaled_dot_product_flash_attention.default": "flash_attention_fwd",
    "npu.npu_fusion_attention.default": "flash_attention_fwd",
    "npu.npu_sparse_flash_attention.default": "flash_attention_fwd",
    "npu.npu_sparse_attn_sharedkv.default": "flash_attention_fwd",
    "npu.npu_lightning_indexer.default": "flash_attention_fwd",
    # normalization
    "aten.native_layer_norm.default": "layer_norm",
    "npu.npu_rms_norm.default": "rms_norm",
    # activation
    "aten.gelu.default": "gelu",
    "aten.silu.default": "silu",
    "aten.softmax.default": "softmax",
    "aten._softmax.default": "softmax",
    "npu.npu_swiglu.default": "swiglu",
    "npu.npu_rotary_mul.default": "rope",
    # MoE dispatch
    "npu.npu_moe_token_permute.default": "moe_token_permute",
    "npu.npu_moe_token_unpermute.default": "moe_token_unpermute",
    "npu.npu_moe_re_routing.default": "moe_re_routing",
    # memory / IO
    "aten.view.default": "view",
    "aten.reshape.default": "reshape",
    "aten.permute.default": "transpose",
    "aten.transpose.int": "transpose",
    "aten.cat.default": "cat",
    "aten.split.default": "split",
    "aten.split_with_sizes.default": "split",
    # optimizer
    "aten._fused_adamw_.default": "adamw_step",
    # communication (synthetic ops registered by comm_events.py)
    "comm.allreduce": "allreduce",
    "comm.allgather": "allgather",
    "comm.reduce_scatter": "reduce_scatter",
    "comm.all_to_all": "all_to_all",
    "comm.broadcast": "broadcast",
    "comm.p2p_send": "p2p_send",
    "comm.p2p_recv": "p2p_recv",
}


def to_canonical_op_type(raw_op_type: str) -> str:
    """Map a raw dispatcher op name (e.g. `str(func)` from
    `__torch_dispatch__`) to its canonical L0 op_type, or `"unknown"`."""
    return OP_MAPPING.get(raw_op_type, "unknown")


def display_op_label(op_type: str, annotations: dict) -> str:
    """Resolve the op label to show in human-facing output (graph
    visualizations, text summaries). `op_type == "unknown"` must never be
    shown verbatim: the *real* dispatcher op name is always available in
    `annotations["raw_op_type"]` (captured by dispatch_capture.py
    regardless of OP_MAPPING coverage), so falling back to it keeps every
    node's displayed identity consistent with the real op that actually
    ran, even for ops absent from the curated OP_MAPPING table."""
    if op_type != "unknown":
        return op_type
    return annotations.get("raw_op_type", "unknown")
