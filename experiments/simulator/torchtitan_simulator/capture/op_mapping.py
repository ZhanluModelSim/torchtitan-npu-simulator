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
    "simulator.stream_wait": "stream_wait",
    # linear algebra
    "aten.addmm.default": "matmul",
    "aten.mm.default": "matmul",
    "aten.bmm.default": "bmm",
    "aten.matmul.default": "matmul",
    "aten.einsum.default": "einsum",
    "aten._grouped_mm.default": "grouped_mm",
    "npu.npu_quant_matmul.default": "matmul",
    "npu.npu_grouped_matmul.default": "grouped_mm",
    "npu.npu_dynamic_mx_quant.default": "quantize",
    "npu.npu_dynamic_mx_quant_with_dual_axis.default": "quantize",
    "npu.npu_grouped_dynamic_mx_quant.default": "quantize",
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
    "npu.npu_rms_norm_backward.default": "rms_norm_backward",
    # activation
    "aten.gelu.default": "gelu",
    "aten.silu.default": "silu",
    "aten.softmax.default": "softmax",
    "aten._softmax.default": "softmax",
    "npu.npu_swiglu.default": "swiglu",
    "npu.npu_swiglu_backward.default": "swiglu",
    "cann_ops_nn.swiglu_group.default": "swiglu",
    "cann_ops_nn.swiglu_group_backward.default": "swiglu",
    "fusion_attention": "fusion_attention",
    "fusion_attention_grad": "fusion_attention",
    "situ_glu": "situ_glu",
    "kimi_sim.situ_glu.default": "situ_glu",
    "kimi_sim.situ_glu_backward.default": "situ_glu",
    "kimi_sim.fusion_attention.default": "fusion_attention",
    "kimi_sim.fusion_attention_grad.default": "fusion_attention",
    "kimi_sim.chunk_kda.default": "chunk_kda",
    "kimi_sim.chunk_kda_grad.default": "chunk_kda",
    "situ_glu_backward": "situ_glu",
    "npu.npu_rotary_mul.default": "rope",
    # MoE dispatch
    "npu.npu_moe_token_permute.default": "moe_token_permute",
    "npu.npu_moe_token_permute_grad.default": "moe_token_permute",
    "npu.npu_moe_token_unpermute.default": "moe_token_unpermute",
    "npu.npu_moe_re_routing.default": "moe_re_routing",
    # memory / IO
    "aten.view.default": "view",
    "aten.reshape.default": "reshape",
    "aten.permute.default": "transpose",
    "aten.transpose.int": "transpose",
    "aten.t.default": "transpose",
    "aten.slice.Tensor": "metadata_view",
    "aten.select.int": "metadata_view",
    "aten.narrow.default": "metadata_view",
    "aten.as_strided.default": "metadata_view",
    "aten.squeeze.dim": "metadata_view",
    "aten.squeeze.default": "metadata_view",
    "aten.unsqueeze.default": "metadata_view",
    "aten.expand.default": "metadata_view",
    "aten.alias.default": "metadata_view",
    "aten.detach.default": "metadata_view",
    "aten.cat.default": "cat",
    "aten.split.default": "split",
    "aten.split.Tensor": "split",
    "aten.split_with_sizes.default": "split",
    # optimizer
    "aten._fused_adamw_.default": "adamw_step",
    "_c10d_functional.all_reduce.default": "allreduce",
    "_c10d_functional.all_gather_into_tensor.default": "allgather",
    "_c10d_functional.reduce_scatter_tensor.default": "reduce_scatter",
    "_c10d_functional.all_to_all_single.default": "all_to_all",
    "c10d.allreduce_.default": "allreduce",
    "c10d.broadcast_.default": "broadcast",
    "c10d._allgather_base_.default": "allgather",
    "c10d._reduce_scatter_base_.default": "reduce_scatter",
    "c10d.alltoall_base_.default": "all_to_all",
    "c10d.alltoall_.default": "all_to_all",
    "_dtensor.shard_dim_alltoall.default": "all_to_all",
    # Legacy collective entry points without a dispatch-visible transport.
    "comm.allreduce": "allreduce",
    "comm.allgather": "allgather",
    "comm.reduce_scatter": "reduce_scatter",
    "comm.all_to_all": "all_to_all",
    "comm.broadcast": "broadcast",
    "comm.p2p_send": "p2p_send",
    "comm.p2p_recv": "p2p_recv",
}

# These dispatcher calls only change tensor metadata or aliasing.  They are
# retained in the L0 IR because their aliases and dependency edges matter for
# replay and liveness, but they are not launched device kernels.
_METADATA_VIEW_PREFIXES = (
    "aten.view.",
    "aten.reshape.",
    "aten.permute.",
    "aten.transpose.",
    "aten.t.",
    "aten.slice.",
    "aten.select.",
    "aten.narrow.",
    "aten.as_strided.",
    "aten.split.",
    "aten.split_with_sizes.",
    "aten.squeeze.",
    "aten.unsqueeze.",
    "aten.expand.",
    "aten.alias.",
    "aten.detach.",
)


def is_metadata_view_op(raw_op_type: str) -> bool:
    """Whether ``raw_op_type`` is an alias-only metadata view operation.

    ``cat``, ``clone`` and ``contiguous`` are deliberately excluded: they can
    allocate/copy storage and remain visible device work.
    """
    return raw_op_type.startswith(_METADATA_VIEW_PREFIXES)


def to_canonical_op_type(raw_op_type: str) -> str:
    """Map a raw dispatcher op name (e.g. `str(func)` from
    `__torch_dispatch__`) to its canonical L0 op_type, or `"unknown"`."""
    return OP_MAPPING.get(raw_op_type, "unknown")


def communication_primitive(raw_op_type: str) -> str | None:
    canonical = OP_MAPPING.get(raw_op_type)
    return (
        canonical
        if canonical in {"allreduce", "allgather", "reduce_scatter", "all_to_all", "broadcast", "p2p_send", "p2p_recv"}
        else None
    )
