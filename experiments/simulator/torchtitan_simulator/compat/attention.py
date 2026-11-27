# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta autograd bridge for native CANN fused attention.

The bridge preserves native forward and gradient dispatch identities and
carries the host-known document lengths needed to price TND attention.
"""

from functools import wraps
from itertools import pairwise

import torch

from torchtitan_simulator.capture.dispatch_capture import capture_attention_geometry


def _v3_geometry(query, key, layout, lengths):
    if layout == "BNSD":
        qs = (query.shape[2],) * query.shape[0]
        ks = (key.shape[2],) * key.shape[0]
        return {"query_lens": qs, "kv_lens": ks}
    if lengths is None:
        return {"query_lens": (query.shape[0],), "kv_lens": (key.shape[0],)}
    if len(lengths) < 2 or lengths[0] != 0:
        raise ValueError("Packed attention offsets must delimit documents from zero")
    lens = tuple(end - begin for begin, end in pairwise(lengths))
    return {"query_lens": lens, "kv_lens": lens}


def causal_fusion_attention_v3(query, key, value, scale=None):
    """Run native causal attention for one dense text document per batch entry."""
    import torch_npu

    # CANN bottom-right causal mode uses the compressed triangular mask template.
    mask = torch.ones((2048, 2048), device=query.device, dtype=torch.bool).triu(1)
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _FusionAttentionV3MetaAutograd.apply(
        query,
        key,
        value,
        mask,
        None,
        query.shape[1],
        scale,
        2147483647,
        0,
        torch_npu.npu_fusion_attention_v3,
        "BNSD",
        3,
        None,
    )


class _FusionAttentionV3MetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        atten_mask,
        actual_seq,
        head_num,
        scale,
        pre_tockens,
        next_tockens,
        original,
        layout,
        sparse_mode,
        host_actual_seq,
    ):
        # The caller's TND lengths are already host-known before being moved to Meta.
        meta_lengths = actual_seq
        if layout == "TND" and host_actual_seq is None and actual_seq is not None:
            raise ValueError("Meta TND attention requires host-known segment lengths")
        facts = _v3_geometry(query, key, layout, host_actual_seq)
        if actual_seq is not None:
            meta_lengths = actual_seq.to(query.device)
        with capture_attention_geometry("npu.npu_fusion_attention_v3.default", **facts):
            result = original(
                query=query,
                key=key,
                value=value,
                head_num=head_num,
                input_layout=layout,
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=1.0,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
                actual_seq_qlen=meta_lengths,
                actual_seq_kvlen=meta_lengths,
                sparse_mode=sparse_mode,
            )
        # (output, softmax_max, softmax_sum, softmax_in placeholder, seed, offset)
        output, softmax_max, softmax_sum = result[0], result[1], result[2]
        ctx.save_for_backward(query, key, value, atten_mask, meta_lengths, softmax_max, softmax_sum, output)
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.facts = facts
        ctx.head_num = head_num
        ctx.scale = scale
        ctx.pre_tockens = pre_tockens
        ctx.next_tockens = next_tockens
        return output

    @staticmethod
    # Fixed gradient arity follows Function outputs; the base stub uses variadic args.
    # pyrefly: ignore [bad-override]
    def backward(ctx, dy):
        query, key, value, atten_mask, actual_seq, softmax_max, softmax_sum, output = ctx.saved_tensors
        with capture_attention_geometry("npu.npu_fusion_attention_grad_v3.default", **ctx.facts):
            gradients = torch.ops.npu.npu_fusion_attention_grad_v3(
                query=query,
                key=key,
                value=value,
                dy=dy.contiguous(),
                head_num=ctx.head_num,
                input_layout=ctx.layout,
                atten_mask=atten_mask,
                softmax_max=softmax_max,
                softmax_sum=softmax_sum,
                attention_in=output,
                scale_value=ctx.scale,
                keep_prob=1.0,
                pre_tockens=ctx.pre_tockens,
                next_tockens=ctx.next_tockens,
                actual_seq_qlen=actual_seq,
                actual_seq_kvlen=actual_seq,
                sparse_mode=ctx.sparse_mode,
            )
        # Schema returns (dq, dk, dv, pse-grad placeholder, scalar tail).
        return (*gradients[:3], *((None,) * 10))


def install_meta_fusion_attention_autograd(scope):
    """Capture native TND forward and backward with their host document lengths."""
    import torch_npu

    from torchtitan_npu.override.common import varlen_attention

    original = torch_npu.npu_fusion_attention_v3
    # Mirror the RMSNorm bridge contract check: the gradient op must keep the
    # documented input order. The return layout (dq, dk, dv, pse placeholder,
    # scalar tail) is positional-only knowledge for torch_npu 2.14.0.dev20260918.
    grad_schema = torch.ops.npu.npu_fusion_attention_grad_v3.default._schema
    if [argument.name for argument in grad_schema.arguments][:5] != ["query", "key", "value", "dy", "head_num"]:
        raise RuntimeError(f"Fusion attention backward contract requires review: {grad_schema}")
    seam = varlen_attention._fusion_attention_v3

    @wraps(seam)
    def fusion_attention_v3(**kwargs):
        if kwargs["query"].device.type != "meta":
            return seam(**kwargs)
        return _FusionAttentionV3MetaAutograd.apply(
            kwargs["query"],
            kwargs["key"],
            kwargs["value"],
            kwargs["atten_mask"],
            kwargs["actual_seq"],
            kwargs["head_num"],
            kwargs["scale"],
            kwargs["pre_tockens"],
            kwargs["next_tockens"],
            original,
            "TND",
            7,
            kwargs.get("host_actual_seq"),
        )

    scope.replace(varlen_attention, "_fusion_attention_v3", fusion_attention_v3)
    return "bridge"


# ---------------------------------------------------------------------------
# npu_fusion_attention (BSND dense) bridge for the system3 encoder attention.
#
# The vendor Meta kernel for the non-varlen fused op touches the device
# manager, so Meta capture needs pure-shape Meta kernels plus an autograd
# bridge; the probe-first flow mirrors the v3 bridge above.
# ---------------------------------------------------------------------------


_NPU_FA_META_LIBRARY: torch.library.Library | None = None


def _npu_fusion_attention_meta(
    query,
    key,
    value,
    head_num,
    input_layout,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    scale=1.0,
    keep_prob=1.0,
    pre_tockens=0,
    next_tockens=0,
    inner_precise=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
    softmax_layout="",
    sink=None,
    dropout_mask=None,
    seed=0,
    offset=0,
):
    batch, seq, heads, dim = query.shape
    # The native fused op requires equal query/key/value head dims (GQA
    # varies the head count, not the width); an unequal width would produce
    # inconsistent geometry, so the boundary rejects it explicitly.
    if key.shape[-1] != dim or value.shape[-1] != dim:
        raise ValueError(
            "npu_fusion_attention requires equal query/key/value head dims, got "
            f"q={dim}, k={key.shape[-1]}, v={value.shape[-1]}"
        )
    kv_heads = key.shape[2]
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError(
            f"npu_fusion_attention requires a positive GQA head geometry, got heads={heads}, kv_heads={kv_heads}"
        )
    out = torch.empty(batch, seq, heads, dim, dtype=query.dtype, device="meta")
    # Softmax statistics follow the native BSND contract: FP32 per query
    # position, padded to 8 lanes (B, N, S, 8); the empty placeholders made
    # the backward drop softmax_sum and understate the activation memory
    # and traffic.
    softmax_max = torch.empty(batch, heads, seq, 8, dtype=torch.float32, device="meta")
    softmax_sum = torch.empty(batch, heads, seq, 8, dtype=torch.float32, device="meta")
    return out, softmax_max, softmax_sum, torch.empty(0, device="meta"), 0, 0, 0


def _npu_fusion_attention_grad_meta(
    query,
    key,
    value,
    dy,
    head_num,
    input_layout,
    *,
    pse=None,
    padding_mask=None,
    atten_mask=None,
    softmax_max=None,
    softmax_sum=None,
    softmax_in=None,
    attention_in=None,
    scale_value=1.0,
    keep_prob=1.0,
    pre_tockens=0,
    next_tockens=0,
    inner_precise=0,
    seed=0,
    offset=0,
    numels=0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode=0,
    gen_mask_parallel=True,
    sync=False,
    softmax_layout="",
    sink=None,
):
    # Gradients mirror each operand's own geometry (GQA/MQA and differing KV
    # widths included); equal-shape MHA calls are unchanged.
    return (
        torch.empty(query.shape, dtype=query.dtype, device="meta"),
        torch.empty(key.shape, dtype=key.dtype, device="meta"),
        torch.empty(value.shape, dtype=value.dtype, device="meta"),
        torch.empty(0, device="meta"),
        torch.empty(0, device="meta"),
    )


class _NpuFusionAttentionMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, scale, original):
        # Call the raw op, not the torch_npu wrapper: the wrapper queries the
        # device count, which fails in device-less Meta containers. The Meta
        # kernels registered above make this dispatch pure.
        result = torch.ops.npu.npu_fusion_attention.default(
            query, key, value, head_num, "BSND", scale=scale, keep_prob=1.0, sparse_mode=0
        )
        output, softmax_max, softmax_sum, softmax_in = result[0], result[1], result[2], result[3]
        ctx.save_for_backward(query, key, value, softmax_max, softmax_sum, softmax_in)
        ctx.head_num = head_num
        ctx.scale = scale
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, dy):
        query, key, value, softmax_max, softmax_sum, softmax_in = ctx.saved_tensors
        gradients = torch.ops.npu.npu_fusion_attention_grad.default(
            query,
            key,
            value,
            dy,
            ctx.head_num,
            "BSND",
            scale_value=ctx.scale,
            softmax_max=softmax_max if softmax_max.numel() else None,
            softmax_sum=softmax_sum if softmax_sum.numel() else None,
            softmax_in=softmax_in if softmax_in.numel() else None,
            sparse_mode=0,
        )
        return gradients[0], gradients[1], gradients[2], None, None, None


def install_npu_fusion_attention_autograd(scope):
    """Bridge ``npu_fusion_attention`` autograd on Meta for the encoder seam."""
    import torch_npu

    from torchtitan_npu.override.common import fusion_attention

    original = torch_npu.npu_fusion_attention
    with torch.enable_grad():
        probe_q = torch.empty(1, 4, 2, 8, dtype=torch.bfloat16, device="meta", requires_grad=True)
        probe_k = torch.empty(1, 4, 2, 8, dtype=torch.bfloat16, device="meta", requires_grad=True)
        probe_v = torch.empty(1, 4, 2, 8, dtype=torch.bfloat16, device="meta", requires_grad=True)
        try:
            output = original(probe_q, probe_k, probe_v, 2, "BSND", scale=8**-0.5, sparse_mode=0)[0]
            gradients = (
                torch.autograd.grad(output.sum(), (probe_q, probe_k, probe_v), allow_unused=True)
                if output.requires_grad
                else (None, None, None)
            )
        except Exception:  # the vendor kernel may touch the device manager
            output, gradients = None, (None, None, None)
    if all(gradient is not None for gradient in gradients):
        return "native"

    grad_schema = torch.ops.npu.npu_fusion_attention_grad.default._schema
    if [argument.name for argument in grad_schema.arguments][:5] != ["query", "key", "value", "dy", "head_num"]:
        raise RuntimeError(f"Fusion attention backward contract requires review: {grad_schema}")

    # The library must outlive this call: a garbage-collected Library
    # deregisters its impls and the vendor device-touching Meta kernel
    # would win the dispatch again.
    global _NPU_FA_META_LIBRARY
    _NPU_FA_META_LIBRARY = torch.library.Library("npu", "FRAGMENT")
    _NPU_FA_META_LIBRARY.impl("npu_fusion_attention", _npu_fusion_attention_meta, "Meta")
    _NPU_FA_META_LIBRARY.impl("npu_fusion_attention_grad", _npu_fusion_attention_grad_meta, "Meta")

    seam = fusion_attention._encoder_fusion_attention

    @wraps(seam)
    def encoder_fusion_attention(q, k, v, head_num, scale):
        if q.device.type != "meta":
            return seam(q, k, v, head_num, scale)
        return _NpuFusionAttentionMetaAutograd.apply(q, k, v, head_num, scale, original)

    scope.replace(fusion_attention, "_encoder_fusion_attention", encoder_fusion_attention)
    return "bridge"
