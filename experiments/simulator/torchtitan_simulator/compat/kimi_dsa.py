# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture Kimi K3.1 absorbed SFA and its independent learned indexer."""

import torch
from torch.nn import functional as F

from torchtitan_simulator.capture.dispatch_capture import capture_attention_geometry
from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)


@torch.library.custom_op("kimi_sim::dsa_indexer", mutates_args=())
def dsa_indexer(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor, topk: int) -> torch.Tensor:
    raise RuntimeError("Kimi virtual DSA indexer is available only for Meta simulation")


@dsa_indexer.register_fake
def _indexer_meta(q, k, weights, topk):
    if q.ndim != 4 or k.shape != (q.shape[0], q.shape[1], q.shape[3]):
        raise ValueError("DSA indexer requires B/S/H/D query and B/S/D key")
    if weights.shape != q.shape[:-1] or weights.dtype != torch.float32 or topk < 1:
        raise ValueError("DSA indexer requires scaled FP32 weights and positive top-k")
    if min(q.shape) <= 0 or q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype != q.dtype:
        raise ValueError("DSA indexer requires positive floating-point query/key geometry")
    if len({t.device for t in (q, k, weights)}) != 1:
        raise ValueError("DSA indexer operands must share placement")
    return torch.empty((*q.shape[:2], min(q.shape[1], topk)), dtype=torch.int64, device=q.device)


_INDICES = TensorMeaning("selected_indices", ("batch", "q_len", "selected_key"))
register_tensor_semantics(
    str(dsa_indexer._opoverload),
    OperatorTensorSemantics(
        (
            ("q", TensorMeaning("query", ("batch", "sequence", "index_head", "index_head_dim"))),
            ("k", TensorMeaning("key", ("batch", "sequence", "index_head_dim"))),
            ("weights", TensorMeaning("index_weights", ("batch", "sequence", "index_head"))),
        ),
        (_INDICES,),
    ),
)


class _NativeSparseMetaAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, qr, kr, indices, lengths, scale, facts):
        with capture_attention_geometry("npu.npu_sparse_flash_attention.default", **facts):
            out, maximum, normalizer = torch.ops.npu.npu_sparse_flash_attention(
                q,
                kv,
                kv,
                indices,
                scale,
                actual_seq_lengths_query=lengths,
                actual_seq_lengths_kv=lengths,
                query_rope=qr,
                key_rope=kr,
                attention_mode=2,
                return_softmax_lse=True,
            )
        ctx.save_for_backward(q, kv, qr, kr, indices, lengths, out, maximum, normalizer)
        ctx.scale = scale
        ctx.facts = facts
        return out

    @staticmethod
    def backward(ctx, dy):
        q, kv, qr, kr, indices, lengths, out, maximum, normalizer = ctx.saved_tensors
        with capture_attention_geometry("npu.npu_sparse_flash_attention_grad.default", **ctx.facts):
            dq, dk, dv, dqr, dkr = torch.ops.npu.npu_sparse_flash_attention_grad(
                q,
                kv,
                kv,
                indices,
                dy.contiguous(),
                out,
                maximum,
                normalizer,
                ctx.scale,
                1,
                query_rope=qr,
                key_rope=kr,
                actual_seq_qlen=lengths,
                actual_seq_kvlen=lengths,
                attention_mode=2,
            )
        return dq, dk + dv, dqr, dkr, None, None, None, None


def _native_sparse(q, kv, qr, kr, indices, scale, topk, source):
    batch, seq, heads, _ = q.shape
    padded_heads = 1 << (heads - 1).bit_length()
    width = max(1024, 1 << (indices.shape[-1] - 1).bit_length())
    indices = F.pad(indices.to(torch.int32).unsqueeze(2), (0, width - indices.shape[-1]), value=-1)
    if padded_heads != heads:
        q = F.pad(q, (0, 0, 0, padded_heads - heads))
        qr = F.pad(qr, (0, 0, 0, padded_heads - heads))
    lengths = torch.full((batch,), seq, dtype=torch.int32, device=q.device)
    per_request = sum(min(i, topk) for i in range(1, seq + 1))
    facts = {
        "query_lens": (seq,) * batch,
        "kv_lens": (seq,) * batch,
        "dense_qk_pairs": batch * seq * seq,
        "effective_qk_pairs": batch * per_request,
        "selected_pairs_by_request": (per_request,) * batch,
        "pair_count_source": source,
        "logical_q_heads": heads,
        "shared_key_value": True,
    }
    output = _NativeSparseMetaAutograd.apply(
        q.contiguous(),
        kv.contiguous(),
        qr.contiguous(),
        kr.contiguous(),
        indices.contiguous(),
        lengths,
        scale,
        facts,
    )
    return output[:, :, :heads]


def install_kimi_dsa_native_capture(scope, *, short_sparse: bool = False):
    """Use the native absorbed SFA boundary for Meta training captures."""
    import spmd_types

    from torchtitan_npu.models.kimi_k3_1.attention import KimiGatedDSA
    from torchtitan_npu.models.kimi_k3_1.dsa import KimiDSAIndexer

    spmd_types.register_local_autograd_function(_NativeSparseMetaAutograd)
    original = KimiGatedDSA._project_attention

    def select(module, queries, keys, weights):
        return dsa_indexer(queries, keys, weights, module.topk)

    def project(module, hidden, query_latent, q_nope, q_rope, latent, k_rope):
        batch, seq = hidden.shape[:2]
        topk = module.indexer.topk
        if seq <= topk and not short_sparse:
            return original(module, hidden, query_latent, q_nope, q_rope, latent, k_rope)
        if seq <= topk:
            positions = torch.arange(seq, device=hidden.device, dtype=torch.int32)
            indices = positions.expand(batch, seq, seq).masked_fill(positions > positions[:, None], -1)
            source = "deterministic_causal_all_keys"
        else:
            indices = module.indexer(hidden, query_latent)
            source = "causal_learned_topk"
        return module._absorbed_attention(
            q_nope,
            q_rope,
            latent,
            k_rope,
            indices,
            lambda q, kv, qr, kr, idx: _native_sparse(q, kv, qr, kr, idx, module.scaling, topk, source),
        )

    scope.replace(KimiDSAIndexer, "_select", select)
    scope.replace(KimiGatedDSA, "_project_attention", project)
