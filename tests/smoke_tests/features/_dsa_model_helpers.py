# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for DSA-related smoke tests.

Builds a minimal, single-layer v32 Attention plus the auxiliary tensors
its sparse kernels expect (``query_indexer`` / ``key_indexer`` / ``weights``
/ ``topk_indices`` / softmax stats) so individual op-level tests can exercise
the Lightning Indexer and sparse flash attention in isolation.

The helper used to depend on the legacy flat ``DeepSeekV32ModelArgs`` +
upstream ``precompute_freqs_cis``. Both are gone in the new Config tree;
this version drives everything from ``Attention.Config`` (built via
``_make_dsv32_attn_config``) and a stand-alone ``RoPE`` instance for the
freqs_cis cache.
"""
from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.models.common.rope import RoPE

from torchtitan_npu.models.deepseek_v32 import _make_dsv32_attn_config
from torchtitan_npu.models.deepseek_v32.model import apply_rotary_emb, Attention


@dataclass
class AttentionTensorState:
    x: torch.Tensor
    freqs_cis: torch.Tensor
    qr: torch.Tensor
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    kv: torch.Tensor
    k_pe: torch.Tensor


@dataclass
class DsaKernelInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    query_indexer: torch.Tensor
    weights: torch.Tensor
    key_indexer: torch.Tensor
    topk_indices: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor


def _build_attn_config(seq_len: int):
    """Build a v32 ``Attention.Config`` sized to match the legacy DSA smoke args."""
    cfg = _make_dsv32_attn_config(
        layer_id=0,
        dim=256,
        n_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    # Push the rope cache out far enough to cover the requested ``seq_len``.
    cfg.rope_max_seq_len = max(cfg.rope_max_seq_len, seq_len)
    return cfg


def _build_freqs_cis(cfg, device) -> torch.Tensor:
    """Replicates the old ``precompute_freqs_cis`` helper for the rope range
    used by v32 attention; returns a host tensor on ``device``."""
    rope = RoPE.Config(
        dim=cfg.qk_rope_head_dim,
        max_seq_len=cfg.rope_max_seq_len,
        theta=10000.0,
        backend="complex",
    ).build()
    rope.init_states(buffer_device=device)
    return rope.cache.to(device)


def _build_attention_tensors(attention, cfg, batch_size, seq_len, device):
    pre = attention.pre_attention
    x = torch.zeros(batch_size, seq_len, cfg.dim, dtype=torch.bfloat16, device=device)
    freqs_cis = _build_freqs_cis(cfg, device)[:seq_len]
    qr = pre.q_norm(pre.wq_a(x))
    q = pre.wq_b(qr).view(batch_size, seq_len, -1, pre.qk_head_dim)
    q_nope, q_pe = torch.split(q, [pre.qk_nope_head_dim, pre.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    kv = pre.wkv_a(x)
    kv, k_pe = torch.split(kv, [pre.kv_lora_rank, pre.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    return AttentionTensorState(
        x=x,
        freqs_cis=freqs_cis,
        qr=qr,
        q_nope=q_nope,
        q_pe=q_pe,
        kv=pre.kv_norm(kv),
        k_pe=k_pe,
    )


def _build_query_key_tensors(attention, attention_state):
    pre = attention.pre_attention
    wkv_b_weight = pre.wkv_b.weight.reshape(
        -1,
        pre.qk_nope_head_dim + pre.v_head_dim,
        pre.kv_lora_rank,
    )
    w_uk = wkv_b_weight[:, : pre.qk_nope_head_dim, :]
    query = torch.einsum("bshq,hqr->bshr", attention_state.q_nope, w_uk)
    key = attention_state.kv.unsqueeze(2)
    value = attention_state.kv.unsqueeze(2)
    if query.shape[2] not in (64, 128):
        raise AssertionError(
            f"DSA helper produced invalid query head count: {query.shape}"
        )
    return query, key, value


def _build_indexer_outputs(attention, attention_state):
    pre = attention.pre_attention
    query_indexer, weights, key_indexer, _ = pre.indexer(
        attention_state.x.detach(),
        attention_state.qr.detach(),
        0,
        attention_state.freqs_cis,
        None,
    )
    topk_indices, _ = torch_npu.npu_lightning_indexer(
        query_indexer.detach(),
        key_indexer.detach(),
        weights.detach(),
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=pre.indexer.index_topk,
        sparse_mode=3,
        return_value=True,
    )
    return DsaKernelInputs(
        query=None,
        key=None,
        value=None,
        query_indexer=query_indexer,
        weights=weights,
        key_indexer=key_indexer,
        topk_indices=topk_indices.to(torch.int32),
        query_rope=None,
        key_rope=None,
    )


def _build_softmax_stats(attention, kernel_inputs, batch_size, seq_len):
    actual_seq_len = torch.full(
        (batch_size,),
        seq_len,
        dtype=torch.int32,
        device=kernel_inputs.query.device,
    )
    _, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        kernel_inputs.query.detach(),
        kernel_inputs.key.detach(),
        kernel_inputs.value.detach(),
        sparse_indices=kernel_inputs.topk_indices,
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len,
        query_rope=kernel_inputs.query_rope.detach(),
        key_rope=kernel_inputs.key_rope.detach(),
        scale_value=attention.pre_attention.softmax_scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return softmax_max.detach(), softmax_sum.detach()


def run_lightning_indexer_smoke(npu_device, *, batch_size=1, seq_len=128):
    q_indexer = torch.randn(
        batch_size, seq_len, 64, 128, dtype=torch.bfloat16, device=npu_device
    )
    k_indexer = torch.randn(
        batch_size, seq_len, 1, 128, dtype=torch.bfloat16, device=npu_device
    )
    weights = torch.randn(
        batch_size, seq_len, 64, dtype=torch.bfloat16, device=npu_device
    )
    return torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=16,
        sparse_mode=3,
        return_value=True,
    )


def build_model_backed_dsa_inputs(
    device, *, batch_size=1, seq_len=2048, requires_grad=False
):
    cfg = _build_attn_config(seq_len)
    # ``num_total_layers=1`` keeps PostAttention's loss tracker sized for a
    # single layer (the helper exercises one Attention in isolation).
    attention = Attention(cfg, num_total_layers=1).to(
        device=device, dtype=torch.bfloat16
    )
    attention.eval()
    attention_state = _build_attention_tensors(
        attention, cfg, batch_size, seq_len, device
    )
    query, key, value = _build_query_key_tensors(attention, attention_state)
    kernel_inputs = _build_indexer_outputs(attention, attention_state)
    kernel_inputs.query = query
    kernel_inputs.key = key
    kernel_inputs.value = value
    kernel_inputs.query_rope = attention_state.q_pe
    kernel_inputs.key_rope = attention_state.k_pe
    softmax_max, softmax_sum = _build_softmax_stats(
        attention, kernel_inputs, batch_size, seq_len
    )

    query_indexer = kernel_inputs.query_indexer.detach()
    key_indexer = kernel_inputs.key_indexer.detach()
    weights = kernel_inputs.weights.detach()
    if requires_grad:
        query_indexer.requires_grad_()
        key_indexer.requires_grad_()
        weights.requires_grad_()

    return {
        "query": query.detach(),
        "key": key.detach(),
        "query_indexer": query_indexer,
        "key_indexer": key_indexer,
        "weights": weights,
        "topk_indices": kernel_inputs.topk_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "query_rope": attention_state.q_pe.detach(),
        "key_rope": attention_state.k_pe.detach(),
    }
