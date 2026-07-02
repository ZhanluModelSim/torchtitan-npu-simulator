# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model, LiCompute, SparseAttention
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute, SimNpuSparseAttention


def _build_sim_sparse_attention(B=2, S=3, N=4, D=8, R=4, K=5):
    args = DeepSeekV4Model.Config(n_heads=N, head_dim=D, compress_ratios=(R,), window_size=2, n_layers=1)
    parent = SparseAttention(SparseAttention.Config(layer_id=0, args=args))
    shim = SimNpuSparseAttention(parent)
    tensors = {
        "query_states": torch.randn(B, S, N, D, requires_grad=True),
        "kv_states": torch.randn(B, S, D, requires_grad=True),
        "attn_sink": torch.randn(N, requires_grad=True),
    }
    if R != 1:
        tensors["kv_compress"] = torch.randn(B, S // R, D, requires_grad=True)
    if R == 4:
        tensors["compress_topk_idxs"] = torch.randint(0, S, (B, S, K), dtype=torch.int32)
    return shim, tensors


def test_sim_sparse_attention_forward_returns_correct_shape_r1():
    shim, t = _build_sim_sparse_attention(B=2, S=3, N=4, D=8, R=1)
    y = shim(t["query_states"], t["kv_states"], t["attn_sink"])
    assert y.shape == (2, 3, 4, 8)


def test_sim_sparse_attention_forward_returns_correct_shape_r4():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    y = shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
    assert y.shape == (2, 8, 4, 8)


def test_sim_sparse_attention_records_real_op_names():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    capture = OpDispatchCapture()
    with capture:
        shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "aclnn.npu_sparse_attn_sharedkv_metadata" in raw_names
    assert "aclnn.npu_sparse_attn_sharedkv" in raw_names


def test_sim_sparse_attention_backward_propagates_gradient_and_records_grad_op():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        y = shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
        phase_box["value"] = "backward"
        y.sum().backward()
    assert t["query_states"].grad is not None
    assert t["query_states"].grad.shape == t["query_states"].shape
    assert t["kv_states"].grad is not None
    assert t["kv_states"].grad.shape == t["kv_states"].shape
    assert t["attn_sink"].grad is not None
    assert t["attn_sink"].grad.shape == t["attn_sink"].shape
    assert t["kv_compress"].grad is not None
    assert t["kv_compress"].grad.shape == t["kv_compress"].shape
    nodes = capture.build_nodes()
    bwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "backward"}
    assert "aclnn.npu_sparse_attn_sharedkv_grad" in bwd_names


def test_sim_sparse_attention_works_on_meta_device():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    meta_t = {k: (v.detach().to("meta").requires_grad_(True) if v.dtype != torch.int32 else v.detach().to("meta")) for k, v in t.items()}
    y = shim(meta_t["query_states"], meta_t["kv_states"], meta_t["attn_sink"], meta_t["kv_compress"], meta_t["compress_topk_idxs"])
    assert y.device.type == "meta"
    y.sum().backward()
    assert meta_t["query_states"].grad is not None


def _build_sim_li_compute(B=2, S=8, N_i=4, D_i=8, K=5, ratio=4):
    parent = LiCompute(LiCompute.Config(ratio=ratio, index_topk=K))
    shim = SimNpuLiCompute(parent)
    tensors = {
        "q_indexer": torch.randn(B, S, N_i, D_i, requires_grad=True),
        "k_indexer": torch.randn(B, S // ratio, D_i, requires_grad=True),
        "weights": torch.randn(B, S, N_i, requires_grad=True),
    }
    return shim, tensors


def test_sim_li_compute_forward_returns_correct_shapes():
    shim, t = _build_sim_li_compute(B=2, S=8, N_i=4, D_i=8, K=5, ratio=4)
    compress_topk_idxs, index_score = shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    assert compress_topk_idxs.shape == (2, 8, 5)
    assert index_score.shape == (2, 8, 5)


def test_sim_li_compute_records_real_op_name():
    shim, t = _build_sim_li_compute()
    capture = OpDispatchCapture()
    with capture:
        shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "aclnn.npu_lightning_indexer" in raw_names


def test_sim_li_compute_backward_does_not_raise_and_returns_none_grad():
    shim, t = _build_sim_li_compute()
    compress_topk_idxs, index_score = shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    # index_score is a real (non-detached) autograd node output -- summing and backpropagating
    # through it must not raise, even though the real op has no gradient (mirrors production,
    # where index_score's actual gradient flows through the separate LiLoss path instead).
    index_score.float().sum().backward()
    assert t["q_indexer"].grad is None  # real non-A5 implementation has no gradient for this path
