# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.deepseek_v4.model import HcPre, HcHead
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcPre, SimHcHead


def _build_sim_hc_pre(n: int = 4, D: int = 8) -> tuple[SimHcPre, dict]:
    parent = HcPre(HcPre.Config(hc_mult=n, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6))
    shim = SimHcPre(parent)
    total = n * n + 2 * n
    tensors = {
        "x": torch.randn(2, 3, n * D, requires_grad=True),
        "hc_fn": torch.randn(total, n * D, requires_grad=True),
        "hc_scale": torch.randn(3, requires_grad=True),
        "hc_base": torch.randn(total, requires_grad=True),
    }
    return shim, tensors


def test_sim_hc_pre_forward_returns_correct_shapes():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    assert y.shape == (2, 3, 8)  # [B,S,D]
    assert h_post.shape == (2, 3, 4)  # [B,S,n]
    assert h_res.shape == (2, 3, 4, 4)  # [B,S,n,n]


def test_sim_hc_pre_records_real_op_names_in_active_capture():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    capture = OpDispatchCapture()
    with capture:
        shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "npu.npu_rms_norm.default" in raw_names
    assert "aten.matmul.default" in raw_names
    assert "triton.hc_pre_fwd" in raw_names
    assert "triton.hc_pre_bmm_forward" in raw_names


def test_sim_hc_pre_backward_propagates_gradient_to_input():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    (y.sum() + h_post.sum() + h_res.sum()).backward()
    assert t["x"].grad is not None
    assert t["x"].grad.shape == t["x"].shape
    assert t["hc_fn"].grad is not None
    assert t["hc_scale"].grad is not None
    assert t["hc_base"].grad is not None


def test_sim_hc_pre_records_backward_op_names_only_during_backward_phase():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
        phase_box["value"] = "backward"
        (y.sum() + h_post.sum() + h_res.sum()).backward()
    nodes = capture.build_nodes()
    bwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "backward"}
    assert "triton.hc_pre_bwd" in bwd_names
    assert "triton.hc_pre_bmm_backward" in bwd_names
    fwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "forward"}
    assert "triton.hc_pre_fwd" in fwd_names


def test_sim_hc_pre_works_on_meta_device():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    meta_tensors = {k: v.detach().to("meta").requires_grad_(True) for k, v in t.items()}
    y, h_post, h_res = shim(meta_tensors["x"], meta_tensors["hc_fn"], meta_tensors["hc_scale"], meta_tensors["hc_base"])
    assert y.device.type == "meta"
    (y.sum() + h_post.sum() + h_res.sum()).backward()
    assert meta_tensors["x"].grad is not None


def _build_sim_hc_head(n: int = 4, D: int = 8) -> tuple["SimHcHead", dict]:
    parent = HcHead(HcHead.Config(norm_eps=1e-6, hc_eps=1e-6, hc_mult=n, dim=D))
    shim = SimHcHead(parent)
    tensors = {"x": torch.randn(2, 3, n, D, requires_grad=True)}
    return shim, tensors


def test_sim_hc_head_forward_returns_correct_shape():
    shim, t = _build_sim_hc_head(n=4, D=8)
    y = shim(t["x"])
    assert y.shape == (2, 3, 8)  # [B,S,D]


def test_sim_hc_head_records_real_op_names():
    shim, t = _build_sim_hc_head(n=4, D=8)
    capture = OpDispatchCapture()
    with capture:
        shim(t["x"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "triton.hc_pre_only_fwd" in raw_names
    assert "triton.hc_pre_bmm_forward" in raw_names


def test_sim_hc_head_backward_propagates_gradient():
    shim, t = _build_sim_hc_head(n=4, D=8)
    y = shim(t["x"])
    y.sum().backward()
    assert t["x"].grad is not None
    assert t["x"].grad.shape == t["x"].shape
