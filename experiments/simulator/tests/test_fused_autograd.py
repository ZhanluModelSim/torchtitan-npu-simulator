# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch_npu
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.norm import install_meta_norm_autograd
from torchtitan_simulator.compat.patches import PatchScope
from torchtitan_simulator.compat.permute import install_meta_permute_autograd


def test_rms_norm_uses_actual_input_scale_and_rstd_in_native_backward():
    original = torch_npu.npu_rms_norm
    x = torch.empty(2, 8, device="meta", requires_grad=True)
    gamma = torch.empty(8, device="meta", requires_grad=True)
    saved = []
    capture = OpDispatchCapture()
    with PatchScope() as scope:
        install_meta_norm_autograd(scope)
        with capture, torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            value = x.sin()
            scale = gamma.sin()
            output, rstd = torch_npu.npu_rms_norm(value, scale)
            output.sum().backward()
    assert x.grad is not None and gamma.grad is not None
    assert any(tensor is value for tensor in saved)
    assert any(tensor is scale for tensor in saved)
    assert any(tensor is rstd for tensor in saved)
    nodes = capture.build_nodes()
    assert any(node.annotations["raw_op_type"] == "npu.npu_rms_norm_backward.default" for node in nodes.values())
    assert torch_npu.npu_rms_norm is original


def test_token_permute_saves_actual_tokens_indices_and_inverse_permutation():
    original = torch_npu.npu_moe_token_permute
    x = torch.empty(4, 8, device="meta", requires_grad=True)
    indices = torch.empty(4, 2, device="meta", dtype=torch.int32)
    saved = []
    capture = OpDispatchCapture()
    with PatchScope() as scope:
        install_meta_permute_autograd(scope)
        with capture, torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            tokens = x.sin()
            output, inverse = torch_npu.npu_moe_token_permute(tokens, indices)
            output.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    for value in (tokens, indices, inverse):
        assert any(tensor is value for tensor in saved)
    assert any(
        node.annotations["raw_op_type"] == "npu.npu_moe_token_permute_grad.default"
        for node in capture.build_nodes().values()
    )
    assert torch_npu.npu_moe_token_permute is original
