# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_npu
from torchtitan_simulator import meta_env
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.runtime import meta_runtime


@pytest.mark.parametrize("mode", ["half", "interleave"])
def test_runtime_rotary_preserves_input_grad_and_saved_identity(mode):
    original = torch_npu.npu_rotary_mul
    x = torch.empty(1, 4, 2, 8, device="meta", requires_grad=True)
    cos = torch.empty(1, 4, 1, 8, device="meta")
    sin = torch.empty_like(cos)
    capture = OpDispatchCapture()

    with meta_runtime(), capture, AutogradSavedTensorCapture():
        value = x.sin()
        output = torch_npu.npu_rotary_mul(value, cos, sin, rotary_mode=mode)
        output.sum().backward()
    capture.finalize_autograd_saved_tensors()
    nodes = capture.build_nodes()
    forward = next(n for n in nodes.values() if n.annotations["raw_op_type"] == "npu.npu_rotary_mul.default")
    backward = next(n for n in nodes.values() if n.annotations["raw_op_type"] == "npu.npu_rotary_mul_backward.default")
    source = next(n for n in nodes.values() if n.annotations["raw_op_type"] == "aten.sin.default")

    assert x.grad is not None and x.grad.shape == x.shape
    assert source.op_id in forward.predecessors
    pending = list(backward.predecessors)
    ancestors = set()
    while pending:
        op_id = pending.pop()
        if op_id not in ancestors:
            ancestors.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    assert source.op_id in ancestors
    assert capture.autograd_saved_tensor_events()
    assert torch_npu.npu_rotary_mul is original


def test_runtime_restores_bindings_and_autograd_tls_after_exception():
    original = torch_npu.npu_rotary_mul
    multithreading = torch.autograd.is_multithreading_enabled()
    original_fsdp = dict(meta_env._fsdp_state)
    original_layer = meta_env._comm_layer
    original_inference = meta_env._in_metadata_inference

    with pytest.raises(ValueError, match="capture failure"), meta_runtime():
        assert not torch.autograd.is_multithreading_enabled()
        meta_env._fsdp_state[99] = "UNSHARDED"
        meta_env._comm_layer = "L2"
        meta_env._in_metadata_inference = True
        raise ValueError("capture failure")

    assert torch_npu.npu_rotary_mul is original
    assert torch.autograd.is_multithreading_enabled() == multithreading
    assert not meta_env._is_meta_simulation
    assert meta_env._fsdp_state == original_fsdp
    assert meta_env._comm_layer == original_layer
    assert meta_env._in_metadata_inference == original_inference
