# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.kda_converter import apply_kda_shims


def _build_attention() -> KimiDeltaAttention:
    return KimiDeltaAttention(
        KimiDeltaAttention.Config(dim=32, num_heads=2, head_dim=16)
    )


def test_kda_binding_preserves_module_identity_and_hooks():
    attention = _build_attention()
    hook = attention.register_forward_hook(lambda module, args, output: None)
    module_id = id(attention)
    hook_ids = set(attention._forward_hooks)

    apply_kda_shims(attention)
    apply_kda_shims(attention)

    assert id(attention) == module_id
    assert set(attention._forward_hooks) == hook_ids
    assert attention._simulator_kda_shim_installed is True
    hook.remove()


def test_kda_core_records_only_fused_forward_and_backward_nodes():
    attention = _build_attention()
    apply_kda_shims(attention)
    tensors = [
        torch.empty(
            1,
            8,
            2,
            16,
            device="meta",
            requires_grad=True,
        )
        for _ in range(5)
    ]
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = attention._chunk_kda(*tensors)
        phase["value"] = "backward"
        output.sum().backward()

    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    assert raw_names.count("triton_ascend_kernels.chunk_kda") == 1
    assert raw_names.count("triton_ascend_kernels.chunk_kda_grad") == 1
    assert "aten.empty_like.default" not in raw_names
    assert output.shape == tensors[2].shape
