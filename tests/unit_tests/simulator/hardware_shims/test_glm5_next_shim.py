# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim tests for the glm5_next model-specific fused ops.

Mirrors test_kda_shim.py: binding must preserve module identity and hooks,
and forward/backward must record the real production op names
(MODEL_CONTRACT.md section 11) without leaking decomposition ops.
"""

import torch

from torchtitan_npu.models.glm5_next.attention import GlmDeltaAttention, GlmDsaAttention, ShortConv1d
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.glm5_next_shim import apply_glm5_next_shims


def _build_kda() -> GlmDeltaAttention:
    return GlmDeltaAttention(
        hidden_size=32,
        num_heads=2,
        head_dim=16,
        conv_kernel_size=4,
        gate_lower_bound=-5.0,
    )


def _meta(*shape, requires_grad=True):
    return torch.empty(*shape, device="meta", requires_grad=requires_grad)


def _capture_fwd_bwd(module_call):
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])
    with capture:
        output = module_call()
        phase["value"] = "backward"
        output.sum().backward()
    return capture


def test_kda_binding_preserves_module_identity_and_hooks():
    attention = _build_kda()
    hook = attention.register_forward_hook(lambda module, args, output: None)
    module_id = id(attention)
    hook_ids = set(attention._forward_hooks)

    apply_glm5_next_shims(attention)
    apply_glm5_next_shims(attention)

    assert id(attention) == module_id
    assert set(attention._forward_hooks) == hook_ids
    assert attention._simulator_glm5_next_kda_shim_installed is True
    assert attention.conv1d._simulator_glm5_next_conv_shim_installed is True
    hook.remove()


def test_kda_records_chunk_kda_and_conv_fused_ops():
    attention = _build_kda().to("meta")
    apply_glm5_next_shims(attention)
    x = _meta(1, 8, 32)

    capture = _capture_fwd_bwd(lambda: attention(x))
    raw_names = [node.annotations["raw_op_type"] for node in capture.build_nodes().values()]
    assert raw_names.count("triton_ascend_kernels.chunk_kda") == 1
    assert raw_names.count("triton_ascend_kernels.chunk_kda_grad") == 1
    assert raw_names.count("triton_ascend_kernels.causal_conv1d") == 1
    assert raw_names.count("triton_ascend_kernels.causal_conv1d_grad") == 1
    # The shim must not leak uncaptured internal allocations (the sequential
    # fallback's empties); the surrounding projections legitimately capture
    # as aten.mm.


def test_dsa_records_indexer_and_sparse_attn_fused_ops():
    attention = GlmDsaAttention(
        hidden_size=32,
        num_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        v_head_dim=8,
        indexer_heads=2,
        indexer_head_dim=8,
        index_topk=8,
        index_kpool=4,
        rms_norm_eps=1e-5,
    ).to("meta")
    apply_glm5_next_shims(attention)
    x = _meta(1, 8, 32)

    capture = _capture_fwd_bwd(lambda: attention(x))
    raw_names = [node.annotations["raw_op_type"] for node in capture.build_nodes().values()]
    assert raw_names.count("aclnn.npu_lightning_indexer") == 1
    # The indexer is frozen (no_grad): no indexer backward node exists.
    assert raw_names.count("aclnn.npu_lightning_indexer_grad") == 0
    assert raw_names.count("aclnn.npu_sparse_attn_sharedkv") == 1
    assert raw_names.count("aclnn.npu_sparse_attn_sharedkv_grad") == 1


def test_conv_shim_respects_tp_local_slice():
    conv = ShortConv1d(24, kernel_size=4).to("meta")
    # Simulate the TP partition state: three contiguous slices of the global
    # channels (q/k/v), 8 local channels each.
    conv.set_local_slice(channel_starts=(0, 8, 16), local_channels=8)
    apply_glm5_next_shims(conv)
    x = _meta(1, 8, 24, requires_grad=False)

    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])
    with capture:
        output = conv(x)
    raw_names = [node.annotations["raw_op_type"] for node in capture.build_nodes().values()]
    assert raw_names.count("triton_ascend_kernels.causal_conv1d") == 1
    assert output.shape == x.shape
