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

from torchtitan_npu.models.glm5_next.attention import GlmDeltaAttention, GlmDsaAttention
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
    # The qkv short conv is intentionally NOT shimmed (kimi_k3 parity).
    assert not hasattr(attention.conv1d, "_simulator_glm5_next_conv_shim_installed")
    hook.remove()


def test_kda_records_chunk_kda_and_conv_fused_ops():
    attention = _build_kda().to("meta")
    apply_glm5_next_shims(attention)
    x = _meta(1, 8, 32)

    capture = _capture_fwd_bwd(lambda: attention(x))
    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    assert raw_names.count("triton_ascend_kernels.chunk_kda") == 1
    assert raw_names.count("triton_ascend_kernels.chunk_kda_grad") == 1
    # The qkv short conv stays a real aten conv (kimi_k3 parity): no
    # invented causal_conv1d fused op may appear.
    assert raw_names.count("aten.convolution.default") == 1
    assert raw_names.count("aten.convolution_backward.default") == 1
    assert not any("causal_conv1d" in n for n in raw_names)
    # chunk_kda_grad cost-model interface: [q, k, v, g, beta, do]
    grad = next(n for n in nodes if n.annotations["raw_op_type"] == "triton_ascend_kernels.chunk_kda_grad")
    grad_in = [t.shape for t in grad.inputs]
    grad_out = [t.shape for t in grad.outputs]
    assert len(grad_in) == 6
    assert list(grad_in[2]) == [1, 8, 2, 16]  # v [B,S,H,Dv] at index 2
    assert len(grad_out) == 5  # same-shape grads
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
    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    # §3 lightning indexer: fwd only, 3 inputs / 2 outputs, no invented
    # backward op (frozen indexer, no_grad).
    assert raw_names.count("aclnn.npu_lightning_indexer") == 1
    assert raw_names.count("aclnn.npu_lightning_indexer_grad") == 0
    # §2 metadata first, then the 6-input top-k variant.
    assert raw_names.count("aclnn.npu_sparse_attn_sharedkv_metadata") == 1
    assert raw_names.count("aclnn.npu_sparse_attn_sharedkv") == 1
    assert raw_names.count("aclnn.npu_sparse_attn_sharedkv_grad") == 1

    def _node(name):
        return next(n for n in nodes if n.annotations["raw_op_type"] == name)

    def _shapes(node, attr="inputs"):
        return [tuple(t.shape) for t in getattr(node, attr)]

    # input counts / ranks per OP_INTERFACE_REFERENCE.md §2/§3.
    indexer = _node("aclnn.npu_lightning_indexer")
    indexer_in = _shapes(indexer)
    indexer_out = _shapes(indexer, "outputs")
    assert len(indexer_in) == 3
    assert len(indexer_in[0]) == 4  # query_idx [B,S,N_idx,D_idx]
    assert len(indexer_in[1]) == 4  # key_idx [B,cl,1,D_idx]
    assert len(indexer_in[2]) == 3  # weights [B,S,N_idx]
    assert len(indexer_out[0]) == 4 and indexer_out[0][-1] == 2  # K=select_pools

    main = _node("aclnn.npu_sparse_attn_sharedkv")
    main_in = _shapes(main)
    main_out = _shapes(main, "outputs")
    assert len(main_in) == 6  # 6-input top-k variant
    assert len(main_in[1]) == 4  # ori_kv [B,S,1,nh*(k+v)]
    assert main_in[1][-1] == 2 * (8 + 8)  # nh*(k_dim+v_dim)
    assert main_in[3] == (1024,)  # metadata
    assert len(main_in[5]) == 4  # cmp_sparse_indices [B,S,1,K]
    assert len(main_out) == 2 and len(main_out[1]) == 4  # softmax_lse

    grad = _node("aclnn.npu_sparse_attn_sharedkv_grad")
    grad_in = _shapes(grad)
    grad_out = _shapes(grad, "outputs")
    assert len(grad_in) == 7  # q, ori_kv, result, lse, sinks, do, cmp_kv
    assert len(grad_out) == 4  # d_query, d_ori_kv, d_sinks, d_cmp_kv


def test_conv_is_not_shimmed_and_tp_slice_still_applies():
    from torchtitan_npu.models.glm5_next.attention import ShortConv1d

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
    assert raw_names.count("aten.convolution.default") == 1
    assert not any("causal_conv1d" in n for n in raw_names)
    assert output.shape == x.shape
