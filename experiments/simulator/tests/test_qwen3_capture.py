# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _all_nodes(workload):
    return [node for template in workload.step_templates.values() for node in template.nodes.values()]


def test_native_qwen3_capture_keeps_real_fused_attention_and_full_gradients():
    workload = simulate(SimulationConfig(config="qwen3_32b_smoketest", seq_len=128))
    validation = validate_workload(workload)
    assert validation["graph_acyclic"] and validation["peak_storage_unique"]

    annotations = workload.iteration.schedule.annotations
    gradients = annotations["gradient_validation"]
    # Four-layer slice of the real 32B geometry: 9 trainable parameters per
    # layer plus the untied head, final norm and token embedding.
    expected = {
        f"layers.{layer}.{name}"
        for layer in range(4)
        for name in (
            "attention.q_norm.weight",
            "attention.k_norm.weight",
            "attention.qkv_linear.wqkv.weight",
            "attention.wo.weight",
            "attention_norm.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
            "ffn_norm.weight",
        )
    } | {"tok_embeddings.weight", "norm.weight", "lm_head.weight"}
    assert set(gradients) == expected
    assert all(entry["dtype"] == "torch.float32" for entry in gradients.values())

    assert all(template.is_acyclic for template in workload.step_templates.values())
    nodes = _all_nodes(workload)
    assert all(
        predecessor in {node.op_id for node in nodes}
        for node in nodes
        for predecessor in node.predecessors
    )

    op_types = Counter(node.annotations["raw_op_type"] for node in nodes)
    # The delivered graph keeps the real NPU kernel boundaries: one fused
    # attention per layer in forward and one gradient kernel per layer in
    # backward, never the generic _flash_attention_forward aten path.
    assert op_types["npu.npu_fusion_attention_v3.default"] == 4
    assert op_types["npu.npu_fusion_attention_grad_v3.default"] == 4
    assert not any("flash_attention_forward" in op_type for op_type in op_types)
    # Fused NPU norms: 4 layers x (attention_norm, ffn_norm, q_norm, k_norm)
    # plus the final model norm, each with a native backward kernel.
    assert op_types["npu.npu_rms_norm.default"] == 17
    assert op_types["npu.npu_rms_norm_backward.default"] == 17

    # Optimizer contract: one fused AdamW step covers exactly the gradient set.
    fused = [node for node in nodes if node.annotations["raw_op_type"] == "aten._fused_adamw_.default"]
    assert sum(node.annotations["optimizer_updated_parameter_count"] for node in fused) == len(gradients)
    expected_shapes = Counter(tuple(item["original_shape"]) for item in annotations["parameter_layouts"].values())
    assert (
        Counter(tuple(shape) for node in fused for shape in node.annotations["optimizer_logical_parameter_shapes"])
        == expected_shapes
    )

    assert annotations["memory_plan"] is not None
    assert annotations["memory_summary"] is not None
    assert annotations["fsdp_communication_view"] is not None
    # Dense decoders take the no-router early return in the balanced dispatch
    # installer; lock the capability marker against accidental regressions.
    assert annotations["runtime_capabilities"]["ep_token_exchange"] == "not-applicable-dense"
    assert annotations["runtime_capabilities"]["ep_token_dispatcher_classes"] == 0
