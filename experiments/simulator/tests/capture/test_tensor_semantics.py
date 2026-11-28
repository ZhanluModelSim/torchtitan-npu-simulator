# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom fusion roles survive dispatch and retain their declared axis meaning."""

import json

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.kimi import chunk_kda, fusion_attention, situ_glu


def test_attention_forward_backward_expose_roles_and_distinct_sequence_axes():
    query = torch.empty(2, 3, 5, 7, device="meta", requires_grad=True)
    key = torch.empty(2, 3, 11, 7, device="meta", requires_grad=True)
    value = torch.empty_like(key, requires_grad=True)
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        output = fusion_attention(query, key, value, 0.5)
        output.backward(torch.empty_like(output))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    forward = nodes["kimi_sim.fusion_attention.default"]
    backward = nodes["kimi_sim.fusion_attention_grad.default"]
    expected_inputs = (
        {"index": 0, "role": "query", "axes": ("batch", "head", "q_len", "qk_head_dim")},
        {"index": 1, "role": "key", "axes": ("batch", "head", "kv_len", "qk_head_dim")},
        {"index": 2, "role": "value", "axes": ("batch", "head", "kv_len", "padded_value_head_dim")},
    )
    assert forward.attrs["tensor_semantics"] == {
        "version": 1,
        "inputs": expected_inputs,
        "outputs": (
            {"index": 0, "role": "attention_output", "axes": ("batch", "head", "q_len", "padded_value_head_dim")},
        ),
    }
    assert forward.inputs[0].shape == (2, 3, 5, 7)
    assert forward.inputs[1].shape == (2, 3, 11, 7)
    assert backward.attrs["tensor_semantics"]["inputs"] == (
        *expected_inputs,
        {"index": 3, "role": "output_gradient", "axes": ("batch", "head", "q_len", "padded_value_head_dim")},
    )
    assert [item["role"] for item in backward.attrs["tensor_semantics"]["outputs"]] == [
        "query_gradient",
        "key_gradient",
        "value_gradient",
    ]
    assert [item["axes"] for item in backward.attrs["tensor_semantics"]["outputs"]] == [
        ("batch", "head", "q_len", "qk_head_dim"),
        ("batch", "head", "kv_len", "qk_head_dim"),
        ("batch", "head", "kv_len", "padded_value_head_dim"),
    ]
    assert all("tensor_semantics" not in node.attrs for raw, node in nodes.items() if raw.startswith("aten."))
    json.dumps(forward.attrs["tensor_semantics"], allow_nan=False)


def test_kda_backward_retains_parameter_axes_and_flattened_bias_axis():
    shapes = [(2, 5, 3, 7)] * 4 + [(2, 5, 3), (3,), (21,)]
    inputs = [torch.empty(shape, device="meta", requires_grad=True) for shape in shapes]
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        output = chunk_kda(*inputs, -5.0)
        output.backward(torch.empty_like(output))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    forward = nodes["kimi_sim.chunk_kda.default"].attrs["tensor_semantics"]
    backward = nodes["kimi_sim.chunk_kda_grad.default"].attrs["tensor_semantics"]
    assert [(item["role"], item["axes"]) for item in forward["inputs"]] == [
        ("query", ("batch", "sequence", "head", "key_head_dim")),
        ("key", ("batch", "sequence", "head", "key_head_dim")),
        ("value", ("batch", "sequence", "head", "value_head_dim")),
        ("decay_gate", ("batch", "sequence", "head", "key_head_dim")),
        ("update_gate", ("batch", "sequence", "head")),
        ("log_decay_parameter", ("head",)),
        ("time_step_bias", (("head", "key_head_dim"),)),
    ]
    assert backward["outputs"][5] == {"index": 5, "role": "log_decay_parameter_gradient", "axes": ("head",)}
    assert backward["outputs"][6] == {
        "index": 6,
        "role": "time_step_bias_gradient",
        "axes": (("head", "key_head_dim"),),
    }
    assert backward["inputs"][-1] == {
        "index": 7,
        "role": "output_gradient",
        "axes": ("batch", "sequence", "head", "value_head_dim"),
    }
    assert all(
        value.grad is not None and value.grad.shape == shape for value, shape in zip(inputs, shapes, strict=True)
    )


@pytest.mark.parametrize(
    "shape", [(7,), (5, 7), (2, 5, 7), (2, 3, 5, 7)], ids=["vector", "rows", "dense", "higher-rank"]
)
def test_situ_keeps_leading_axes_unknown_for_dense_and_routed_inputs(shape):
    gate = torch.empty(shape, device="meta", requires_grad=True)
    up = torch.empty_like(gate, requires_grad=True)
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        output = situ_glu(gate, up, 4.0, 25.0)
        output.backward(torch.empty_like(output))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    forward = nodes["kimi_sim.situ_glu.default"].attrs["tensor_semantics"]
    backward = nodes["kimi_sim.situ_glu_backward.default"].attrs["tensor_semantics"]
    assert [item["role"] for item in forward["inputs"]] == ["gate", "up"]
    assert [item["role"] for item in backward["outputs"]] == ["gate_gradient", "up_gradient"]
    for semantics in (forward, backward):
        for item in (*semantics["inputs"], *semantics["outputs"]):
            assert item["axes"] == (None,) * (len(shape) - 1) + ("feature",)


def test_roles_follow_keyword_occurrences_even_when_the_same_tensor_is_reused():
    tensor = torch.empty(2, 3, 5, 7, device="meta")
    capture = OpDispatchCapture(record_memory=False)

    capture.__torch_dispatch__(
        torch.ops.kimi_sim.fusion_attention.default, (), (), {"v": tensor, "q": tensor, "k": tensor, "scale": 0.5}
    )

    (node,) = capture.build_nodes().values()
    assert [(item["index"], item["role"]) for item in node.attrs["tensor_semantics"]["inputs"]] == [
        (0, "value"),
        (1, "query"),
        (2, "key"),
    ]
    assert len(node.inputs) == 3


def test_declared_attention_rank_mismatch_is_not_silently_labelled():
    tensor = torch.empty(2, 3, 7, device="meta")
    with OpDispatchCapture(record_memory=False), pytest.raises(ValueError, match=r"Tensor role 'query'.*got shape"):
        fusion_attention(tensor, tensor, tensor, 0.5)


@pytest.mark.parametrize("with_scales", [False, True], ids=["unweighted", "weighted"])
def test_rerouting_semantics_preserve_optional_scale_input_and_real_empty_output(with_scales):
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.ops.ascendc.moe_re_routing import npu_moe_re_routing

    tokens = torch.empty(6, 7, device="meta", requires_grad=True)
    counts = torch.empty(2, 3, device="meta", dtype=torch.int64)
    scales = torch.empty(6, device="meta", requires_grad=True) if with_scales else None
    capture = OpDispatchCapture(record_memory=False)

    with meta_runtime(), capture:
        result = npu_moe_re_routing(tokens, counts, scales)
        result[0].backward(torch.empty_like(result[0]))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    node = nodes["torchtitan_npu.npu_moe_re_routing.default"]
    semantics = node.attrs["tensor_semantics"]
    expected_inputs = [
        {"index": 0, "role": "rank_major_tokens", "axes": ("rank_major_row", "feature")},
        {"index": 1, "role": "expert_token_counts", "axes": ("ep_rank", "local_expert")},
    ]
    if with_scales:
        expected_inputs.append({"index": 2, "role": "rank_major_scales", "axes": ("rank_major_row",)})
    assert semantics["inputs"] == tuple(expected_inputs)
    assert semantics["outputs"] == (
        {"index": 0, "role": "expert_major_tokens", "axes": ("expert_major_row", "feature")},
        {"index": 1, "role": "expert_major_scales", "axes": ("expert_major_row",)},
        {"index": 2, "role": "restore_indices", "axes": ("rank_major_row",)},
        {"index": 3, "role": "global_tokens_per_local_expert", "axes": ("local_expert",)},
    )
    assert node.outputs[1].shape == ((6,) if with_scales else (0,))
    backward = nodes["torchtitan_npu.npu_moe_token_unpermute.default"]
    assert backward.attrs["tensor_semantics"]["outputs"] == (
        {"index": 0, "role": "restored_rows", "axes": ("routed_row", "feature")},
    )


@pytest.mark.parametrize("topk", [None, 2], ids=["single-route-vector", "topk-matrix"])
def test_permute_semantics_describe_route_indices_and_backward_output(topk):
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.ops.ascendc.moe_token_permute import npu_moe_token_permute

    tokens = torch.empty(3, 7, device="meta", requires_grad=True)
    indices = torch.empty((3,) if topk is None else (3, topk), device="meta", dtype=torch.int32)
    capture = OpDispatchCapture(record_memory=False)

    with meta_runtime(), capture:
        output, _ = npu_moe_token_permute(tokens, indices)
        output.backward(torch.empty_like(output))

    nodes = {node.annotations["raw_op_type"]: node for node in capture.build_nodes().values()}
    forward = nodes["torchtitan_npu.npu_moe_token_permute.default"].attrs["tensor_semantics"]
    assert forward["inputs"][1] == {
        "index": 1,
        "role": "expert_indices",
        "axes": ("token",) if topk is None else ("token", "topk"),
    }
    backward = nodes["torchtitan_npu.npu_moe_token_permute_backward.default"].attrs["tensor_semantics"]
    assert backward["outputs"] == ({"index": 0, "role": "tokens_gradient", "axes": ("token", "feature")},)
    assert tokens.grad is not None and tokens.grad.shape == (3, 7)


@pytest.mark.parametrize("num_tokens", [0, 3], ids=["empty", "nonempty"])
def test_unpermute_weighted_output_uses_presence_not_probability_size(num_tokens):
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.ops.ascendc.moe_token_unpermute import npu_moe_token_unpermute

    values = torch.empty(num_tokens * 2, 7, device="meta")
    indices = torch.empty(num_tokens * 2, device="meta", dtype=torch.int32)
    probabilities = torch.empty(num_tokens, 2, device="meta")
    capture = OpDispatchCapture(record_memory=False)

    with meta_runtime(), capture:
        npu_moe_token_unpermute(values, indices, probabilities)

    (node,) = capture.build_nodes().values()
    assert node.attrs["tensor_semantics"]["outputs"] == (
        {"index": 0, "role": "combined_tokens", "axes": ("token", "feature")},
    )
    assert node.attrs["tensor_semantics"]["inputs"][2] == {
        "index": 2,
        "role": "routing_probabilities",
        "axes": ("token", "topk"),
    }
    assert node.outputs[0].shape == (num_tokens, 7)


def test_unweighted_unpermute_gradient_keeps_routed_rows_and_scalar_feature():
    from torchtitan_simulator.runtime import meta_runtime

    from torchtitan_npu.ops.ascendc.moe_token_unpermute import npu_moe_token_unpermute

    values = torch.empty(6, 1, device="meta", requires_grad=True)
    indices = torch.empty(6, device="meta", dtype=torch.int32)
    capture = OpDispatchCapture(record_memory=False)

    with meta_runtime(), capture:
        output = npu_moe_token_unpermute(values, indices)
        output.backward(torch.empty_like(output))

    node = next(
        node
        for node in capture.build_nodes().values()
        if node.annotations["raw_op_type"] == "torchtitan_npu.npu_moe_token_unpermute_grad_unweighted.default"
    )
    assert node.attrs["tensor_semantics"] == {
        "version": 1,
        "inputs": (
            {"index": 0, "role": "restored_rows_gradient", "axes": ("routed_row", "feature")},
            {"index": 1, "role": "sorted_indices", "axes": ("routed_row",)},
        ),
        "outputs": ({"index": 0, "role": "permuted_tokens_gradient", "axes": ("routed_row", "feature")},),
    }
    assert node.outputs[0].shape == (6, 1)
