# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native dispatch argument roles and literal values exposed through OpNode.attrs."""

import json

import pytest
import torch
import torch.distributed as dist
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.operator_arguments import operator_attributes


@pytest.mark.parametrize(
    ("operation", "other"),
    [(torch.ops.aten.add.Tensor, 3), (torch.ops.aten.mul.Tensor, 0.25), (torch.ops.aten.floor_divide.default, 2)],
    ids=["add-integer", "mul-float", "floor-divide-integer"],
)
def test_tensor_overload_keeps_runtime_scalar_without_reading_tensor(operation, other):
    source = torch.empty(2, 3, device="meta")
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        operation(source, other)

    (node,) = capture.build_nodes().values()
    assert node.annotations["raw_op_type"] == str(operation)
    assert node.attrs["other"] == other
    assert node.attrs["input_tensor_bindings"] == (("self", 0),)
    assert [tensor.shape for tensor in node.inputs] == [(2, 3)]


def test_index_binding_retains_none_positions_and_repeated_tensor_occurrences():
    source = torch.empty(2, 3, device="meta")
    index = torch.empty(2, dtype=torch.int64, device="meta")
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        torch.ops.aten.index.Tensor(source, [None, index])
        torch.cat([source, source])

    nodes = list(capture.build_nodes().values())
    index_node = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.index.Tensor")
    cat_node = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.cat.default")
    assert index_node.attrs["input_tensor_bindings"] == (("self", 0), ("indices", (None, 1)))
    assert [tensor.shape for tensor in index_node.outputs] == [(2, 2)]
    assert cat_node.attrs["input_tensor_bindings"] == (("tensors", (0, 1)),)
    assert [tensor.shape for tensor in cat_node.inputs] == [(2, 3), (2, 3)]


def test_binding_tracks_keyword_occurrence_order_independently_of_schema_order():
    source = torch.empty(2, 3, device="meta")
    index = torch.empty(2, dtype=torch.int64, device="meta")
    values = torch.empty(2, 3, device="meta")

    attrs = operator_attributes(
        torch.ops.aten.index_put.default,
        (source,),
        {"values": values, "indices": [index], "accumulate": True},
    )

    assert attrs["input_tensor_bindings"] == (("self", 0), ("indices", (2,)), ("values", 1))
    assert attrs["accumulate"] is True


def test_native_dispatch_keeps_empty_tensor_lists_and_optional_defaults():
    source = torch.empty(2, 3, device="meta")
    # A scoped native schema isolates argument capture from any vendor kernels.
    library = torch.library.Library("simulator_argument_capture", "DEF")
    try:
        library.define("lists(Tensor source, Tensor[] empty, *, Tensor? optional=None) -> Tensor")
        library.impl("lists", lambda source, empty, optional=None: source.clone(), "Meta")
        capture = OpDispatchCapture(record_memory=False)

        with capture:
            torch.ops.simulator_argument_capture.lists.default(source, [])

        (node,) = capture.build_nodes().values()
        assert node.attrs["input_tensor_bindings"] == (("source", 0), ("empty", ()), ("optional", None))
        assert [tensor.shape for tensor in node.inputs] == [(2, 3)]
    finally:
        library._destroy()


def test_native_keyword_tensor_bindings_reference_the_actual_input_metadata():
    source = torch.empty(2, 3, device="meta")
    left = torch.empty(4, device="meta")
    right = torch.empty(5, 6, device="meta")
    library = torch.library.Library("simulator_keyword_capture", "DEF")
    try:
        library.define("roles(Tensor source, *, Tensor left, Tensor right) -> Tensor")
        library.impl("roles", lambda source, left, right: source.clone(), "Meta")
        capture = OpDispatchCapture(record_memory=False)

        with capture:
            torch.ops.simulator_keyword_capture.roles.default(source, right=right, left=left)

        (node,) = capture.build_nodes().values()
        shapes_by_role = {name: node.inputs[index].shape for name, index in node.attrs["input_tensor_bindings"]}
        assert shapes_by_role == {"source": (2, 3), "left": (4,), "right": (5, 6)}
    finally:
        library._destroy()


def test_native_tensor_dictionary_does_not_expose_incomplete_bindings():
    source = torch.empty(2, 3, device="meta")
    library = torch.library.Library("simulator_dictionary_capture", "DEF")
    try:
        library.define("dictionary(Tensor source, Dict(str, Tensor) values) -> Tensor")
        library.impl("dictionary", lambda source, values: source.clone(), "Meta")
        capture = OpDispatchCapture(record_memory=False)

        with capture:
            torch.ops.simulator_dictionary_capture.dictionary.default(source, {"mask": source})

        (node,) = capture.build_nodes().values()
        assert "input_tensor_bindings" not in node.attrs
        assert [tensor.shape for tensor in node.inputs] == [(2, 3), (2, 3)]
    finally:
        library._destroy()


def test_fused_adamw_dispatch_hook_binds_keyword_self_and_all_six_tensor_lists():
    parameter = torch.empty(2, 3, device="meta")
    tensors = [[parameter], [torch.empty_like(parameter)], [torch.empty_like(parameter)], [torch.empty_like(parameter)]]
    kwargs = {
        "self": tensors[0],
        "grads": tensors[1],
        "exp_avgs": tensors[2],
        "exp_avg_sqs": tensors[3],
        "max_exp_avg_sqs": [],
        "state_steps": [torch.tensor(1.0)],
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }
    capture = OpDispatchCapture(record_memory=False)

    # Enter the dispatch hook with kwargs intact; dispatcher versions may
    # otherwise canonicalize these Tensor arguments into positional inputs.
    capture.__torch_dispatch__(torch.ops.aten._fused_adamw_.default, (), (), kwargs)

    (node,) = capture.build_nodes().values()
    assert node.annotations["optimizer_updated_parameter_count"] == 1
    assert node.annotations["optimizer_logical_parameter_shapes"] == [(2, 3)]
    assert node.attrs["input_tensor_bindings"] == (
        ("self", (0,)),
        ("grads", (1,)),
        ("exp_avgs", (2,)),
        ("exp_avg_sqs", (3,)),
        ("max_exp_avg_sqs", ()),
        ("state_steps", (4,)),
        ("grad_scale", None),
        ("found_inf", None),
    )


@pytest.mark.parametrize(
    ("value", "literal"),
    [(float("-inf"), "-inf"), (float("inf"), "+inf"), (float("nan"), "nan")],
    ids=["negative-infinity", "positive-infinity", "nan"],
)
def test_factory_nonfinite_literal_has_strict_json_representation(value, literal):
    source = torch.empty(2, 3, device="meta")
    capture = OpDispatchCapture(record_memory=False)

    with capture:
        source.new_full((1,), value)

    (node,) = capture.build_nodes().values()
    assert node.attrs["fill_value"] == ("__sim_scalar_float_v1__", literal)
    encoded = json.dumps(node.attrs, allow_nan=False)
    assert json.loads(encoded)["fill_value"] == ["__sim_scalar_float_v1__", literal]


@pytest.mark.parametrize("operation", ["SUM", "PRODUCT", "MIN", "MAX", "AVG", "BAND", "BOR", "BXOR"])
def test_legacy_collective_retains_actual_reduce_op_enum(tmp_path, operation):
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=2)
    try:
        source = torch.empty(4, device="meta")
        capture = OpDispatchCapture(record_memory=False)

        with capture:
            dist.all_reduce(source, op=getattr(dist.ReduceOp, operation))

        node = next(
            node
            for node in capture.build_nodes().values()
            if node.annotations["raw_op_type"] == "c10d.allreduce_.default"
        )
        assert node.attrs["reduce_op"] == operation.lower()
        assert node.attrs["input_tensor_bindings"] == (("tensors", (0,)), ("sparse_indices", None))
    finally:
        dist.destroy_process_group()


def test_premul_sum_is_explicitly_unsupported_instead_of_plain_sum():
    reduction = torch._C._distributed_c10d._make_nccl_premul_sum(0.5)
    attrs = operator_attributes(torch.ops.c10d.allreduce_.default, ([], None, reduction, None), {})

    assert attrs["reduce_op"] == "unsupported:premul_sum"
