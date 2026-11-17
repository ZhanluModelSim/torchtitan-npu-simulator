# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic producer facts using real dispatch and a data-free native process group."""

import weakref

import pytest
import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan_simulator.capture.collective_facts import (
    collective_fact_scope,
    install_reduce_scatter_observer,
    reduction_name,
    retain_backward_facts,
)
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.patches import PatchScope
from torchtitan_simulator.compat.routing import install_balanced_dispatch


@pytest.fixture
def sparse_group():
    # This backend executes native schemas/Meta/autograd without network data.
    # It gives no evidence about numerical collectives or hardware performance.
    dist.init_process_group("fake", store=FakeStore(), rank=3, world_size=4)
    try:
        yield dist.new_group([1, 3], backend="fake")
    finally:
        dist.destroy_process_group()


def _communication_nodes(capture):
    return [node for node in capture.build_nodes().values() if "collective_facts" in node.attrs]


def test_native_ag_rs_ar_keep_actual_sparse_rank_group_and_named_reduction(sparse_group):
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        shard = torch.empty(3, device="meta")
        full = torch.empty(6, device="meta")
        dist.all_gather_into_tensor(full, shard, group=sparse_group)
        dist.reduce_scatter_tensor(shard, full, op=dist.ReduceOp.SUM, group=sparse_group)
        dist.all_reduce(shard, op=dist.ReduceOp.MAX, group=sparse_group)

    ag, rs, ar = _communication_nodes(capture)
    for node in (ag, rs, ar):
        facts = node.attrs["collective_facts"]
        assert facts["capture_rank"] == 3
        assert facts["ordered_group"] == [1, 3]
        assert facts["tensor_contract"]["dense"] is True
        assert facts["tensor_contract"]["uniform_across_ranks"] is True
        assert "unresolved_fields" not in facts
    assert ag.attrs["collective_facts"]["buffer_relation"] == "distinct"
    assert rs.attrs["collective_facts"]["reduction"]["operation"] == "SUM"
    assert rs.attrs["reduce_op"] == "sum"
    assert ar.attrs["collective_facts"]["reduction"]["operation"] == "MAX"
    assert ar.attrs["reduce_op"] == "max"
    assert ar.attrs["collective_facts"]["buffer_relation"] == "inplace"


@pytest.mark.parametrize("factor", [0.25, 1.0, 0.0, -0.5])
def test_native_rs_preserves_host_factor_and_restores_scope(sparse_group, factor):
    original = dist.reduce_scatter_single
    full = torch.empty(6, device="meta")
    shard = torch.empty(3, device="meta")
    op = dist.ReduceOp.PREMUL_SUM(factor)
    capture = OpDispatchCapture(record_memory=False)
    with pytest.raises(RuntimeError, match="leave scope"), PatchScope() as scope, capture:
        install_reduce_scatter_observer(scope)
        dist.reduce_scatter_single(shard, full, op=op, group=sparse_group)
        raise RuntimeError("leave scope")

    assert dist.reduce_scatter_single is original
    (node,) = _communication_nodes(capture)
    facts = node.attrs["collective_facts"]
    assert facts["schema_version"] == 2
    assert facts["reduction"] == {
        "operation": "PREMUL_SUM",
        "pre_scale": factor,
        "post_scale": 1.0,
        "scale_implemented_by_nodes": False,
        "source": "native_reduceop_factor_v1",
    }
    assert "unresolved_fields" not in facts
    assert node.attrs["reduce_op"] == "premul_sum"
    assert len(capture.build_nodes()) == 1


@pytest.mark.parametrize("name", ["SUM", "AVG"])
def test_reduce_scatter_observer_preserves_unscaled_native_enums(sparse_group, name):
    full = torch.empty(6, device="meta")
    shard = torch.empty(3, device="meta")
    capture = OpDispatchCapture(record_memory=False)
    with PatchScope() as scope, capture:
        install_reduce_scatter_observer(scope)
        dist.reduce_scatter_single(shard, full, op=getattr(dist.ReduceOp, name), group=sparse_group)

    (node,) = _communication_nodes(capture)
    assert node.attrs["reduce_op"] == name.lower()
    assert node.attrs["collective_facts"]["reduction"]["operation"] == name


def test_native_rs_does_not_read_tensor_factor_or_reuse_prior_factor(sparse_group):
    full = torch.empty(6, device="meta")
    shard = torch.empty(3, device="meta")
    scalar_op = dist.ReduceOp.PREMUL_SUM(0.125)
    tensor_op = dist.ReduceOp.PREMUL_SUM(torch.empty((), device="meta"))
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        with PatchScope() as scope:
            install_reduce_scatter_observer(scope)
            dist.reduce_scatter_single(shard, full, op=scalar_op, group=sparse_group)
            dist.reduce_scatter_single(shard, full, op=tensor_op, group=sparse_group)
        dist.reduce_scatter_single(shard, full, op=scalar_op, group=sparse_group)
    known, *missing = (node.attrs["collective_facts"] for node in _communication_nodes(capture))
    assert known["reduction"]["pre_scale"] == 0.125
    for facts in missing:
        assert facts["reduction"]["operation"] == "PREMUL_SUM"
        assert facts["reduction"]["pre_scale"] is None
        assert facts["unresolved_fields"] == ["reduction.pre_scale"]


def test_ag_local_alias_does_not_prove_all_rank_output_alias(sparse_group):
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        full = torch.empty(6, device="meta")
        local = full[3:]
        dist.all_gather_into_tensor(full, local, group=sparse_group)

    (node,) = _communication_nodes(capture)
    assert node.attrs["collective_facts"]["buffer_relation"] == "distinct"


def test_ag_single_rank_proven_output_alias_elides_copy(sparse_group):
    single = dist.new_group([3], backend="fake")
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        full = torch.empty(6, device="meta")
        dist.all_gather_into_tensor(full, full, group=single)

    (node,) = _communication_nodes(capture)
    assert node.attrs["collective_facts"]["buffer_relation"] == "same_region"


def test_opaque_reduceop_print_text_does_not_become_sum():
    assert reduction_name("ScriptObject <__torch__.torch.classes.c10d.ReduceOp>") is None
    assert reduction_name("SUM") is None
    assert reduction_name(torch.classes.c10d.ReduceOp()) == "SUM"


def test_local_equal_a2a_splits_alone_do_not_prove_full_group(sparse_group):
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        all_to_all_single(torch.empty(4, 2, device="meta"), [2, 2], [2, 2], sparse_group)

    (node,) = _communication_nodes(capture)
    facts = node.attrs["collective_facts"]
    assert facts["tensor_contract"]["uniform_across_ranks"] is False
    assert facts["unresolved_fields"] == ["a2a_full_group_split_facts"]


def test_native_a2a_backward_uses_transpose_of_exact_forward_proof(sparse_group):
    facts = {
        "source": "torchtitan_balanced_ep_v1",
        "a2a_proof": {
            "kind": "torchtitan_balanced_ep_v1",
            "num_tokens_per_rank": [5, 5],
            "top_k": 3,
            "num_experts": 4,
            "ordered_group": [1, 3],
            "all_routers_force_load_balance": True,
            "direction": "dispatch",
            "trailing_shape": [2],
            "dtype": "float32",
            "source_version": "torchtitan_simulator_balanced_ep_v1",
        },
    }
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        source = torch.empty(15, 2, device="meta", requires_grad=True)
        with collective_fact_scope(facts):
            received = all_to_all_single(source, [7, 7], [8, 7], sparse_group)
        retain_backward_facts(received, facts)
        received.sum().backward()

    forward, backward = _communication_nodes(capture)
    assert forward.attrs["collective_facts"]["a2a_proof"]["direction"] == "dispatch"
    assert backward.attrs["collective_facts"]["a2a_proof"]["direction"] == "combine"
    assert backward.attrs["input_split_sizes"] == [7, 7]
    assert backward.attrs["output_split_sizes"] == [8, 7]
    assert source.grad.shape == (15, 2)
    assert facts["a2a_proof"]["direction"] == "dispatch"


def test_native_a2a_backward_swaps_full_group_tensor_shapes_with_transposed_rows(sparse_group):
    facts = {
        "source": "explicit_full_group_splits_v1",
        "send_rows": [[1, 2], [3, 4]],
        "recv_rows": [[1, 3], [2, 4]],
        "rank_input_shapes": [[3, 2], [7, 2]],
        "rank_output_shapes": [[4, 2], [6, 2]],
    }
    capture = OpDispatchCapture(record_memory=False)
    with capture:
        source = torch.empty(7, 2, device="meta", requires_grad=True)
        with collective_fact_scope(facts):
            received = all_to_all_single(source, [2, 4], [3, 4], sparse_group)
        retain_backward_facts(received, facts)
        received.sum().backward()

    forward, backward = _communication_nodes(capture)
    forward_facts = forward.attrs["collective_facts"]
    backward_facts = backward.attrs["collective_facts"]
    assert forward.inputs[0].shape == (7, 2)
    assert backward.inputs[0].shape == (6, 2)
    assert backward.outputs[0].shape == (7, 2)
    assert backward_facts["send_rows"] == [[1, 3], [2, 4]]
    assert backward_facts["recv_rows"] == [[1, 2], [3, 4]]
    assert backward_facts["rank_input_shapes"] == [[4, 2], [6, 2]]
    assert backward_facts["rank_output_shapes"] == [[3, 2], [7, 2]]
    assert forward_facts["rank_input_shapes"] == [[3, 2], [7, 2]]
    assert source.grad.shape == (7, 2)


class _Dispatcher:
    """Minimal producer call protocol; transport and autograd remain native."""

    def __init__(self, group):
        self.ep_mesh = DeviceMesh.from_group(group, "cpu", mesh=[1, 3])

    def dispatch(self, x, scores, indices, counts):
        count_tensor = torch.empty(4, device="meta", dtype=torch.int64)
        received = self._token_count_exchange(count_tensor, self.ep_mesh.get_group(), 2)
        synchronized, sends, receives = self._sync_token_count_exchange(count_tensor, received, 2)
        self.received_count_ref = weakref.ref(synchronized)
        routed = x.repeat_interleave(scores.shape[-1], dim=0)
        return self._dispatch_token_exchange(routed, self.ep_mesh.get_group(), receives, sends), sends, receives

    def _token_count_exchange(self, counts, pg, ep_size):
        raise AssertionError("Expected the scoped Meta exchange adapter")

    def _sync_token_count_exchange(self, local_counts, received_counts, ep_size):
        raise AssertionError("Expected the scoped Meta count adapter")

    def _dispatch_token_exchange(self, routed, pg, output_splits, input_splits):
        raise AssertionError("Expected the scoped Meta dispatch adapter")

    def _combine_token_exchange(self, routed, pg, input_splits, output_splits):
        raise AssertionError("Expected the scoped Meta combine adapter")


class _RouterModel(torch.nn.Module):
    def __init__(self, group, *, balanced=True):
        super().__init__()
        self._debug_force_load_balance = balanced
        self.token_dispatcher = _Dispatcher(group)


def test_balanced_routing_produces_count_dispatch_combine_and_backward_facts(sparse_group):
    model = _RouterModel(sparse_group)
    dispatcher = model.token_dispatcher
    capture = OpDispatchCapture(record_memory=False)
    original = _Dispatcher.dispatch
    with PatchScope() as scope, capture:
        install_balanced_dispatch(scope, model, uniform_geometry_source="test_uniform_spmd_input_v1")
        x = torch.empty(5, 2, device="meta", requires_grad=True)
        received, sends, receives = dispatcher.dispatch(x, torch.empty(5, 3, device="meta"), None, None)
        assert dispatcher.received_count_ref() is None
        combined = dispatcher._combine_token_exchange(received, sparse_group, sends, receives)
        combined.sum().backward()

    count, dispatch, combine, grad_combine, grad_dispatch = _communication_nodes(capture)
    assert count.attrs["collective_facts"]["send_rows"] == [[1, 1], [1, 1]]
    assert count.attrs["collective_facts"]["tensor_contract"]["uniform_across_ranks"] is True
    assert [
        n.attrs["collective_facts"]["a2a_proof"]["direction"] for n in (dispatch, combine, grad_combine, grad_dispatch)
    ] == ["dispatch", "combine", "dispatch", "combine"]
    proof = dispatch.attrs["collective_facts"]["a2a_proof"]
    assert proof["num_tokens_per_rank"] == [5, 5]
    assert proof["ordered_group"] == [1, 3]
    assert proof["uniform_geometry_source"] == "test_uniform_spmd_input_v1"
    assert x.grad.shape == x.shape
    assert _Dispatcher.dispatch is original


def test_balanced_routing_rejects_an_unverified_router_before_producing_proof(sparse_group):
    model = _RouterModel(sparse_group)
    model.add_module("unbalanced", _RouterModel(sparse_group, balanced=False))
    with PatchScope() as scope, pytest.raises(ValueError, match="every router"):
        install_balanced_dispatch(scope, model, uniform_geometry_source="test_uniform_spmd_input_v1")
