# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan_simulator.compat.routing import balanced_splits


@pytest.mark.parametrize(
    "rank,expected_receives", [(0, [8, 8]), (1, [7, 7])], ids=["first-partition", "last-partition"]
)
def test_balanced_splits_match_uneven_round_robin_expert_assignments(rank, expected_receives):
    # Five tokens, three choices, four experts: [4, 4, 4, 3] routed rows.
    expert_ids = torch.arange(15) % 4
    histogram = torch.bincount(expert_ids, minlength=4).reshape(2, 2)

    sends, receives = balanced_splits(5, 3, 4, 2, rank)

    assert sends == histogram.sum(1).tolist() == [8, 7]
    assert receives == expected_receives


def test_meta_single_exchange_preserves_split_direction_count_dependency_and_scope(monkeypatch):
    from types import SimpleNamespace

    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
    from torchtitan_simulator.capture.comm_events import _collective_dependencies
    from torchtitan_simulator.compat import routing
    from torchtitan_simulator.compat.patches import PatchScope

    cls = AllToAllTokenDispatcher
    names = (
        "dispatch",
        "_token_count_exchange",
        "_sync_token_count_exchange",
        "_dispatch_token_exchange",
        "_combine_token_exchange",
    )
    originals = {name: getattr(cls, name) for name in names}
    group = object()
    dispatcher = SimpleNamespace(ep_mesh=group)
    counts = torch.empty(4, device="meta", dtype=torch.int64)
    tensor = torch.empty(15, 3, device="meta")
    result = torch.empty_like(tensor)
    calls = []

    def single(value, output_splits, input_splits, mesh):
        calls.append((value.shape, output_splits, input_splits, mesh, _collective_dependencies.get()))
        return result

    monkeypatch.setattr(routing, "all_to_all_single", single)
    token = routing._geometry.set(routing._Geometry(id(dispatcher), 5, 3, counts))
    try:
        with PatchScope() as scope:
            routing._install_dispatch_class(scope, cls)
            assert (
                cls._token_count_exchange(dispatcher, num_local_tokens_per_expert_E=counts, pg=group, ep_size=2)
                is result
            )
            assert (
                cls._dispatch_token_exchange(
                    dispatcher, routed_input_ND=tensor, pg=group, output_splits=[8, 7], input_splits=[6, 9]
                )
                is result
            )
            assert (
                cls._combine_token_exchange(
                    dispatcher, routed_output_RD=tensor, pg=group, input_splits=[6, 9], output_splits=[8, 7]
                )
                is result
            )
        assert calls == [
            ((2, 2), None, None, group, ()),
            ((15, 3), [8, 7], [6, 9], group, (counts,)),
            ((15, 3), [6, 9], [8, 7], group, ()),
        ]
        assert _collective_dependencies.get() == ()
        assert {name: getattr(cls, name) for name in names} == originals
    finally:
        routing._geometry.reset(token)


def test_non_meta_exchange_keeps_original_bindings_and_arguments(monkeypatch):
    from types import SimpleNamespace

    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
    from torchtitan_simulator.compat import routing
    from torchtitan_simulator.compat.patches import PatchScope

    cls = AllToAllTokenDispatcher
    calls = []
    result = object()
    names = ("_token_count_exchange", "_dispatch_token_exchange", "_combine_token_exchange")
    for name in names:

        def original(self, *args, _name=name):
            calls.append((_name, args))
            return result

        monkeypatch.setattr(cls, name, original)
    tensor = torch.empty(15, 3)
    group = object()
    dispatcher = SimpleNamespace(ep_mesh=group)
    with PatchScope() as scope:
        routing._install_dispatch_class(scope, cls)
        assert cls._token_count_exchange(dispatcher, tensor, group, 2) is result
        assert cls._dispatch_token_exchange(dispatcher, tensor, group, [8, 7], [6, 9]) is result
        assert cls._combine_token_exchange(dispatcher, tensor, group, [6, 9], [8, 7]) is result
    assert [name for name, _ in calls] == list(names)
    assert calls[0][1] == (tensor, group, 2)
    assert calls[1][1] == (tensor, group, [8, 7], [6, 9])
    assert calls[2][1] == (tensor, group, [6, 9], [8, 7])


def _reaches(nodes, source, target):
    pending = [target]
    seen = set()
    while pending:
        op_id = pending.pop()
        if op_id == source:
            return True
        if op_id not in seen:
            seen.add(op_id)
            pending.extend(nodes[op_id].predecessors)
    return False


@pytest.mark.parametrize("family", ["deepseek_v4", "kimi_k3"])
@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("quantized", [False, True], ids=["unquantized", "mxfp8"])
def test_default_single_ep_capture_preserves_gradients_and_causal_dependencies(family, mode, quantized, tmp_path):
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

    workload = simulate(
        SimulationConfig(
            config=f"{family}_smoketest",
            world_size=4,
            logical_rank=3,
            expert_parallel_degree=2,
            seq_len=128,
            activation_checkpoint_mode=mode,
            enable_quantized_training=quantized,
            output_dir=str(tmp_path),
        )
    )
    annotations = workload.iteration.schedule.annotations
    assert annotations["runtime_capabilities"]["ep_token_exchange"] == "functional_all_to_all_single"
    assert annotations["runtime_capabilities"]["ep_token_dispatcher_classes"] > 0
    assert annotations["capture_validation"]["gradient_parameters"] == (112 if family == "deepseek_v4" else 110)
    assert annotations["capture_validation"]["graph_acyclic"]
    assert annotations["capture_validation"]["peak_storage_unique"]
    nodes = {op_id: node for template in workload.step_templates.values() for op_id, node in template.nodes.items()}
    ep = [e for e in annotations["communication_events"] if e["comm_dim"] == "ep"]
    assert ep and any(e["comp_type"] == "B" for e in ep)
    for event in ep:
        assert nodes[event["op_id"]].annotations["raw_op_type"] == "_c10d_functional.all_to_all_single.default"
        assert event["comm_ranks"] == [[2, 3]]
        assert event["volume_bytes"] == nodes[event["op_id"]].attrs["communication_input_bytes"]
    counts = [
        e for e in ep if e["dtype"] == "int64" and nodes[e["op_id"]].annotations["execution_kind"] == "original_forward"
    ]
    assert counts
    for count in counts:
        path = nodes[count["op_id"]].annotations["module_path"]
        forward = [
            e
            for e in ep
            if nodes[e["op_id"]].annotations.get("module_path") == path
            and nodes[e["op_id"]].annotations["execution_kind"] == "original_forward"
        ]
        # Kimi applies router scores after combine; DSV4 sends them to W2.
        assert len(forward) == (4 if family == "deepseek_v4" else 3)
        count_id, token_id = [e["op_id"] for e in forward[:2]]
        combine_id = forward[-1]["op_id"]
        gmms = [
            n.op_id
            for n in nodes.values()
            if n.op_type == "grouped_mm"
            and n.annotations.get("module_path", "").startswith(path + ".inner_experts")
            and n.annotations["execution_kind"] == "original_forward"
        ]
        assert len(gmms) == 3
        unpermutes = [
            n.op_id
            for n in nodes.values()
            if n.op_id > combine_id
            and n.annotations.get("module_path") == path
            and n.annotations["execution_kind"] == "original_forward"
            and n.annotations["raw_op_type"].endswith(".npu_moe_token_unpermute.default")
        ]
        assert len(unpermutes) == 1
        pairs = [
            (count_id, token_id),
            (token_id, gmms[0]),
            (token_id, gmms[1]),
            (gmms[2], combine_id),
            (combine_id, unpermutes[0]),
        ]
        if family == "deepseek_v4":
            score_id = forward[2]["op_id"]
            pairs.extend([(count_id, score_id), (score_id, gmms[2])])
        for source, target in pairs:
            assert _reaches(nodes, source, target), (path, source, target)


def test_failed_single_capture_restores_dispatch_methods_and_process_group(monkeypatch, tmp_path):
    import torch.distributed as dist
    from torchtitan_simulator.compat import routing
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

    from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher

    cls = AscAllToAllTokenDispatcher
    names = (
        "dispatch",
        "_token_count_exchange",
        "_sync_token_count_exchange",
        "_dispatch_token_exchange",
        "_combine_token_exchange",
    )
    originals = {name: getattr(cls, name) for name in names}

    def fail(*args, **kwargs):
        raise RuntimeError("single transport failure")

    monkeypatch.setattr(routing, "all_to_all_single", fail)
    with pytest.raises(RuntimeError, match="single transport failure"):
        simulate(SimulationConfig(world_size=2, expert_parallel_degree=2, output_dir=str(tmp_path)))
    assert {name: getattr(cls, name) for name in names} == originals
    assert routing._geometry.get() is None
    assert not dist.is_initialized()


@pytest.mark.parametrize("state", ["missing", "foreign-dispatcher", "unsynchronized"])
def test_meta_dispatch_rejects_missing_count_dependency_before_transport(state, monkeypatch):
    from types import SimpleNamespace

    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
    from torchtitan_simulator.compat import routing
    from torchtitan_simulator.compat.patches import PatchScope

    dispatcher = SimpleNamespace(ep_mesh=object())
    tensor = torch.empty(15, 3, device="meta")
    received = torch.empty(4, device="meta", dtype=torch.int64)
    geometry = (
        None
        if state == "missing"
        else routing._Geometry(
            -1 if state == "foreign-dispatcher" else id(dispatcher),
            5,
            3,
            None if state == "unsynchronized" else received,
        )
    )

    def unexpected_transport(*args, **kwargs):
        pytest.fail("Data exchange must not launch without synchronized count ownership")

    monkeypatch.setattr(routing, "all_to_all_single", unexpected_transport)
    token = routing._geometry.set(geometry)
    try:
        with PatchScope() as scope:
            routing._install_dispatch_class(scope, AllToAllTokenDispatcher)
            with pytest.raises(RuntimeError, match="no synchronized count dependency"):
                AllToAllTokenDispatcher._dispatch_token_exchange(dispatcher, tensor, dispatcher.ep_mesh, [8, 7], [6, 9])
    finally:
        routing._geometry.reset(token)
