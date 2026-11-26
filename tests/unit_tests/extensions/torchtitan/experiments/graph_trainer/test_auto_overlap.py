"""Tests for auto-overlap pass construction and calibration rounds."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import auto_overlap as graph_trainer_auto_overlap
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.utils import node_id


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        compile=SimpleNamespace(
            passes=[],
            precompile_artifact_dir="",
        )
    )


def _named_pass(name: str):
    def pass_fn(gm, example_inputs):
        return gm

    pass_fn.__name__ = name
    return pass_fn


def test_auto_overlap_pipeline_is_registered():
    assert PASS_PIPELINE_REGISTRY["npu_auto_overlap"] is graph_trainer_auto_overlap.construct_npu_auto_overlap_passes
    assert (
        PASS_PIPELINE_REGISTRY["mutation-functionalization+npu_auto_overlap"]
        is graph_trainer_auto_overlap.construct_mutation_functionalization_npu_auto_overlap_passes
    )


@pytest.mark.parametrize(
    ("pipeline", "installs_runner"),
    [
        ("default", False),
        ("npu_auto_overlap", True),
        ("mutation-functionalization+npu_auto_overlap", True),
    ],
)
@pytest.mark.parametrize("model_owned", [False, True], ids=["generic", "dsv41"])
def test_runtime_context_patch_is_enabled_only_for_runtime_auto_overlap(
    monkeypatch,
    pipeline,
    installs_runner,
    model_owned,
):
    from torchtitan.experiments.graph_trainer import graph_builder

    original = graph_builder._build_fwd_bwd_graphs
    monkeypatch.setattr(graph_builder, "_build_fwd_bwd_graphs", original)
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
    from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41GraphTrainer

    monkeypatch.setattr(TrainerEx, "__init__", lambda self, config: None)
    trainer_cls = DeepSeekV41GraphTrainer if model_owned else GraphTrainerEx
    trainer_cls(SimpleNamespace(compile=SimpleNamespace(pass_pipeline=pipeline)))

    builder = graph_builder._build_fwd_bwd_graphs
    assert (builder is not original) == installs_runner
    if installs_runner:
        assert builder.__wrapped__ is original


def test_npu_auto_overlap_runs_after_eager_chunk_metadata():
    config = _config()
    passes = [
        _named_pass("before"),
        _named_pass("joint_transformer_block_bucketing_reordering_pass"),
        _named_pass("populate_eager_chunk_metadata_pass"),
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "before",
        "joint_transformer_block_bucketing_reordering_pass",
        "populate_eager_chunk_metadata_pass",
        "ep_overlap_validate_pass",
        "npu_moe_auto_overlap_pass",
        "regional_inductor_pass",
    ]


def test_npu_auto_overlap_keeps_original_pipeline_without_ep_preparation(caplog):
    config = _config()
    passes = [_named_pass("before"), _named_pass("regional_inductor_pass")]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "ep_overlap_schedule_pass=0" in caplog.text


def test_npu_auto_overlap_keeps_original_pipeline_with_duplicate_manual_pass(caplog):
    config = _config()
    manual_pass = _named_pass("ep_overlap_schedule_pass")
    passes = [
        manual_pass,
        manual_pass,
        _named_pass("populate_eager_chunk_metadata_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "ep_overlap_schedule_pass=2" in caplog.text


def test_npu_auto_overlap_keeps_original_pipeline_without_eager_chunk_metadata(caplog):
    config = _config()
    passes = [
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "populate_eager_chunk_metadata_pass=0" in caplog.text


def test_npu_auto_overlap_keeps_precompiled_runtime_pipeline(caplog):
    config = _config()
    config.compile.precompile_artifact_dir = "/tmp/precompiled"
    passes = [_named_pass("cudagraph_pass")]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert configured is passes
    assert "precompiled artifact" in caplog.text


def test_npu_auto_overlap_writes_back_scheduler_result(monkeypatch, caplog):
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import npu_moe_auto_scheduler

    class Scheduler:
        def __init__(self, candidate_gm, **_kwargs):
            self.candidate_gm = candidate_gm

        def run(self):
            return self.candidate_gm

    monkeypatch.setattr(
        npu_moe_auto_scheduler,
        "NpuMoeAutoOverlapScheduler",
        Scheduler,
    )
    config = _config()
    passes = [
        _named_pass("before"),
        _named_pass("populate_eager_chunk_metadata_pass"),
        _named_pass("ep_overlap_schedule_pass"),
        _named_pass("regional_inductor_pass"),
    ]

    configured = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        passes,
        config,
    )

    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "before",
        "populate_eager_chunk_metadata_pass",
        "ep_overlap_validate_pass",
        "npu_moe_auto_overlap_pass",
        "regional_inductor_pass",
    ]
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    gm = torch.fx.GraphModule({}, graph)
    assert configured[3](gm, ()) is gm
    assert "did not provide a candidate runner" in caplog.text


def test_pipeline_factory_preserves_default_preparation_and_partial_passes(monkeypatch):
    from torchtitan.experiments.graph_trainer import passes as upstream_passes

    traced_result, parallelism_context = object(), object()
    config = _config()
    config.compile.ep_overlap = SimpleNamespace(module_fqn="layers.*.moe")
    before = _named_pass("before")
    schedule = partial(
        _named_pass("ep_overlap_schedule_pass"),
        module_pattern="layers.*.moe",
        require_all_to_all=True,
        pair_first_token_exchange=True,
    )
    metadata = _named_pass("populate_eager_chunk_metadata_pass")
    terminal = partial(_named_pass("regional_inductor_pass"))
    default_passes = [
        before,
        metadata,
        schedule,
        terminal,
    ]
    factory = Mock(return_value=default_passes)
    monkeypatch.setattr(upstream_passes, "construct_default_graph_passes", factory)
    run_graph = Mock()

    configured = graph_trainer_auto_overlap.construct_npu_auto_overlap_passes(
        traced_result,
        config,
        parallelism_context=parallelism_context,
        run_graph=run_graph,
    )

    factory.assert_called_once_with(traced_result, config, parallelism_context=parallelism_context)
    assert configured[0] is before
    assert configured[1] is metadata
    assert graph_trainer_auto_overlap._pass_name(configured[2]) == "ep_overlap_validate_pass"
    assert configured[2].keywords == schedule.keywords | {"require_all_to_all": False}
    assert configured[3].func is graph_trainer_auto_overlap.npu_moe_auto_overlap_pass
    assert configured[3].keywords["config"] is config
    assert configured[3].keywords["run_graph"] is run_graph
    assert configured[4] is terminal
    assert len(default_passes) == 4  # Do not mutate the upstream pass list.


def test_mutation_functionalization_pipeline_is_composed_before_auto_overlap(
    monkeypatch,
):
    traced_result, parallelism_context = object(), object()
    config = _config()
    functionalize = _named_pass("functionalize_recompute_mutations_pass")
    manual = _named_pass("ep_overlap_schedule_pass")
    metadata = _named_pass("populate_eager_chunk_metadata_pass")
    base_passes = [functionalize, metadata, manual]
    factory = Mock(return_value=base_passes)
    monkeypatch.setitem(
        PASS_PIPELINE_REGISTRY,
        "mutation-functionalization",
        factory,
    )

    configured = graph_trainer_auto_overlap.construct_mutation_functionalization_npu_auto_overlap_passes(
        traced_result,
        config,
        parallelism_context=parallelism_context,
    )

    factory.assert_called_once_with(
        traced_result,
        config,
        parallelism_context=parallelism_context,
    )
    assert [graph_trainer_auto_overlap._pass_name(pass_fn) for pass_fn in configured] == [
        "functionalize_recompute_mutations_pass",
        "populate_eager_chunk_metadata_pass",
        "ep_overlap_validate_pass",
        "npu_moe_auto_overlap_pass",
    ]
    assert base_passes == [functionalize, metadata, manual]


def test_two_whole_graph_rounds_measure_previous_schedule_and_keep_latest_fallback(monkeypatch):
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import (
        npu_moe_auto_scheduler as scheduler_module,
    )
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import whole_graph_benchmark as benchmark_module

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    compute = graph.call_function(torch.ops.aten.neg.default, (x,))
    comm = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (compute, "sum", "test_group"))
    graph.output(comm)
    gm = torch.fx.GraphModule({}, graph)
    schedulers, profiles = [], []
    runner = Mock()

    class Scheduler:
        def __init__(self, candidate, **kwargs):
            assert candidate is gm
            self.kwargs = kwargs
            self.costs_by_node_id = {}
            schedulers.append(self)

        def run(self):
            # A real scheduler calls the cost callbacks synchronously in run().
            self.costs_by_node_id[node_id(compute)] = (
                self.kwargs["compute_cost_fn"](compute) if "compute_cost_fn" in self.kwargs else 10.0
            )
            self.costs_by_node_id[node_id(comm)] = (
                self.kwargs["collective_cost_fn"](comm) if "collective_cost_fn" in self.kwargs else 20.0
            )
            gm.meta["test_schedule_generation"] = len(schedulers)
            return gm

    def profile(candidate, run_candidate, *, profile_node_ids, calibration_round, debug):
        assert candidate is gm
        assert run_candidate is runner
        assert candidate.meta["test_schedule_generation"] == calibration_round
        cid, aid = node_id(compute), node_id(comm)
        assert profile_node_ids == frozenset((cid, aid))
        profiles.append(calibration_round)
        # Round 2 missing collective must reuse round 1's 3ms, not initial 20ms.
        # A measured zero is valid, not a missing mapping.
        costs = {cid: 4.0, aid: 3.0} if calibration_round == 1 else {cid: 0.0}
        return benchmark_module.WholeGraphProfileCosts(costs, costs, frozenset(costs))

    monkeypatch.setattr(scheduler_module, "NpuMoeAutoOverlapScheduler", Scheduler)
    monkeypatch.setattr(benchmark_module, "profile_whole_graph_costs", profile)
    passes = graph_trainer_auto_overlap._configure_npu_auto_overlap_passes(
        [_named_pass("populate_eager_chunk_metadata_pass"), _named_pass("ep_overlap_schedule_pass")],
        _config(),
        run_graph=runner,
    )
    result = passes[2](gm, ())
    assert result is gm
    assert profiles == [1, 2]
    assert len(schedulers) == 3
    assert gm.meta["test_schedule_generation"] == 3
    for scheduler in schedulers[1:]:
        assert scheduler.kwargs["use_profile_compute_costs"] is True
        assert scheduler.kwargs["align_across_ranks"] is False
        assert scheduler.kwargs["canonical_order_override"] is schedulers[0].kwargs["canonical_order_override"]
    assert schedulers[-1].costs_by_node_id[node_id(compute)] == 0.0
    assert schedulers[-1].costs_by_node_id[node_id(comm)] == 3.0
