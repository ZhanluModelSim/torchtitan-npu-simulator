# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test explicit candidate runner binding for NPU graph passes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torchtitan.experiments.graph_trainer import graph_builder, spmd_graph_builder
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import graph_trainer_runtime_context


def test_joint_graph_build_passes_candidate_runner_to_pipeline(monkeypatch):
    model = SimpleNamespace(named_parameters=lambda **_: ())
    stage = SimpleNamespace(submod=model, is_first=True, is_last=True)
    inputs, labels, tokens = object(), object(), object()
    extra_kwargs = {"metadata": object()}
    activate_spmd = Mock(side_effect=lambda **_: nullcontext())
    parallelism = SimpleNamespace(activate_spmd=activate_spmd)
    compile_config = GraphTrainerCompileConfig(pass_pipeline="test_pipeline")
    config = SimpleNamespace(compile=compile_config, debug=SimpleNamespace(spmd_typechecking=True))
    traced = SimpleNamespace(gm=torch.fx.symbolic_trace(torch.nn.Identity()), example_inputs=())
    run_graph = Mock()

    def make_runner(result, *, model, training_args, train_context):
        assert result is traced
        assert model is stage.submod
        assert training_args == (inputs, labels, tokens, extra_kwargs)
        with train_context():
            return run_graph

    original = Mock()
    pipeline = Mock(return_value=[])
    executor = Mock()
    monkeypatch.setattr(graph_builder, "_build_fwd_bwd_graphs", original)
    monkeypatch.setattr(spmd_graph_builder, "minimal_fx_tracer", Mock(return_value=Mock(return_value=traced)))
    monkeypatch.setattr(spmd_graph_builder, "GraphTrainerJointStageGraphs", executor)
    monkeypatch.setitem(spmd_graph_builder.PASS_PIPELINE_REGISTRY, "test_pipeline", pipeline)
    graph_trainer_runtime_context.apply(
        uses_candidate_runner=lambda name: name == "test_pipeline",
        make_runner=make_runner,
    )

    graph_builder._build_fwd_bwd_graphs(
        stage,
        (inputs,),
        extra_kwargs,
        labels,
        {"global_loss_token_counts": tokens},
        loss_fn=Mock(),
        parallelism_context=parallelism,
        trainer_config=config,
    )

    pipeline.assert_called_once_with(traced, config, parallelism_context=parallelism, run_graph=run_graph)
    activate_spmd.assert_called_once_with(typechecking=True)
    original.assert_not_called()
    assert stage.graphs is executor.return_value
