# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4763
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport the candidate runner interface for PP1 joint graphs.

Remove this module after the pinned TorchTitan supplies a candidate runner.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import torchtitan.experiments.graph_trainer.graph_builder
from torchtitan.experiments.graph_trainer import spmd_graph_builder

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def apply(*, uses_candidate_runner: Callable[[str], bool], make_runner: Callable) -> None:
    """Bind the runner explicitly in the selected live joint-graph pipelines."""
    current = torchtitan.experiments.graph_trainer.graph_builder._build_fwd_bwd_graphs
    if getattr(current, "_npu_candidate_runner_patch", False) is True:
        return

    @functools.wraps(current)
    def patched_build_fwd_bwd_graphs(
        stage,
        args,
        kwargs,
        target,
        loss_kwargs,
        *,
        loss_fn,
        trainer_config,
        parallelism_context,
    ):
        compile_config = trainer_config.compile
        pipeline_name = compile_config.pass_pipeline
        pipeline_fn = spmd_graph_builder.PASS_PIPELINE_REGISTRY.get(pipeline_name)
        if (
            pipeline_fn is None
            or not uses_candidate_runner(pipeline_name)
            or not compile_config.enable_passes
            or compile_config.precompile_artifact_dir
        ):
            return current(
                stage,
                args,
                kwargs,
                target,
                loss_kwargs,
                loss_fn=loss_fn,
                trainer_config=trainer_config,
                parallelism_context=parallelism_context,
            )

        traced, runtime_meshes, num_param_grads = spmd_graph_builder._trace_spmd_stage_graph(
            stage,
            args,
            kwargs,
            target,
            loss_kwargs,
            loss_fn=loss_fn,
            compile_config=compile_config,
            parallelism_context=parallelism_context,
        )
        model = stage.submod
        runtime_args = (args[0], target, loss_kwargs["global_loss_token_counts"], kwargs)
        run_graph = make_runner(
            traced,
            model=model,
            training_args=runtime_args,
            train_context=functools.partial(
                parallelism_context.activate_spmd, typechecking=trainer_config.debug.spmd_typechecking
            ),
        )
        passes = pipeline_fn(traced, trainer_config, parallelism_context=parallelism_context, run_graph=run_graph)
        traced.gm = spmd_graph_builder.apply_graph_passes(
            traced.gm,
            traced.example_inputs,
            passes,
            compile_config=compile_config,
            respect_disable_passes=compile_config.enable_passes,
        )
        stage.graphs = spmd_graph_builder.GraphTrainerJointStageGraphs(
            traced=traced,
            module=model,
            num_param_grads=num_param_grads,
            runtime_meshes=runtime_meshes,
        )

    patched_build_fwd_bwd_graphs._npu_candidate_runner_patch = True  # pyrefly: ignore [missing-attribute]
    torchtitan.experiments.graph_trainer.graph_builder._build_fwd_bwd_graphs = patched_build_fwd_bwd_graphs
    logger.info("[PATCH] GraphTrainer PP1 joint graph candidate runner enabled")
