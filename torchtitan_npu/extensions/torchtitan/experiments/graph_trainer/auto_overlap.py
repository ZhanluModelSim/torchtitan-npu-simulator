# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU auto-overlap pass for GraphTrainer."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    register_pass_pipeline,
)
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.utils import (
    assign_stable_node_tags,
    canonical_order,
    dump_fx_graph,
    node_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable

AUTO_OVERLAP_PIPELINE = "npu_auto_overlap"
MUTATION_FUNCTIONALIZATION_PIPELINE = "mutation-functionalization"
MUTATION_FUNCTIONALIZATION_AUTO_OVERLAP_PIPELINE = "mutation-functionalization+npu_auto_overlap"
_AUTO_OVERLAP_PIPELINES = {
    "default": AUTO_OVERLAP_PIPELINE,
    MUTATION_FUNCTIONALIZATION_PIPELINE: (MUTATION_FUNCTIONALIZATION_AUTO_OVERLAP_PIPELINE),
}
_EP_OVERLAP_SCHEDULE_PASS = "ep_overlap_schedule_pass"
_CONCRETIZE_EP_SHAPES_PASS = "concretize_ep_chunk_symbolic_shapes_pass"
_WHOLE_GRAPH_CALIBRATION_ROUNDS = 2


def _pass_name(pass_fn: Callable) -> str:
    return getattr(getattr(pass_fn, "func", pass_fn), "__name__", "")


def is_npu_auto_overlap_pipeline(name: str) -> bool:
    """Return whether ``name`` selects an NPU auto-overlap composition."""
    return name in _AUTO_OVERLAP_PIPELINES.values()


def npu_moe_auto_overlap_pass(
    gm,
    _example_inputs,
    *,
    config,
    traced_result=None,
    runtime_context=None,
):
    """Benchmark and schedule the configured NPU MoE regions."""
    debug = bool(config.compile.debug_graph_passes)
    if debug:
        dump_fx_graph(gm, prefix="auto_overlap_before")
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.npu_moe_auto_scheduler import (
        NpuMoeAutoOverlapScheduler,
    )
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.whole_graph_benchmark import (
        make_calibration_runner,
        profile_whole_graph_costs,
    )

    ep_overlap_config = getattr(config.compile, "ep_overlap", None)
    module_pattern = getattr(
        ep_overlap_config,
        "module_fqn",
        "layers.*.moe",
    )
    assign_stable_node_tags(gm)
    original_order = canonical_order(gm)

    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern=module_pattern,
        canonical_order_override=original_order,
        debug=debug,
    )
    scheduled = scheduler.run()
    if runtime_context is None:
        logger.warning(
            "Skipped NPU auto-overlap whole-graph calibration because "
            "apply_graph_passes did not provide a runtime context; keeping "
            "the first benchmark-driven schedule"
        )
    else:
        run_candidate = make_calibration_runner(
            traced_result,
            runtime_context,
        )
        for calibration_round in range(1, _WHOLE_GRAPH_CALIBRATION_ROUNDS + 1):
            # Profile and reschedule the graph produced by the previous round.
            # Stable IDs and canonical heuristic order stay fixed.
            fallback_costs = scheduler.costs_by_node_id
            logger.info(
                "NPU auto-overlap calibration round starting: calibration_round=%d total_rounds=%d cost_nodes=%d",
                calibration_round,
                _WHOLE_GRAPH_CALIBRATION_ROUNDS,
                len(fallback_costs),
            )
            profile_costs = profile_whole_graph_costs(
                scheduled,
                run_candidate,
                profile_node_ids=frozenset(fallback_costs),
                calibration_round=calibration_round,
                debug=debug,
            )
            if profile_costs is None:
                logger.warning(
                    "Stopped NPU auto-overlap calibration after round %d; keeping the previous schedule",
                    calibration_round,
                )
                break
            # Whole-graph measurements replace prior costs directly; missing
            # measurements retain the latest available cost.
            applied_costs = fallback_costs | profile_costs.aligned_ms

            def calibrated_cost(
                node,
                *,
                applied_costs=applied_costs,
                fallback_costs=fallback_costs,
                aligned_costs=profile_costs.aligned_ms,
                calibration_round=calibration_round,
            ):
                stable_id = node_id(node)
                if stable_id is None:
                    raise RuntimeError(f"FX node {node.name} has no auto-overlap node ID")
                previous_ms = fallback_costs.get(stable_id, 0.0)
                measured_ms = aligned_costs.get(stable_id)
                cost = applied_costs.get(stable_id, 0.0)
                if debug:
                    logger.info(
                        "NPU auto-overlap calibration applied cost: "
                        "calibration_round=%d schedule_round=%d node_id=%s "
                        "node=%s target=%s previous_ms=%.6f scheduler_ms=%.6f "
                        "delta_ms=%.6f source=%s",
                        calibration_round,
                        calibration_round + 1,
                        stable_id,
                        node.name,
                        node.target,
                        previous_ms,
                        cost,
                        cost - previous_ms,
                        "previous_round_fallback" if measured_ms is None else "whole_graph_profile",
                    )
                return cost

            scheduler = NpuMoeAutoOverlapScheduler(
                scheduled,
                module_pattern=module_pattern,
                compute_cost_fn=calibrated_cost,
                use_profile_compute_costs=True,
                collective_cost_fn=calibrated_cost,
                align_across_ranks=False,
                canonical_order_override=original_order,
                debug=debug,
            )
            # Callbacks are consumed synchronously here, before loop state
            # changes. No single-op benchmark is rerun in calibrated rounds.
            scheduled = scheduler.run()
            next_costs = scheduler.costs_by_node_id
            logger.info(
                "NPU auto-overlap calibration round completed: "
                "calibration_round=%d measured_nodes=%d aligned_nodes=%d "
                "fallback_nodes=%d",
                calibration_round,
                len(profile_costs.local_ms),
                len(profile_costs.aligned_ms),
                len(next_costs.keys() - profile_costs.aligned_ms.keys()),
            )
    scheduled.graph.lint()
    if debug:
        dump_fx_graph(scheduled, prefix="auto_overlap_after")
    return scheduled


def _configure_npu_auto_overlap_passes(
    passes: list[Callable],
    config,
    *,
    traced_result=None,
    runtime_context=None,
) -> list[Callable]:
    """Insert the only supported NPU MoE scheduler pass into ``passes``."""
    precompile_artifact_dir = getattr(
        config.compile,
        "precompile_artifact_dir",
        "",
    )
    if precompile_artifact_dir:
        logger.warning(
            "Skipped NPU auto-overlap because a precompiled artifact is "
            "being loaded; keeping the original runtime graph-pass pipeline"
        )
        return passes

    schedule_passes = [pass_fn for pass_fn in passes if _pass_name(pass_fn) == _EP_OVERLAP_SCHEDULE_PASS]
    concretize_passes = [pass_fn for pass_fn in passes if _pass_name(pass_fn) == _CONCRETIZE_EP_SHAPES_PASS]
    if len(schedule_passes) != 1 or len(concretize_passes) != 1:
        logger.warning(
            "Skipped NPU auto-overlap because its required pass pipeline "
            "is unavailable: %s=%d, %s=%d; keeping the "
            "original graph-pass pipeline",
            _EP_OVERLAP_SCHEDULE_PASS,
            len(schedule_passes),
            _CONCRETIZE_EP_SHAPES_PASS,
            len(concretize_passes),
        )
        return passes

    schedule_index = passes.index(schedule_passes[0])
    concretize_index = passes.index(concretize_passes[0])
    if schedule_index >= concretize_index:
        logger.warning(
            "Skipped NPU auto-overlap because the original pass order is "
            "unsupported: %s must precede %s; keeping the original "
            "graph-pass pipeline",
            _EP_OVERLAP_SCHEDULE_PASS,
            _CONCRETIZE_EP_SHAPES_PASS,
        )
        return passes

    # Install the compatibility patch only after confirming that this pipeline
    # will use auto-overlap. Remove it after the torch_npu support lands.
    from torchtitan_npu.patches.torch_npu import inductor_benchmarking

    inductor_benchmarking.apply()

    pass_kwargs = {
        "config": config,
        "traced_result": traced_result,
    }
    if runtime_context is not None:
        pass_kwargs["runtime_context"] = runtime_context
    configured_pass = functools.partial(npu_moe_auto_overlap_pass, **pass_kwargs)
    if runtime_context is None:
        # Compatibility with TorchTitan revisions that do not pass runtime
        # context to pipeline construction. The local patch binds it later.
        configured_pass._requires_runtime_context = True  # pyrefly: ignore [missing-attribute]

    # Retain the native EP structural checks at the original scheduling
    # position. Runtime benchmarking requires concrete tensor dimensions, so
    # defer the NPU scheduler until after concretization.
    from torchtitan.experiments.graph_trainer.ep_overlap_pass import (
        ep_overlap_validate_pass,
    )

    schedule_pass = schedule_passes[0]
    if isinstance(schedule_pass, functools.partial):
        validation_kwargs = dict(schedule_pass.keywords or {})
        # The pinned validator uses ``require_all_to_all`` only on its
        # scheduling branch and raises when ``reorder=False``. The NPU
        # scheduler below independently requires a matching chunked region.
        validation_kwargs["require_all_to_all"] = False
        validation_pass = functools.partial(
            ep_overlap_validate_pass,
            *schedule_pass.args,
            **validation_kwargs,
        )
    else:
        validation_pass = ep_overlap_validate_pass
    configured = [validation_pass if pass_fn is schedule_pass else pass_fn for pass_fn in passes]
    concretize_index = configured.index(concretize_passes[0])
    configured.insert(concretize_index + 1, configured_pass)
    return configured


def enable_npu_auto_overlap(config):
    """Select the NPU auto-overlap composition for the current base pipeline."""
    base_pipeline = config.compile.pass_pipeline
    try:
        config.compile.pass_pipeline = _AUTO_OVERLAP_PIPELINES[base_pipeline]
    except KeyError as error:
        supported = ", ".join(sorted(_AUTO_OVERLAP_PIPELINES))
        raise ValueError(
            f"NPU auto-overlap does not support base pass pipeline {base_pipeline!r}; supported pipelines: {supported}"
        ) from error
    return config


def _construct_npu_auto_overlap_pipeline(
    base_pipeline: str,
    traced_result,
    config,
    *,
    parallel_dims=None,
    runtime_context=None,
) -> list[Callable]:
    """Build ``base_pipeline`` with EP validation and NPU scheduling."""
    if base_pipeline == "default":
        from torchtitan.experiments.graph_trainer.passes import (
            construct_default_graph_passes,
        )

        pipeline_fn = construct_default_graph_passes
    else:
        pipeline_fn = PASS_PIPELINE_REGISTRY.get(base_pipeline)
        if pipeline_fn is None:
            raise ValueError(f"Base pass pipeline {base_pipeline!r} is not registered")

    passes = pipeline_fn(
        traced_result,
        config,
        parallel_dims=parallel_dims,
    )
    return _configure_npu_auto_overlap_passes(
        passes,
        config,
        traced_result=traced_result,
        runtime_context=runtime_context,
    )


@register_pass_pipeline(AUTO_OVERLAP_PIPELINE)
def construct_npu_auto_overlap_passes(
    traced_result,
    config,
    *,
    parallel_dims=None,
    runtime_context=None,
) -> list[Callable]:
    """Build the default pipeline with the NPU MoE scheduler as its EP pass."""
    return _construct_npu_auto_overlap_pipeline(
        "default",
        traced_result,
        config,
        parallel_dims=parallel_dims,
        runtime_context=runtime_context,
    )


# Compatibility composition for the local functionalization backport in
# torchtitan_npu/patches/torchtitan/experiments/graph_trainer/
# functionalize_recompute_mutations.py.
# Remove this pipeline after pytorch/torchtitan#4708 lands in the pinned
# TorchTitan dependency with equivalent pipeline support.
@register_pass_pipeline(MUTATION_FUNCTIONALIZATION_AUTO_OVERLAP_PIPELINE)
def construct_mutation_functionalization_npu_auto_overlap_passes(
    traced_result,
    config,
    *,
    parallel_dims=None,
    runtime_context=None,
) -> list[Callable]:
    """Compose the local mutation-functionalization pipeline with the NPU scheduler."""
    return _construct_npu_auto_overlap_pipeline(
        MUTATION_FUNCTIONALIZATION_PIPELINE,
        traced_result,
        config,
        parallel_dims=parallel_dims,
        runtime_context=runtime_context,
    )
