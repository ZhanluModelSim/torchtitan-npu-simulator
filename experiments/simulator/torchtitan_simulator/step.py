# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture a native training step as L0-L3 graphs and logical memory events."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from torchtitan_simulator.capture.checkpoint_execution import install_checkpoint_execution_tracking
from torchtitan_simulator.capture.comm_events import observe_collectives
from torchtitan_simulator.capture.comm_group_resolver import resolve_comm_event_groups
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.graph_normalization import fold_metadata_views
from torchtitan_simulator.capture.module_path import ModulePathTracker
from torchtitan_simulator.capture.schedule_builder import (
    build_schedule_plan,
    project_schedule_plan_to_graph,
)
from torchtitan_simulator.capture.step_boundary import StepBoundaryTracker, build_step_graphs
from torchtitan_simulator.capture.workload_builder import build_workload_graph
from torchtitan_simulator.memory.schedule_replay import estimate_schedule_memory
from torchtitan_simulator.meta_env import patch_device_type_to_meta
from torchtitan_simulator.rank_table import build_rank_table

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable

    from torchtitan_simulator.ir.workload_graph import WorkloadGraph


def _l0_csv_filename(
    template_id: str,
    step_type: str,
    step_type_counts: Counter[str],
) -> str:
    label = step_type
    if step_type_counts[step_type] > 1:
        label = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_" for character in template_id
        )
    return f"step_{label}_l0_ops.csv"


def run_simulation_step(
    *,
    model_parts: list[nn.Module],
    parallel_dims: Any,
    forward_backward_step: Callable[..., torch.Tensor],
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    optimizer_step: Callable[[], None],
    lr_scheduler_step: Callable[[], None],
    local_batch_size: int,
    seq_len: int,
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
    rank: int = 0,
    capture_process_rank: int | None = None,
    enable_memory_tracking: bool = True,
    memory_parameter_storage_dtype: str = "",
    memory_device_type: str = "",
    memory_offload_ac_saved_tensors: bool = False,
    pp_schedule: Any | None = None,
    microbatches: list[tuple[dict[str, torch.Tensor], torch.Tensor]] | None = None,
    global_valid_tokens: float | torch.Tensor | None = None,
    local_valid_tokens: int | None = None,
    optimizer_zero_grad: Callable[[], None] | None = None,
    clip_grad_norm: Callable[[], Any] | None = None,
    scoped_backward_phases: bool = False,
) -> WorkloadGraph:
    """Capture one non-PP microbatch, or the configured PP schedule, and
    return the resulting four-layer WorkloadGraph. Bypasses
    `Trainer.train_step()` deliberately: that method's `dist_sum`-based
    token counting and loss/grad-norm logging both call `.item()` on
    device tensors, which raises under meta-device execution (see design
    doc §9) -- `global_valid_tokens` is instead supplied here as a plain
    Python float derived from the static input shape.

    Calls `patch_device_type_to_meta()` unconditionally (idempotent) so
    this function is safe to call standalone -- not just via
    `SimulationTrainer.__init__` -- since `optimizer_step()` alone is
    enough to trigger a real `torch_npu` hardware probe otherwise (see
    `meta_env._neutralize_torch_npu_optimizer_device_probe`).
    """
    import time

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    if parallel_dims.pp_enabled and gradient_accumulation > 1:
        raise ValueError(
            "PP gradient accumulation across multiple schedule steps is not supported by simulator capture"
        )
    configured_gradient_accumulation = gradient_accumulation
    if microbatches is None:
        microbatches = [(input_dict, labels)]
    if not microbatches:
        raise ValueError("Supply at least one representative microbatch")
    capture_metadata = {}
    if not parallel_dims.pp_enabled:
        input_dict, labels = microbatches[0]
        microbatches = [(input_dict, labels)]
        num_micro_batches = configured_gradient_accumulation
        capture_metadata = {
            "scope": "single_microbatch",
            "captured_microbatches": 1,
            "configured_microbatches": configured_gradient_accumulation,
            "configured_gradient_accumulation": configured_gradient_accumulation,
            "microbatch_batch_size": local_batch_size,
        }
        if configured_gradient_accumulation > 1:
            import logging

            logging.getLogger(__name__).warning(
                "Non-PP capture executes one microbatch (configured GA=%s); "
                "multi-MB time scaling is left to downstream consumers.",
                configured_gradient_accumulation,
            )
    elif len(microbatches) != gradient_accumulation:
        raise ValueError("microbatches must match gradient_accumulation")
    if global_valid_tokens is None and local_valid_tokens is None:
        from torchtitan.components.loss import IGNORE_INDEX

        if any(target.device.type == "meta" for _, target in microbatches):
            raise ValueError("Meta labels require an explicit local_valid_tokens or global_valid_tokens count")
        local_valid_tokens = sum(int((target != IGNORE_INDEX).sum().item()) for _, target in microbatches)
    if global_valid_tokens is None and not getattr(parallel_dims, "dp_enabled", False):
        global_valid_tokens = float(local_valid_tokens or 0)
    if isinstance(global_valid_tokens, (int, float)) and global_valid_tokens <= 0:
        raise ValueError("A simulation step requires at least one valid target token")

    patch_device_type_to_meta()
    install_checkpoint_execution_tracking(model_parts)

    # Default PP stage attribution: non-PP steps use stage 0 (the single
    # stage); PP steps use -1 ("unattributed") so framework setup ops captured
    # outside any compute chunk (FSDP _lazy_init / inter-chunk comm, which run
    # before the first forward_one_chunk stamps the real stage) bucket into a
    # clearly-labelled `s-1_*` template instead of masquerading as stage 0.
    # `pp_enabled` (pp_degree > 1) is the real signal — the schedule string
    # is "1F1B" even when PP degree is 1, so it cannot gate this.
    from torchtitan_simulator.meta_env import _pp_context

    _pp_context["stage"] = -1 if parallel_dims.pp_enabled else 0
    _pp_context.update(mb_idx=0, comp_type="F", phase="forward")

    boundary = StepBoundaryTracker(scoped_backward=scoped_backward_phases)
    # Interleaved schedules such as DualPipeV place multiple virtual stages
    # on one physical rank. Track every local model part so the second stage's
    # forward/backward ops retain stable global module FQNs as well.
    module_path_tracker = ModulePathTracker(model_parts)

    # Phase provider: when PP is active, _pp_context["phase"] is updated by
    # the patched forward_one_chunk / backward_one_chunk (which is called
    # for every microbatch).  StepBoundaryTracker.current_phase is only
    # updated by Tensor.backward / Optimizer.step hooks, but PP uses
    # torch.autograd.backward (not Tensor.backward), so the tracker misses
    # the backward phase.  Reading _pp_context["phase"] when it differs
    # from the tracker gives us the correct phase for every captured op.
    def _phase_provider() -> str:
        from torchtitan_simulator.meta_env import _pp_context

        # When boundary says "optimizer" (set by boundary.mark("optimizer")),
        # always return "optimizer" — _pp_context may still say "backward"
        # from the last backward_one_chunk call.
        if boundary.current_phase == "optimizer":
            return "optimizer"
        # During forward_backward_step, _pp_context is updated by the
        # patched forward_one_chunk / backward_one_chunk.  Use it to
        # detect backward phase (PP uses autograd.backward, which
        # bypasses the Tensor.backward hook).
        pp_phase = _pp_context.get("phase", "")
        if pp_phase == "backward":
            return "backward"
        return boundary.current_phase

    capture = OpDispatchCapture(
        module_path_tracker=module_path_tracker,
        phase_provider=_phase_provider,
        record_memory=enable_memory_tracking,
    )

    t1 = time.perf_counter()
    timings["setup"] = t1 - t0

    from contextlib import nullcontext

    from torchtitan_simulator.capture.comm_events import (
        default_collective_context,
    )
    from torchtitan_simulator.capture.saved_tensors import (
        AutogradSavedTensorCapture,
    )

    get_optional_mesh = getattr(parallel_dims, "get_optional_mesh", None)
    tp_mesh = get_optional_mesh("tp") if callable(get_optional_mesh) else None
    saved_tensor_context = AutogradSavedTensorCapture() if enable_memory_tracking else nullcontext()
    with (
        observe_collectives(
            memory_tracking_enabled=enable_memory_tracking,
            capture_process_rank=rank if capture_process_rank is None else capture_process_rank,
        ) as comm_recorder,
        boundary,
        module_path_tracker,
        capture,
        saved_tensor_context,
    ):
        capture.record_initial_memory_tensors((input_dict, labels, global_valid_tokens))
        if optimizer_zero_grad is not None:
            optimizer_zero_grad()
        else:
            for part in model_parts:
                part.zero_grad()
        boundary.mark("forward")
        if global_valid_tokens is None:
            from torch.distributed import _functional_collectives as funcol

            token_count = torch.tensor(local_valid_tokens, dtype=torch.int64, device=labels.device)
            global_valid_tokens = funcol.all_reduce(token_count, "sum", parallel_dims.get_mesh("batch")).float()
        for mb_idx, (batch_inputs, batch_labels) in enumerate(microbatches):
            boundary.mark("forward")
            if not parallel_dims.pp_enabled:
                _pp_context.update(mb_idx=mb_idx, comp_type="F", phase="forward")
            with (
                default_collective_context("tp", tp_mesh.get_group()) if tp_mesh is not None else nullcontext(),
            ):
                forward_backward_step(
                    input_dict=batch_inputs,
                    labels=batch_labels,
                    global_valid_tokens=global_valid_tokens,
                )
        t2 = time.perf_counter()
        timings["forward_backward"] = t2 - t1
        capture.finalize_autograd_saved_tensors()

        boundary.mark("optimizer")
        # Always capture L0 for optimizer phase (not controlled by microbatch)
        capture._capture_l0 = True
        if clip_grad_norm is not None:
            clip_grad_norm()
        from torchtitan_simulator.capture.optimizer import (
            capture_optimizer_param_groups,
        )

        with capture_optimizer_param_groups(optimizer_step):
            optimizer_step()
        lr_scheduler_step()
        t3 = time.perf_counter()
        timings["optimizer"] = t3 - t2

    t4 = time.perf_counter()
    rank_table = build_rank_table(parallel_dims)
    group_names_by_op_id = resolve_comm_event_groups(
        comm_recorder.events,
        rank_table,
    )
    capture.apply_comm_group_names(group_names_by_op_id)
    timings["build_rank_table"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    nodes = capture.build_nodes()
    native_dispatch_ids = [
        node.op_id
        for node in nodes.values()
        if not node.annotations.get("metadata_view")
        and str(node.annotations.get("raw_op_type", "")).startswith(
            ("aten.", "npu.", "cann_ops_", "fsdp.", "c10d.", "_dtensor.", "_c10d_functional.")
        )
    ]
    compute_instance_events = capture.compute_instance_events()
    capture.release_operator_tracking()
    nodes = fold_metadata_views(nodes)
    timings["build_nodes"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    step_templates = build_step_graphs(nodes)
    timings["build_step_graphs"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    schedule_plan = build_schedule_plan(
        step_templates=step_templates,
        rank_table=rank_table,
        comm_events=comm_recorder.events,
        fsdp_residency_events=comm_recorder.fsdp_schedule_events,
        timeline_events=comm_recorder.timeline_events,
        pp_schedule_obj=pp_schedule,
        pipeline_schedule=pipeline_schedule,
        num_micro_batches=num_micro_batches,
        gradient_accumulation=gradient_accumulation,
        rank=rank,
        captured_trace_primary=True,
        compute_instance_events=compute_instance_events if not parallel_dims.pp_enabled else None,
    )
    timings["build_schedule_plan"] = time.perf_counter() - t7

    # Schedule assembly knows the logical rank. The recorder also knows the
    # physical capture worker, which can differ in a single-worker simulation.
    worker_rank = rank if capture_process_rank is None else capture_process_rank
    schedule_plan.annotations["capture_process_rank"] = worker_rank
    schedule_plan.annotations["logical_rank"] = rank
    for action in schedule_plan.action_map.values():
        action.annotations["capture_process_rank"] = worker_rank
        action.annotations["logical_rank"] = rank

    # Ownership normalization refreshes each template independently. Restore
    # outgoing cross-template edges from the authoritative predecessor DAG.
    graph_nodes = {op_id: node for template in step_templates.values() for op_id, node in template.nodes.items()}
    for node in graph_nodes.values():
        node.successors = []
    for op_id, node in graph_nodes.items():
        for predecessor in node.predecessors:
            if predecessor in graph_nodes:
                graph_nodes[predecessor].successors.append(op_id)

    t7b = time.perf_counter()
    schedule_graph = project_schedule_plan_to_graph(
        schedule_plan=schedule_plan,
        rank_table=rank_table,
    )
    timings["build_schedule_graph"] = time.perf_counter() - t7b

    t8 = time.perf_counter()
    wg = build_workload_graph(
        schedule_graph=schedule_graph,
        step_templates=step_templates,
        local_batch_size=local_batch_size,
        seq_len=seq_len,
        num_micro_batches=num_micro_batches,
        schedule_plan=schedule_plan,
    )
    if capture_metadata:
        schedule_plan.annotations["capture"] = capture_metadata
        wg.iteration.schedule.annotations["capture"] = capture_metadata
    wg.iteration.schedule.annotations["communication_events"] = [asdict(event) for event in comm_recorder.events]
    wg.iteration.schedule.annotations["native_dispatch_op_ids"] = native_dispatch_ids
    wg.iteration.schedule.annotations["suppressed_dispatch_counts"] = [
        {"scope": scope, "operator": operator, "count": count}
        for (scope, operator), count in sorted(capture.suppressed_dispatch_counts.items())
    ]
    timings["build_workload_graph"] = time.perf_counter() - t8

    memory_plan = None
    if enable_memory_tracking:
        t9 = time.perf_counter()
        memory_plan = estimate_schedule_memory(
            capture.memory_events(),
            schedule_plan=schedule_plan,
            model_parts=model_parts,
            comm_events=comm_recorder.events,
            fsdp_residency_events=comm_recorder.fsdp_residency_events,
            checkpoint_boundary_events=capture.checkpoint_boundary_events(),
            autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
            storage_lifetimes=capture.storage_lifetime_events(),
            initial_tensors=capture.initial_memory_tensors(),
            counted_device_type=memory_device_type,
            parameter_storage_dtype=memory_parameter_storage_dtype or None,
            offload_ac_saved_tensors=memory_offload_ac_saved_tensors,
        )
        if capture_metadata:
            memory_plan.capture_metadata = capture_metadata
            memory_plan.notes.append(
                "Non-PP memory describes one microbatch plus one optimizer update; configured GA is not expanded."
            )
        wg.iteration.schedule.annotations["memory_plan"] = memory_plan
        wg.iteration.schedule.annotations["memory_summary"] = memory_plan.to_summary_dict()
        for step_graph in wg.step_templates.values():
            step_graph.param_mem = memory_plan.persistent_param_bytes
            step_graph.peak_active_mem = memory_plan.model_active_bytes_peak
            step_graph.annotations["memory_summary"] = memory_plan.to_summary_dict()
        timings["estimate_memory"] = time.perf_counter() - t9

    timings["total"] = time.perf_counter() - t0

    # Print timing table
    print("\n" + "=" * 60)
    print("Simulation Step Timing Breakdown")
    print("=" * 60)
    print(f"{'Stage':<30} {'Time (s)':>10} {'%':>8}")
    print("-" * 60)
    total = timings["total"]
    for name in [
        "setup",
        "forward_backward",
        "optimizer",
        "build_rank_table",
        "build_nodes",
        "build_step_graphs",
        "build_schedule_graph",
        "build_schedule_plan",
        "build_workload_graph",
        "estimate_memory",
    ]:
        if name not in timings:
            continue
        t = timings[name]
        print(f"{name:<30} {t:>10.2f} {t / total * 100:>7.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total:>10.2f} {'100.0%':>8}")
    print("=" * 60)
    print(f"Captured ops: {len(nodes)}, comm events: {len(comm_recorder.events)}")
    execution_summary: dict[str, dict[str, int]] = {}
    for node in nodes.values():
        kind = node.annotations.get("execution_kind", "unknown")
        repeat_count = int(node.annotations.get("repeat_count", 1))
        summary = execution_summary.setdefault(kind, {"nodes": 0, "invocations": 0, "flops": 0})
        summary["nodes"] += 1
        summary["invocations"] += repeat_count
        summary["flops"] += node.flops * repeat_count
    print(
        "Execution kinds: "
        + ", ".join(
            f"{kind}=nodes:{values['nodes']}/calls:{values['invocations']}/flops:{values['flops']}"
            for kind, values in execution_summary.items()
        )
    )
    print(f"Step templates: {list(step_templates.keys())}")
    communication_ownership = schedule_plan.annotations.get(
        "communication_ownership",
        {},
    )
    if communication_ownership:
        print(
            "Communication ownership: "
            f"L1_collectives={communication_ownership.get('stage_owned_collectives', 0)} "
            f"L1_fsdp_transitions={communication_ownership.get('internal_fsdp_transitions', 0)} "
            f"L2_fsdp_prefetches={communication_ownership.get('external_fsdp_prefetches', 0)} "
            f"L1_gradient_reductions={communication_ownership.get('internal_gradient_reductions', 0)} "
            f"L2_gradient_reductions={communication_ownership.get('external_gradient_reductions', 0)} "
            f"removed_noop_gradient_intents={communication_ownership.get('removed_noop_gradient_intents', 0)}"
        )
    if memory_plan is not None:
        print(
            "Static memory: "
            f"persistent_param_bytes={memory_plan.persistent_param_bytes} "
            f"active_bytes_peak={memory_plan.peak_active_bytes} "
            f"model_active_bytes_peak={memory_plan.model_active_bytes_peak} "
            f"forward_peak={memory_plan.forward_peak_active_bytes} "
            f"backward_peak={memory_plan.backward_peak_active_bytes} "
            f"optimizer_peak={memory_plan.optimizer_peak_active_bytes} "
            f"peak_seq_idx={memory_plan.peak_seq_idx}"
        )
    else:
        print("Static memory tracking: disabled")
    print()

    return wg
