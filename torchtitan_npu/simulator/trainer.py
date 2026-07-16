# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimulationTrainer: a Trainer subclass that captures one training step's
four-layer IR (L0-L3) instead of running a full multi-step training loop,
with zero real NPU hardware and zero real memory allocation. See
docs/superpowers/specs/2026-07-01-npu-simulator-design.md."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torchtitan.trainer import Trainer

from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.module_path import ModulePathTracker
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_graph, build_schedule_plan
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker, build_step_graphs
from torchtitan_npu.simulator.capture.workload_builder import build_workload_graph
from torchtitan_npu.simulator.hardware_shims.mhc_converter import apply_mhc_shims
from torchtitan_npu.simulator.hardware_shims.smla_converter import apply_smla_shims
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph
from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta
from torchtitan_npu.simulator.moe_force_balance import force_deterministic_seed, force_moe_load_balance
from torchtitan_npu.simulator.rank_table import build_rank_table
from torchtitan_npu.simulator.viz.csv_export import export_kernel_summary_csv
from torchtitan_npu.simulator.viz.dot_export import export_dot
from torchtitan_npu.simulator.viz.html_export import export_html
from torchtitan_npu.simulator.viz.json_export import export_json
from torchtitan_npu.simulator.viz.text_summary import write_text_summary


@dataclass(kw_only=True, slots=True)
class SimulationConfig:
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(default_factory=lambda: [])
    target_npu_device_type: str = "non_a5"
    csv_max_ranks: int | None = None
    simulated_parallel_degrees: dict[str, int] = field(default_factory=dict)
    enable_fusion: bool = False


@dataclass(kw_only=True, slots=True)
class SimulationTrainerConfig(Trainer.Config):
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# Model converters that unconditionally require real accelerator hardware to
# run their forward computation (a Triton-JIT kernel launch or a JIT-compiled
# aclnn extension, either gated behind a private "custom_ops" module
# unavailable outside Huawei's internal build), with no meta-device-compatible
# fallback other than the model's own base (pre-conversion) module -- see
# `_strip_hardware_dependent_model_converters`.
# npu_mhc_pre/npu_mhc_post: no longer stripped -- SimulationTrainer.__init__ calls
# apply_mhc_shims() (torchtitan_npu.simulator.hardware_shims.mhc_converter) to install
# SimHcPre/SimHcHead/SimHcPost instead, preserving the real op names in the captured graph. See
# docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md.
# npu_smla: no longer stripped either -- SimulationTrainer.__init__ calls apply_smla_shims()
# (torchtitan_npu.simulator.hardware_shims.smla_converter) to install SimNpuSparseAttention/
# SimNpuLiCompute/SimNpuLiLoss instead. See
# docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md.
_HARDWARE_DEPENDENT_CONVERTER_NAMES = frozenset()


def _strip_hardware_dependent_model_converters(config: Any) -> None:
    """Remove `_HARDWARE_DEPENDENT_CONVERTER_NAMES` from
    `config.model_converters.converters` so `model_converters.convert(model)`
    never invokes them, leaving each affected submodule on its BASE class --
    verified by reading their forward() methods to be pure-PyTorch (no
    custom kernel, no Triton) implementations that run correctly on meta
    tensors (module-level comment above has the per-converter detail).
    Uses the same `_owner`/`_model_config.name` introspection as this
    repo's own `torchtitan_npu.converters.registry.has_npu_converter`."""
    model_converters_config = getattr(config, "model_converters", None)
    converters = getattr(model_converters_config, "converters", None) if model_converters_config else None
    if not converters:
        return

    kept = []
    for converter_config in converters:
        owner = getattr(converter_config, "_owner", None) or converter_config
        model_config = getattr(owner, "_model_config", None)
        name = getattr(model_config, "name", None) if model_config is not None else None
        if name not in _HARDWARE_DEPENDENT_CONVERTER_NAMES:
            kept.append(converter_config)
    model_converters_config.converters = kept



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
    pp_schedule: Any | None = None,
) -> WorkloadGraph:
    """Run one forward+backward+optimizer step under full capture and
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

    patch_device_type_to_meta()
    global_valid_tokens = float(labels.numel())

    # Default PP stage attribution: non-PP steps use stage 0 (the single
    # stage); PP steps use -1 ("unattributed") so framework setup ops captured
    # outside any compute chunk (FSDP _lazy_init / inter-chunk comm, which run
    # before the first forward_one_chunk stamps the real stage) bucket into a
    # clearly-labelled `s-1_*` template instead of masquerading as stage 0.
    # `pp_enabled` (pp_degree > 1) is the real signal — the schedule string
    # is "1F1B" even when PP degree is 1, so it cannot gate this.
    from torchtitan_npu.simulator.meta_env import _pp_context
    _pp_context["stage"] = -1 if parallel_dims.pp_enabled else 0

    boundary = StepBoundaryTracker()
    module_path_tracker = ModulePathTracker(model_parts[0])

    # Phase provider: when PP is active, _pp_context["phase"] is updated by
    # the patched forward_one_chunk / backward_one_chunk (which is called
    # for every microbatch).  StepBoundaryTracker.current_phase is only
    # updated by Tensor.backward / Optimizer.step hooks, but PP uses
    # torch.autograd.backward (not Tensor.backward), so the tracker misses
    # the backward phase.  Reading _pp_context["phase"] when it differs
    # from the tracker gives us the correct phase for every captured op.
    def _phase_provider() -> str:
        from torchtitan_npu.simulator.meta_env import _pp_context
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

    capture = OpDispatchCapture(module_path_tracker=module_path_tracker, phase_provider=_phase_provider)

    t1 = time.perf_counter()
    timings["setup"] = t1 - t0

    with capture_fake_collectives() as comm_recorder, boundary, module_path_tracker, capture:
        boundary.mark("forward")
        forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )
        t2 = time.perf_counter()
        timings["forward_backward"] = t2 - t1

        boundary.mark("optimizer")
        # Always capture L0 for optimizer phase (not controlled by microbatch)
        capture._capture_l0 = True
        optimizer_step()
        lr_scheduler_step()
        t3 = time.perf_counter()
        timings["optimizer"] = t3 - t2

    t4 = time.perf_counter()
    nodes = capture.build_nodes()
    timings["build_nodes"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    step_templates = build_step_graphs(nodes)
    timings["build_step_graphs"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    rank_table = build_rank_table(parallel_dims)
    timings["build_rank_table"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    schedule_graph = build_schedule_graph(
        step_templates=step_templates,
        rank_table=rank_table,
        comm_events=comm_recorder.events,
        timeline_events=comm_recorder.timeline_events,
        pipeline_schedule=pipeline_schedule,
        num_micro_batches=num_micro_batches,
        gradient_accumulation=gradient_accumulation,
        rank=rank,
    )
    timings["build_schedule_graph"] = time.perf_counter() - t7

    t7b = time.perf_counter()
    schedule_plan = build_schedule_plan(
        step_templates=step_templates,
        rank_table=rank_table,
        comm_events=comm_recorder.events,
        timeline_events=comm_recorder.timeline_events,
        pp_schedule_obj=pp_schedule,
        pipeline_schedule=pipeline_schedule,
        num_micro_batches=num_micro_batches,
        gradient_accumulation=gradient_accumulation,
        rank=rank,
    )
    timings["build_schedule_plan"] = time.perf_counter() - t7b

    t8 = time.perf_counter()
    wg = build_workload_graph(
        schedule_graph=schedule_graph,
        step_templates=step_templates,
        local_batch_size=local_batch_size,
        seq_len=seq_len,
        num_micro_batches=num_micro_batches,
        schedule_plan=schedule_plan,
    )
    timings["build_workload_graph"] = time.perf_counter() - t8

    timings["total"] = time.perf_counter() - t0

    # Print timing table
    print("\n" + "=" * 60)
    print("Simulation Step Timing Breakdown")
    print("=" * 60)
    print(f"{'Stage':<30} {'Time (s)':>10} {'%':>8}")
    print("-" * 60)
    total = timings["total"]
    for name in ["setup", "forward_backward", "optimizer", "build_nodes",
                 "build_step_graphs", "build_rank_table", "build_schedule_graph",
                 "build_schedule_plan", "build_workload_graph"]:
        t = timings[name]
        print(f"{name:<30} {t:>10.2f} {t/total*100:>7.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total:>10.2f} {'100.0%':>8}")
    print("=" * 60)
    print(f"Captured ops: {len(nodes)}, comm events: {len(comm_recorder.events)}")
    print(f"Step templates: {list(step_templates.keys())}")
    print()

    return wg


class SimulationTrainer(Trainer):
    """Drop-in replacement for `Trainer` that captures the four-layer IR of
    one training step instead of training for `config.training.steps`
    steps. See design doc §6 for the end-to-end data flow.

    `Config = SimulationTrainerConfig` (an attribute assignment, not nested
    class syntax) is enough for `torchtitan.config.configurable.
    Configurable.__init_subclass__` to auto-wire `SimulationTrainerConfig.
    _owner = SimulationTrainer` -- verified directly against the pinned
    torchtitan commit during design: any name bound in a class body
    (including plain assignment) lands in that class's own `__dict__`,
    which is exactly what `__init_subclass__` checks for. This makes
    `some_simulation_trainer_config.build()` correctly return a
    `SimulationTrainer` instance (not a plain `Trainer`), the same pattern
    the sibling project's docs describe for their own `SimulationTrainer`.
    """

    Config = SimulationTrainerConfig

    def __init__(self, config: SimulationTrainerConfig) -> None:
        force_moe_load_balance(config)
        force_deterministic_seed(config)
        config.compile.enable = False  # tracing needs eager dispatch, not a compiled graph
        # comm.mode is set by entry.py / config_registry; do not override here
        # (fake_backend for single-process, multi_proc_meta for multi-process)
        apply_mhc_shims()
        apply_smla_shims()
        _strip_hardware_dependent_model_converters(config)
        if hasattr(config.optimizer, "swap_optimizer"):
            # swap_optimizer (NPU-specific host/device state swapping to
            # save real memory during real training) is irrelevant to
            # capture -- no real memory is ever allocated under meta
            # simulation, so there is nothing to save -- and its
            # host-state initialization unconditionally allocates pinned
            # host memory (`torch.zeros_like(..., pin_memory=True)`),
            # which triggers a real NPU hardware init that crashes with no
            # NPU device present. `hasattr` guards callers using a plain
            # (non-NPU) optimizer sub-config that has no such field.
            config.optimizer.swap_optimizer = False
        # Keep implementation="fused" — _patch_fused_adamw_for_meta in
        # meta_env.py patches torch._fused_adamw_ with a meta-safe shim
        # that records npu.npu_apply_adam_w and uses standard foreach
        # math for shape inference.

        patch_device_type_to_meta()

        # In multi_proc_meta mode, set TORCHTITAN_SIM_WORLD_SIZE so that
        # _patched_init_distributed returns the full simulated world_size
        # (not gloo's PP-only world_size) for ParallelDims validation.
        if config.comm.mode == "multi_proc_meta":
            import os
            sim_degrees = config.simulation.simulated_parallel_degrees
            if sim_degrees and "world_size" in sim_degrees:
                os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_degrees["world_size"])
            # Patch init_device_mesh to use fake backend for meshes larger
            # than gloo world_size (e.g. world_mesh of size 64 with 4 gloo procs)
            from torchtitan_npu.simulator.meta_env import _patch_parallel_dims_for_multi_proc
            import torch.distributed as dist
            gloo_ws = dist.get_world_size() if dist.is_initialized() else 1
            full_ws = int(sim_degrees.get("world_size", gloo_ws)) if sim_degrees else gloo_ws
            _patch_parallel_dims_for_multi_proc(full_ws, gloo_ws)

        super().__init__(config)
        self.simulation_config = config.simulation
        self.workload_graph: WorkloadGraph | None = None

    def train(self) -> None:
        import time
        t0 = time.perf_counter()

        data_iterator = iter(self.dataloader)
        input_dict, labels = next(data_iterator)
        for key, value in list(input_dict.items()):
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)
        labels = labels.to(self.device)

        t1 = time.perf_counter()

        # Determine rank for this process
        import torch.distributed as dist
        is_multi_proc = self.config.comm.mode == "multi_proc_meta"
        rank = dist.get_rank() if (is_multi_proc and dist.is_initialized()) else 0

        self.workload_graph = run_simulation_step(
            model_parts=self.model_parts,
            parallel_dims=self.parallel_dims,
            forward_backward_step=lambda **kwargs: self.forward_backward_step(**kwargs),
            input_dict=input_dict,
            labels=labels,
            optimizer_step=self.optimizers.step,
            lr_scheduler_step=self.lr_schedulers.step,
            local_batch_size=self.config.training.local_batch_size,
            seq_len=self.config.training.seq_len,
            pipeline_schedule=self.config.parallelism.pipeline_parallel_schedule,
            num_micro_batches=self.gradient_accumulation_steps,
            gradient_accumulation=self.gradient_accumulation_steps,
            rank=rank,
            pp_schedule=getattr(self, "pp_schedule", None),
        )

        t2 = time.perf_counter()

        # Host-only GE-catalog fusion pass: populate StepGraph.fused_regions
        # so _export writes the fused-region view alongside the original IR.
        if self.simulation_config.enable_fusion:
            from torchtitan_npu.simulator.ir.ge_fusion import (
                build_ge_fusion_profile, apply_ge_fusion_profile,
            )
            for sg in self.workload_graph.step_templates.values():
                apply_ge_fusion_profile(sg, build_ge_fusion_profile(sg))

        # Each rank exports independently to rank_N/ directory (no merge)
        if is_multi_proc and dist.is_initialized():
            self._export(rank=rank)
            dist.barrier()
        else:
            self._export()

        t3 = time.perf_counter()

        print(f"\n{'Stage':<30} {'Time (s)':>10}")
        print("-" * 42)
        print(f"{'dataloader':<30} {t1-t0:>10.2f}")
        print(f"{'run_simulation_step':<30} {t2-t1:>10.2f}")
        print(f"{'export':<30} {t3-t2:>10.2f}")
        print(f"{'TOTAL train()':<30} {t3-t0:>10.2f}")
        print()

    def _export_per_rank(self, rank: int) -> None:
        """Write this rank's IR to a per-rank JSON file."""
        assert self.workload_graph is not None
        out_dir = self.simulation_config.output_dir
        per_rank_dir = os.path.join(out_dir, "per_rank")
        os.makedirs(per_rank_dir, exist_ok=True)
        # Serialize this rank's captured data
        try:
            import orjson as _json
            _dumps = lambda d: _json.dumps(d)
            _mode = "wb"
        except ImportError:
            import json as _json
            _dumps = lambda d: _json.dumps(d)
            _mode = "w"
        schedule = self.workload_graph.iteration.schedule
        rank_data = {
            "rank": rank,
            "pipeline_stage": rank,  # In multi_proc_meta, rank == PP stage
            "step_templates": {},
            "execution_timeline": [
                {
                    "seq_idx": e.seq_idx,
                    "op_id": e.op_id,
                    "rank": rank,
                    "pipeline_stage": rank,
                    "micro_batch_idx": e.micro_batch_idx,
                    "phase": e.phase,
                    "comp_type": e.comp_type,
                    "comm_type": e.comm_type,
                    "comm_peer_rank": e.comm_peer_rank,
                }
                for e in schedule.execution_timeline
            ],
        }
        # Serialize step templates (L0 ops)
        for tid, sg in self.workload_graph.step_templates.items():
            rank_data["step_templates"][tid] = {
                "step_type": sg.step_type,
                "nodes": [
                    {
                        "op_id": n.op_id,
                        "seq_idx": n.seq_idx,
                        "op_type": n.op_type,
                        "raw_op_type": n.annotations.get("raw_op_type", ""),
                        "phase": n.annotations.get("phase", ""),
                        "comp_type": n.annotations.get("comp_type", ""),
                        "fsdp_state": n.annotations.get("fsdp_state", "NA"),
                        "pp_stage": n.annotations.get("pp_stage", -1),
                        "inputs_shape": [list(m.shape) for m in n.inputs],
                        "outputs_shape": [list(m.shape) for m in n.outputs],
                        "inputs_dtype": [str(m.dtype) for m in n.inputs],
                        "outputs_dtype": [str(m.dtype) for m in n.outputs],
                        "flops": n.flops,
                        "peak_mem": n.peak_mem,
                        "comm_bytes": n.comm_bytes,
                        "module_path": n.annotations.get("module_path", ""),
                        "comm_dim": n.annotations.get("comm_dim", ""),
                        "comm_ranks": n.annotations.get("comm_ranks", ""),
                    }
                    for n in sg.nodes.values()
                ],
            }
        with open(os.path.join(per_rank_dir, f"rank_{rank}.json"), _mode) as f:
            f.write(_dumps(rank_data))

    def _merge_per_rank_ir(self) -> None:
        """Merge all per-rank IR files into the final output."""
        out_dir = self.simulation_config.output_dir
        per_rank_dir = os.path.join(out_dir, "per_rank")
        import glob

        try:
            import orjson as _json
            _loads = _json.loads
            _read_mode = "rb"
        except ImportError:
            import json as _json
            _loads = _json.load
            _read_mode = "r"

        rank_files = sorted(glob.glob(os.path.join(per_rank_dir, "rank_*.json")),
                            key=lambda p: int(p.split("rank_")[-1].split(".")[0]))
        all_ranks = []
        for f in rank_files:
            with open(f, _read_mode) as fh:
                if _read_mode == "rb":
                    all_ranks.append(_loads(fh.read()))
                else:
                    all_ranks.append(_loads(fh))

        # Write merged summary
        with open(os.path.join(out_dir, "merged_summary.txt"), "w") as f:
            f.write(f"Merged IR from {len(all_ranks)} PP stages\n\n")
            for r in all_ranks:
                stage = r["pipeline_stage"]
                templates = r["step_templates"]
                timeline = r["execution_timeline"]
                from collections import Counter
                phases = Counter(e["phase"] for e in timeline)
                f.write(f"[Stage {stage}] rank={r['rank']} phases={dict(phases)}\n")
                for tid, sg in templates.items():
                    f.write(f"  {tid}: {sg['step_type']}, {len(sg['nodes'])} nodes\n")
                p2p = [e for e in timeline if e["comm_type"] and ("send" in e["comm_type"] or "recv" in e["comm_type"])]
                f.write(f"  P2P events: {len(p2p)}\n")
                for e in p2p[:5]:
                    f.write(f"    seq={e['seq_idx']} mb={e['micro_batch_idx']} comm={e['comm_type']} peer={e['comm_peer_rank']}\n")
                if len(p2p) > 5:
                    f.write(f"    ... and {len(p2p)-5} more\n")
                f.write("\n")

        # Also export the rank 0's workload graph as the base, plus
        # per-stage L0 CSVs
        formats = self.simulation_config.output_formats
        if "csv" in formats:
            ir_dir = os.path.join(out_dir, "ir_export")
            os.makedirs(ir_dir, exist_ok=True)
            for r in all_ranks:
                stage = r["pipeline_stage"]
                for tid, sg in r["step_templates"].items():
                    fname = os.path.join(ir_dir, f"stage_{stage}_{sg['step_type']}_l0_ops.csv")
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write("topo_order,op_id,seq_idx,op_type,raw_op_type,phase,comp_type,fsdp_state,pp_stage,inputs_shape,outputs_shape,flops,peak_mem,comm_bytes,module_path,comm_dim,comm_ranks\n")
                        for i, n in enumerate(sg["nodes"]):
                            f.write(f"{i},{n['op_id']},{n['seq_idx']},{n['op_type']},{n['raw_op_type']},{n['phase']},{n.get('comp_type','')},{n.get('fsdp_state','NA')},{n.get('pp_stage',-1)},{';'.join(str(s) for s in n['inputs_shape'])},{';'.join(str(s) for s in n['outputs_shape'])},{n['flops']},{n['peak_mem']},{n['comm_bytes']},{n['module_path']},{n['comm_dim']},{n['comm_ranks']}\n")

        print(f"Merged {len(all_ranks)} PP stages' IR to {out_dir}")

    def _export(self, rank: int = 0) -> None:
        import time
        t0 = time.perf_counter()
        assert self.workload_graph is not None
        # In multi_proc mode, write to rank_N/ subdirectory; else top-level
        base_dir = self.simulation_config.output_dir
        if rank > 0 or self.config.comm.mode == "multi_proc_meta":
            out_dir = os.path.join(base_dir, f"rank_{rank}")
        else:
            out_dir = base_dir
        os.makedirs(out_dir, exist_ok=True)
        formats = self.simulation_config.output_formats
        export_timings: dict[str, float] = {}

        if "json" in formats:
            t = time.perf_counter()
            export_json(self.workload_graph, os.path.join(out_dir, "simulation_result.json"))
            export_timings["json"] = time.perf_counter() - t
        if "dot" in formats:
            t = time.perf_counter()
            export_dot(self.workload_graph, os.path.join(out_dir, "compute_graph.dot"))
            export_timings["dot"] = time.perf_counter() - t
        if "text" in formats:
            t = time.perf_counter()
            write_text_summary(self.workload_graph, os.path.join(out_dir, "summary.txt"))
            export_timings["text"] = time.perf_counter() - t
        if "html" in formats:
            t = time.perf_counter()
            export_html(self.workload_graph, os.path.join(out_dir, "trace.html"))
            export_timings["html"] = time.perf_counter() - t
        if "csv" in formats:
            t = time.perf_counter()
            export_kernel_summary_csv(
                self.workload_graph,
                os.path.join(out_dir, "kernel_summary"),
                max_ranks=self.simulation_config.csv_max_ranks,
            )
            export_timings["csv_kernel_summary"] = time.perf_counter() - t

            t = time.perf_counter()
            # Per-level IR export: scheduling relationships
            ir_dir = os.path.join(out_dir, "ir_export")
            os.makedirs(ir_dir, exist_ok=True)
            # L3: inter-rank schedule
            self.workload_graph.export_schedule_csv(os.path.join(ir_dir, "rank_schedule.csv"))
            # L2: structured schedule plan (ordered ScheduleActions + DataSlots)
            if self.workload_graph.schedule_plan is not None:
                self.workload_graph.schedule_plan.export_schedule_plan_csv(
                    os.path.join(ir_dir, "schedule_plan.csv")
                )
            # L2: per-stage L1 schedule
            self.workload_graph.iteration.schedule.export_l1_schedule_csv(
                os.path.join(ir_dir, "l1_schedule"),
                max_ranks=self.simulation_config.csv_max_ranks,
            )
            # L1: per-step L0 ops (export_l0_csv appends region_id/fused_op_type
            # columns automatically when fused_regions is populated)
            for tid, sg in self.workload_graph.step_templates.items():
                sg.export_l0_csv(os.path.join(ir_dir, f"step_{sg.step_type}_l0_ops.csv"))
            # L1 fusion: per-step fused regions (only when fusion was applied)
            if any(getattr(sg, "fused_regions", None) for sg in self.workload_graph.step_templates.values()):
                fused_dir = os.path.join(ir_dir, "fused_regions")
                os.makedirs(fused_dir, exist_ok=True)
                for tid, sg in self.workload_graph.step_templates.items():
                    if getattr(sg, "fused_regions", None):
                        sg.export_fused_regions_csv(
                            os.path.join(fused_dir, f"step_{sg.step_type}_fused_regions.csv")
                        )
            export_timings["csv_ir_export"] = time.perf_counter() - t

        total_export = time.perf_counter() - t0
        print(f"\n{'Export Stage':<30} {'Time (s)':>10}")
        print("-" * 42)
        for name, t in export_timings.items():
            print(f"{name:<30} {t:>10.2f}")
        print(f"{'TOTAL export':<30} {total_export:>10.2f}")
        print()
