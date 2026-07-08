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
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_graph
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
    output_formats: list[str] = field(default_factory=lambda: ["json", "dot", "text", "html", "csv"])
    target_npu_device_type: str = "non_a5"
    csv_max_ranks: int | None = None
    simulated_parallel_degrees: dict[str, int] = field(default_factory=dict)


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
    patch_device_type_to_meta()
    global_valid_tokens = float(labels.numel())

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

    with capture_fake_collectives() as comm_recorder, boundary, module_path_tracker, capture:
        boundary.mark("forward")
        forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )
        boundary.mark("optimizer")
        optimizer_step()
        lr_scheduler_step()

    nodes = capture.build_nodes()
    step_templates = build_step_graphs(nodes)
    rank_table = build_rank_table(parallel_dims)
    schedule_graph = build_schedule_graph(
        step_templates=step_templates,
        rank_table=rank_table,
        comm_events=comm_recorder.events,
        pipeline_schedule=pipeline_schedule,
        num_micro_batches=num_micro_batches,
        gradient_accumulation=gradient_accumulation,
    )
    return build_workload_graph(
        schedule_graph=schedule_graph,
        step_templates=step_templates,
        local_batch_size=local_batch_size,
        seq_len=seq_len,
        num_micro_batches=num_micro_batches,
    )


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
        if getattr(config.optimizer, "implementation", None) == "fused":
            # torch.optim's fused implementation validates the parameter
            # device against a hardcoded supported-device list (mps/cuda/
            # xpu/hpu/cpu/mtia/npu) that does not include "meta", raising
            # `RuntimeError: fused=True requires all the params to be
            # floating point Tensors of supported devices` -- found via
            # the 16-layer DeepSeek-V4-Pro smoke run's optimizer.step().
            # "foreach" is the standard non-fused vectorized fallback,
            # device-agnostic and meta-safe.
            config.optimizer.implementation = "foreach"

        patch_device_type_to_meta()
        super().__init__(config)
        self.simulation_config = config.simulation
        self.workload_graph: WorkloadGraph | None = None

    def train(self) -> None:
        data_iterator = iter(self.dataloader)
        input_dict, labels = next(data_iterator)
        for key, value in list(input_dict.items()):
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)
        labels = labels.to(self.device)

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
        )
        # In multi_proc_meta mode, each rank writes its IR to a per-rank
        # file; rank 0 merges them after all ranks finish.
        import torch.distributed as dist
        is_multi_proc = self.config.comm.mode == "multi_proc_meta"
        if is_multi_proc and dist.is_initialized():
            rank = dist.get_rank()
            # Each rank writes its per-rank IR
            self._export_per_rank(rank)
            # Barrier to ensure all ranks have written
            dist.barrier()
            # Rank 0 merges all per-rank IRs
            if rank == 0:
                self._merge_per_rank_ir()
        else:
            self._export()

    def _export_per_rank(self, rank: int) -> None:
        """Write this rank's IR to a per-rank JSON file."""
        assert self.workload_graph is not None
        out_dir = self.simulation_config.output_dir
        per_rank_dir = os.path.join(out_dir, "per_rank")
        os.makedirs(per_rank_dir, exist_ok=True)
        # Serialize this rank's captured data
        import json
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
                        "inputs_shape": [list(m.shape) for m in n.inputs],
                        "outputs_shape": [list(m.shape) for m in n.outputs],
                        "inputs_dtype": [m.dtype for m in n.inputs],
                        "outputs_dtype": [m.dtype for m in n.outputs],
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
        with open(os.path.join(per_rank_dir, f"rank_{rank}.json"), "w") as f:
            json.dump(rank_data, f)

    def _merge_per_rank_ir(self) -> None:
        """Merge all per-rank IR files into the final output."""
        out_dir = self.simulation_config.output_dir
        per_rank_dir = os.path.join(out_dir, "per_rank")
        import json
        import glob

        rank_files = sorted(glob.glob(os.path.join(per_rank_dir, "rank_*.json")),
                            key=lambda p: int(p.split("rank_")[-1].split(".")[0]))
        all_ranks = []
        for f in rank_files:
            with open(f) as fh:
                all_ranks.append(json.load(fh))

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
                        f.write("topo_order,op_id,seq_idx,op_type,raw_op_type,phase,inputs_shape,outputs_shape,flops,peak_mem,comm_bytes,module_path,comm_dim,comm_ranks\n")
                        for i, n in enumerate(sg["nodes"]):
                            f.write(f"{i},{n['op_id']},{n['seq_idx']},{n['op_type']},{n['raw_op_type']},{n['phase']},{';'.join(str(s) for s in n['inputs_shape'])},{';'.join(str(s) for s in n['outputs_shape'])},{n['flops']},{n['peak_mem']},{n['comm_bytes']},{n['module_path']},{n['comm_dim']},{n['comm_ranks']}\n")

        print(f"Merged {len(all_ranks)} PP stages' IR to {out_dir}")

    def _export(self) -> None:
        assert self.workload_graph is not None
        out_dir = self.simulation_config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        formats = self.simulation_config.output_formats
        if "json" in formats:
            export_json(self.workload_graph, os.path.join(out_dir, "simulation_result.json"))
        if "dot" in formats:
            export_dot(self.workload_graph, os.path.join(out_dir, "compute_graph.dot"))
        if "text" in formats:
            write_text_summary(self.workload_graph, os.path.join(out_dir, "summary.txt"))
        if "html" in formats:
            export_html(self.workload_graph, os.path.join(out_dir, "trace.html"))
        if "csv" in formats:
            export_kernel_summary_csv(
                self.workload_graph,
                os.path.join(out_dir, "kernel_summary"),
                max_ranks=self.simulation_config.csv_max_ranks,
            )
            # Per-level IR export: scheduling relationships
            ir_dir = os.path.join(out_dir, "ir_export")
            os.makedirs(ir_dir, exist_ok=True)
            # L3: inter-rank schedule
            self.workload_graph.export_schedule_csv(os.path.join(ir_dir, "rank_schedule.csv"))
            # L2: per-stage L1 schedule
            self.workload_graph.iteration.schedule.export_l1_schedule_csv(
                os.path.join(ir_dir, "l1_schedule"),
                max_ranks=self.simulation_config.csv_max_ranks,
            )
            # L1: per-step L0 ops
            for tid, sg in self.workload_graph.step_templates.items():
                sg.export_l0_csv(os.path.join(ir_dir, f"step_{sg.step_type}_l0_ops.csv"))
