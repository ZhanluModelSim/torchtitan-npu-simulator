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
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph
from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta
from torchtitan_npu.simulator.moe_force_balance import force_deterministic_seed, force_moe_load_balance
from torchtitan_npu.simulator.rank_table import build_rank_table
from torchtitan_npu.simulator.viz.dot_export import export_dot
from torchtitan_npu.simulator.viz.html_export import export_html
from torchtitan_npu.simulator.viz.json_export import export_json
from torchtitan_npu.simulator.viz.text_summary import write_text_summary


@dataclass(kw_only=True, slots=True)
class SimulationConfig:
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(default_factory=lambda: ["json", "dot", "text", "html"])


@dataclass(kw_only=True, slots=True)
class SimulationTrainerConfig(Trainer.Config):
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# Model converters that unconditionally require real accelerator hardware to
# run their forward computation (a Triton-JIT kernel launch or a JIT-compiled
# aclnn extension, either gated behind a private "custom_ops" module
# unavailable outside Huawei's internal build), with no meta-device-compatible
# fallback other than the model's own base (pre-conversion) module -- see
# `_strip_hardware_dependent_model_converters`.
#   - npu_mhc_pre/npu_mhc_post: MHCPreConverter/MHCPostConverter select
#     between a Triton-JIT kernel and a fused NPU custom op gated behind
#     "custom_ops"; the base HcPre/HcPost/HcSplitSinkhorn classes are
#     pure PyTorch and meta-safe as-is.
#   - npu_smla: NpuSMLAConverter's non-"A5" path replaces THREE independent
#     submodule types (SparseAttention, LiCompute, LiLoss), each building
#     its own JIT-compiled aclnn extension via build_op(...)
#     (`sparse_attn_sharedkv`, `lightning_indexer`,
#     `sparse_lightning_indexer_grad_kl_loss`). ALL THREE crash on meta
#     tensors with no NPU device present (SIGABRT/"No NPUs are available"
#     for SparseAttention; "Invalid device ID"/LazySetDevice for
#     LiCompute) -- real hardware checks unreachable from Python-level
#     monkeypatching. Stripping the whole converter falls back to the
#     base (pre-conversion) SparseAttention/LiCompute classes, which are
#     pure PyTorch and meta-safe (verified by reading their forward()
#     methods; SparseAttention's one hardcoded `device="npu"` literal is
#     redirected by `meta_env._patch_torch_full_npu_device_literal`).
#     The base LiLoss class has its OWN real, pre-existing shape bug in
#     `_current_selected_attn_dist`'s einsum (never exercised in real
#     production, since production always uses the NPU-converted LiLoss);
#     see `meta_env._patch_li_loss_to_skip_buggy_einsum` for that fix.
_HARDWARE_DEPENDENT_CONVERTER_NAMES = frozenset({"npu_mhc_pre", "npu_mhc_post", "npu_smla"})


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
    capture = OpDispatchCapture(module_path_tracker=module_path_tracker, phase_provider=lambda: boundary.current_phase)

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
        config.comm.mode = "fake_backend"
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
        self._export()

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
