# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NPU-specific extensions of upstream config dataclasses.

Mirrors `torchtitan/config/configs.py` semantically: each class subclasses
its upstream counterpart and adds NPU-only fields. The top-level container
`TrainerConfig` subclasses `torchtitan.trainer.Trainer.Config` and wires the
NPU sub-configs in so that tyro exposes the extra fields as CLI flags.

Usage in npu config_registry functions:

    from torchtitan_npu.config.configs import (
        OptimizerConfig,
        ParallelismConfig,
        TrainingConfig,
        ProfilingConfig,
        TrainerConfig,
    )

    def my_recipe() -> TrainerConfig:
        return TrainerConfig(
            training=TrainingConfig(num_mtp_modules=1, ...),
            optimizer=OptimizerConfig(swap_optimizer=True, ...),
            ...
        )
"""

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ParallelismConfig as _BaseParallelismConfig,
)
from torchtitan.config import (
    TrainingConfig as _BaseTrainingConfig,
)
from torchtitan.tools.profiling import ProfilingConfig as _BaseProfilingConfig
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(OptimizersContainer.Config):
    """Optimizer config with NPU-specific (swap / virtual / Muon) fields."""

    swap_optimizer: bool = False
    """
    Whether to apply swap optimizer.
    Offloads optimizer states to host (CPU) during forward and backward;
    loads / updates / offloads in slices during optimizer.step(). Pipelined
    approach significantly reduces NPU memory pressure during the optimizer
    step. More info (zh):
    https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md
    """

    swap_optimizer_times: int = 16
    """
    Number of slices for the pipelined swap_optimizer update. A higher value
    creates more, smaller slices, further reducing peak memory usage.
    """

    virtual_optimizer: bool = False
    """
    Whether to apply virtual optimizer. Offloads optimizer states to host
    (CPU) during forward and backward. More info (zh):
    https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/virtual-optimizer.md
    """

    virtual_optimizer_size: float | list[float] | str | None = None
    """
    Configures swap memory size for optimizer momentum in each pipeline
    parallel (PP) stage. Accepts: 'all', a single numeric, or a list of
    values for full / uniform / per-stage allocation respectively.
    """

    muon_lr: float | None = None
    """Learning rate for Muon optimizer. If None, falls back to lr."""

    muon_momentum: float = 0.95
    """Momentum factor for Muon optimizer."""

    muon_enable_nesterov: bool = True
    """Whether to use Nesterov momentum for Muon."""

    muon_ns_steps: int = 5
    """Number of Newton-Schulz iteration steps for Muon."""

    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = "match_rms_adamw"
    """
    Learning rate adjustment function for Muon. Options:
      - None or 'original': sqrt(max(1, A/B)) ratio (muon_lr is used if specified)
      - 'match_rms_adamw': 0.18 * sqrt(max(A, B)) ratio (muon_lr ignored, uses base lr)
    """

    muon_hybrid_ns: bool = False
    """
    Whether to use hybrid Newton-Schulz iteration for Muon.
    When True, uses 8 primary + 2 secondary NS steps instead of a single ns_steps count.
    """

    extra_param_group_split_rules: list[dict] | None = None
    """
    Extra parameter group split rules for Muon optimizer.
    Each rule is a dict with 'str_match' (regex) and optional overrides
    (lr, backend_steps, etc.) for matching parameter groups.
    """

    swap_merge_buckets: int = 1
    """
    (Muon only) Number of communication buckets to merge during swap optimizer step.
    Higher values reduce the number of H2D/D2H roundtrips at the cost of
    higher peak memory during the merge group.
    """


@dataclass(kw_only=True, slots=True)
class ParallelismConfig(_BaseParallelismConfig):
    """Parallelism config with NPU-specific CP load balancer override."""

    context_parallel_load_balancer: str | None = None
    """
    NPU override of the upstream default (``"headtail"``).

    The NPU Ulysses CP (DeepSeek-V3/V32/V4) rebuilds the full sequence via an
    all-to-all and applies a standard contiguous causal mask, which is
    incompatible with the head/tail (or ptrr) sequence reordering: the
    reordered sequence breaks causality and degrades accuracy. ``None``
    disables load balancing (contiguous CP sharding), which is also optimal
    for Ulysses since every rank already does equal full-sequence attention.
    To opt back into a reordering balancer, set it explicitly.
    """


@dataclass(kw_only=True, slots=True)
class TrainingConfig(_BaseTrainingConfig):
    """Training config with NPU memory and Multi-Token-Prediction fields."""

    torch_npu_memory_ratio: float = 1.0
    """
    Maximum proportion of NPU memory PyTorch is allowed to occupy in [0.0, 1.0].
    Helps avoid out-of-memory (OOM) errors on NPU devices.
    """

    num_mtp_modules: int = 0
    """
    Number of tokens to predict at once via Multi-Token-Prediction
    (deepseek_v32 / deepseek_v4).
    """

    mtp_loss_weight: float = 0.3
    """Weight of the Multi-Token-Prediction loss term."""


@dataclass(kw_only=True, slots=True)
class ProfilingConfig(_BaseProfilingConfig):
    """Profiling config with NPU-specific step-range and Ascend trace options."""

    profile_step_start: int = 0
    """Step at which to start profiling (continues for `profiler_active` steps)."""

    profile_step_end: int = 0
    """
    Step at which to end profiling.
    If 0, uses `profile_step_start + profiler_active`.
    """

    profile_ranks: list[int] = field(default_factory=lambda: [-1])
    """List of ranks to profile (e.g. [0, 1, 2]). Use [-1] to profile all ranks."""

    profile_record_shapes: bool = True
    """Whether to record tensor shapes during profiling."""

    profile_with_memory: bool = False
    """Whether to profile memory usage."""

    profile_with_stack: bool = False
    """Whether to record stack traces during profiling."""

    enable_online_parse: bool = True
    """
    Whether to enable online parsing of profiling data.
    If False, on_trace_ready is set to None and ASCEND_WORK_PATH is set to
    trace_dir for offline parsing.
    """


@dataclass(kw_only=True, slots=True)
class CheckpointConfig(CheckpointManager.Config):
    """Checkpoint config with NPU-specific writer and cache controls."""

    sync_files: bool = True
    """
    Whether FileSystemWriter fsyncs checkpoint files before returning.
    Disabling this can reduce checkpoint latency but weakens crash consistency.
    """

    drop_page_cache_after_save: bool = False
    """
    Whether to ask Linux to drop checkpoint file pages from host page cache after writing.
    This reduces host memory pressure for large checkpoints without changing checkpoint files.
    """

    empty_cache_after_save: bool = True
    """
    Whether to clear the NPU caching allocator after checkpoint save returns.
    This helps release temporary checkpoint buffers before training resumes.
    """


@dataclass(kw_only=True, slots=True)
class TrainerConfig(Trainer.Config):
    """Top-level NPU training config.

    Subclass of `Trainer.Config` with the NPU sub-configs wired in via
    overridden field types. Use this in npu config_registry functions so
    tyro picks up the NPU CLI flags.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # type: ignore[assignment]
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)  # type: ignore[assignment]
    training: TrainingConfig = field(default_factory=TrainingConfig)  # type: ignore[assignment]
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)  # type: ignore[assignment]
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)  # type: ignore[assignment]
