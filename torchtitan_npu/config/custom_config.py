# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/config/job_config.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.job_config import (
    Checkpoint as BaseCheckpoint,
    JobConfig as BaseJobConfig,
    Optimizer as BaseOptimizer,
    Parallelism as BaseParallelism,
    Profiling as BaseProfiling,
    Training as BaseTraining,
)


@dataclass
class Optimizer(BaseOptimizer):
    """
    Whether to apply swap optimizer.
    This feature will offload the optimizer states to the host (CPU) during the forward and backward passes.
    During the optimizer.step(), it will load, update, and offload these states in slices.
    This pipelined approach significantly reduces GPU memory pressure during the optimizer step,
    making it highly beneficial for memory-intensive scenarios.
    More info (in Chinese): https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md
    """

    swap_optimizer: bool = False

    """
    Specifies the number of slices for the pipelined swap_optimizer update.
    A higher value creates more, smaller slices, further reducing peak memory usage during the optimizer step.
    """
    swap_optimizer_times: int = 16

    """
    Whether to apply virtual optimizer.
    This feature will offload the optimizer states to the host (CPU) during the forward and backward passes.
    More info (in Chinese): https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/virtual-optimizer.md
    """
    virtual_optimizer: bool = False

    """
    virtual_optimizer_size configures the swap memory size for optimizer momentum in each pipeline
    parallelism (PP) stage.
    It accepts all, a single numeric value, or a list of values to enable full momentum swapping,
    uniform swap allocation across all stages, and stage-specific differentiated swap allocation respectively.
    """
    virtual_optimizer_size: float | list[float] | str | None = None

    # Muon-specific parameters (used when name is "Muon")
    """
    Learning rate for Muon optimizer. If None, falls back to lr.
    """
    muon_lr: float | None = None

    """Momentum factor for Muon optimizer"""
    muon_momentum: float = 0.95

    """Whether to use Nesterov momentum for Muon"""
    muon_enable_nesterov: bool = True

    """Number of Newton-Schulz iteration steps for Muon"""
    muon_ns_steps: int = 5

    """Whether to use hybrid Newton-Schulz (8 primary + 2 secondary steps)"""
    muon_hybrid_ns: bool = False

    """
    Learning rate adjustment function for Muon. Options:
    - None or "original": Use sqrt(max(1, A/B)) ratio (muon_lr is used if specified)
    - "match_rms_adamw": Use 0.18 * sqrt(max(A, B)) ratio (muon_lr is ignored, uses base lr)
    """
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = "match_rms_adamw"

    """Extra parameter group split rules for distributed Muon (list of dicts with str_match + overrides)"""
    extra_param_group_split_rules: list[dict] | None = None

    """Whether to enable virtual optimizer (swap optimizer states to CPU for Muon)"""
    virtual_allocator: bool = False

    """(Muon only) When using swap optimizer with Muon, configures the number of communication buckets
    to merge for swap H2D/D2H batching. Higher values reduce stream synchronization
    overhead but increase peak NPU memory. Default 1 = no merging (original behavior).
    2 = merge 2 buckets per H2D/D2H batch.
    """
    swap_merge_buckets: int = 1


@dataclass
class Parallelism(BaseParallelism):
    """
    Whether to use a custom context parallel implementation.
    When True and context_parallel_degree > 1, Ulysses-style CP is applied to attention modules.
    """

    enable_custom_context_parallel: bool = False

    """
    Load balancer type for context parallelism, defaults to None.
    Override this due to inconvenience of specifying None in TOML.
    """

    context_parallel_load_balancer: str | None = None


@dataclass
class Training(BaseTraining):
    """
    Specifies the maximum proportion of NPU memory that PyTorch is allowed to occupy.
    The value ranges from 0.0 to 1.0, where 0.9 means PyTorch can use up to 90% of the total NPU memory.
    Adjusting this value helps control memory usage and avoid out-of-memory (OOM) errors on NPU devices.
    """

    torch_npu_memory_ratio: float = 1.0

    """Number of tokens to predict at once using multi-token prediction"""
    num_mtp_modules: int = 0

    """Weight of multi-token prediction loss term"""
    mtp_loss_weight: float = 0.3

    dataset_type: Literal["pretrain", "sft"] = "pretrain"


@dataclass
class Profiling(BaseProfiling):
    """
    The step at which to start profiling.
    Profiling will begin at this step and continue for `profiler_active` steps.
    """

    profile_step_start: int = 0

    """
    The step at which to end profiling.
    If set to 0, will use profile_step_start + profiler_active.
    """
    profile_step_end: int = 0

    """
    List of ranks to profile, e.g., [0, 1, 2].
    Use [-1] to profile all ranks.
    Default is [-1] (all ranks).
    """
    profile_ranks: list[int] = field(default_factory=lambda: [-1])

    """
    Whether to record tensor shapes during profiling.
    """
    profile_record_shapes: bool = True

    """
    Whether to profile memory usage.
    """
    profile_with_memory: bool = False

    """
    Whether to record stack traces during profiling.
    """
    profile_with_stack: bool = False

    """
    Whether to enable online parsing of profiling data.
    If disabled, on_trace_ready will be set to None and ASCEND_WORK_PATH environment
    variable will be set to trace_dir for offline parsing.
    """
    enable_online_parse: bool = True


@dataclass
class Checkpoint(BaseCheckpoint):
    """
    Whether FileSystemWriter fsyncs checkpoint files before returning.
    Disabling this can reduce checkpoint latency but weakens crash consistency.
    """

    sync_files: bool = True

    """
    Whether to ask Linux to drop checkpoint file pages from host page cache after writing.
    This reduces host memory pressure for large checkpoints without changing checkpoint files.
    """
    drop_page_cache_after_save: bool = False

    """
    Whether to clear the NPU caching allocator after a checkpoint save.
    This helps release temporary checkpoint buffers before training resumes.
    """
    empty_cache_after_save: bool = True


@dataclass
class JobConfig(BaseJobConfig):
    # pyrefly: ignore [bad-override]
    optimizer: Optimizer = field(default_factory=Optimizer)
    # pyrefly: ignore [bad-override]
    parallelism: Parallelism = field(default_factory=Parallelism)
    # pyrefly: ignore [bad-override]
    training: Training = field(default_factory=Training)
    # pyrefly: ignore [bad-override]
    profiling: Profiling = field(default_factory=Profiling)
    # pyrefly: ignore [bad-override]
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
