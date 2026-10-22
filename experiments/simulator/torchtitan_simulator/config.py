# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compose simulation settings around a native training configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchtitan_simulator.model_overrides import ModelOverrides
    from torchtitan_simulator.selective_ac import SelectiveACSaveOp


@dataclass(kw_only=True)
class SimulationConfig:
    config: str = "deepseek_v4_smoketest"
    world_size: int = 1
    logical_rank: int = 0
    pipeline_parallel_degree: int = 1
    seq_len: int = 128
    num_mtp_modules: int = 0
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(default_factory=list)
    csv_max_ranks: int = 1
    enable_fsdp_allgather_fp8: bool = False
    enable_ep_dispatch_fp8: bool = False
    target_npu_device_type: str = "non_a5"
    override_imports: list[str] | None = None
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = -1
    tensor_parallel_degree: int = 1
    context_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    activation_checkpoint_mode: str = "none"
    selective_ac_save_ops: list[SelectiveACSaveOp] | None = None
    memory_parameter_storage_dtype: str = ""
    memory_offload_ac_saved_tensors: bool = False
    local_batch_size: int | None = None
    global_batch_size: int | None = None
    enable_quantized_training: bool = False
    quantization_recipe: str = "all_mxfp8"
    model_overrides: ModelOverrides | None = None

    def validate(self):
        if self.csv_max_ranks < 1:
            raise ValueError("csv_max_ranks must be positive; it limits export only, not the logical world")
        if self.model_overrides is not None:
            self.model_overrides.validate()
        if self.quantization_recipe != "all_mxfp8":
            raise ValueError("This simulator stage supports only the native all_mxfp8 quantization recipe")
        if self.activation_checkpoint_mode not in {"none", "full", "selective"}:
            raise ValueError("activation_checkpoint_mode must be none, full, or selective")
        if self.selective_ac_save_ops is not None:
            from torchtitan_simulator.selective_ac import _validate_selection

            if self.activation_checkpoint_mode != "selective":
                raise ValueError("selective_ac_save_ops requires selective activation checkpointing")
            _validate_selection(self.selective_ac_save_ops)
        if any(value is not None and value < 1 for value in (self.local_batch_size, self.global_batch_size)):
            raise ValueError("Explicit batch sizes must be positive")
        if self.memory_parameter_storage_dtype:
            from torchtitan_simulator.capture.tensor_utils import normalize_supported_dtype

            normalize_supported_dtype(self.memory_parameter_storage_dtype)
        if not 0 <= self.logical_rank < self.world_size:
            raise ValueError("logical_rank must be in [0, world_size)")
        if self.pipeline_parallel_degree != 1:
            raise ValueError("PP is deferred; pipeline_parallel_degree must be 1")
        if self.world_size < 1 or self.seq_len < 1 or self.num_mtp_modules < 0:
            raise ValueError("world_size/seq_len must be positive; num_mtp_modules must be nonnegative")
        if self.enable_fsdp_allgather_fp8 or self.enable_ep_dispatch_fp8:
            raise ValueError("FP8 communication modeling is deferred; disable FP8 all-gather and EP dispatch")
        if self.target_npu_device_type not in {"non_a5", "a5"}:
            raise ValueError("target_npu_device_type must be 'non_a5' or 'a5'")
        unknown = set(self.output_formats) - {"text", "html", "dot", "json", "csv", "mem"}
        if unknown:
            raise ValueError(f"Unsupported output formats: {sorted(unknown)}")
        degrees = (
            self.data_parallel_replicate_degree,
            self.tensor_parallel_degree,
            self.context_parallel_degree,
            self.expert_parallel_degree,
        )
        if any(degree < 1 for degree in degrees):
            raise ValueError("Parallel degrees must be positive")
        fixed = self.data_parallel_replicate_degree * self.tensor_parallel_degree * self.context_parallel_degree
        shard = self.data_parallel_shard_degree
        if shard == -1:
            if self.world_size % fixed:
                raise ValueError("world_size must be divisible by DP replicate * TP * CP")
            shard = self.world_size // fixed
        if shard < 1 or fixed * shard != self.world_size:
            raise ValueError("world_size must equal DP replicate * DP shard * TP * CP")
        if (shard * self.tensor_parallel_degree * self.context_parallel_degree) % self.expert_parallel_degree:
            raise ValueError("EP must divide DP shard * TP * CP")


@dataclass
class PreparedConfig:
    simulation: SimulationConfig
    trainer: Any
    config_factory: str
    adaptations: list[str]


_DSV4_OVERRIDES = [
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.token_dispatcher.asc",
    "torchtitan_npu.override.common.rope.asc_complex",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
]


def prepare_config(simulation: SimulationConfig) -> PreparedConfig:
    """Resolve a fresh native config before any model or parallel initialization."""
    simulation.validate()
    simulation = copy.deepcopy(simulation)
    factories = {
        "deepseek_v4_smoketest": "torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_debugmodel",
    }
    for preset, factory in (
        ("flash", "torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_flash"),
        ("pro", "torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_pro"),
        ("pro_20t", "torchtitan_simulator.presets.deepseek_v4_pro_20t"),
    ):
        for precision in ("bf16", "mxfp8"):
            name = f"deepseek_v4_{preset}_baseline_{precision}"
            factories[name] = factory
            if simulation.config == name and precision == "mxfp8":
                simulation.enable_quantized_training = True
    target = factories.get(simulation.config, simulation.config)
    if "." not in target:
        raise ValueError(f"Unknown simulator preset {simulation.config!r}; use a supported preset or module.factory")
    module_name, factory_name = target.rsplit(".", 1)
    factory = getattr(import_module(module_name), factory_name)
    trainer = factory(num_mtp_layers=simulation.num_mtp_modules)
    if simulation.model_overrides is not None:
        from torchtitan_simulator.model_overrides import apply_model_overrides

        if target not in set(factories.values()):
            raise ValueError(
                "Flat model_overrides require a supported simulator preset; custom factories own their config trees"
            )
        apply_model_overrides(trainer, simulation.model_overrides, num_mtp_modules=simulation.num_mtp_modules)
    model = trainer.model_spec.model
    if trainer.model_spec.name != "deepseek_v4":
        raise ValueError("This migration stage supports DeepSeek V4 only")
    adaptations = []
    if simulation.model_overrides is not None:
        adaptations.append("typed geometry overrides rebuilt the complete native model configuration tree")
    if trainer.compile.enable:
        adaptations.append("compile disabled for dispatch capture")
    trainer.compile.enable = False
    if not hasattr(trainer.training, "disable_cuda_graphs"):
        raise RuntimeError("Training configuration has no explicit CUDA graph control; capability review required")
    trainer.training.disable_cuda_graphs = True
    trainer.training.seq_len = simulation.seq_len
    if simulation.local_batch_size is not None:
        trainer.training.local_batch_size = simulation.local_batch_size
    if simulation.global_batch_size is not None:
        trainer.training.global_batch_size = simulation.global_batch_size
    trainer.parallelism.pipeline_parallel_degree = 1
    for name in (
        "data_parallel_replicate_degree",
        "data_parallel_shard_degree",
        "tensor_parallel_degree",
        "context_parallel_degree",
        "expert_parallel_degree",
    ):
        setattr(trainer.parallelism, name, getattr(simulation, name))
    from torchtitan.distributed.activation_checkpoint import FullAC

    from torchtitan_simulator.selective_ac import SimulatorSelectiveAC

    policies = {"none": None, "full": FullAC.Config, "selective": SimulatorSelectiveAC.Config}
    policy = policies[simulation.activation_checkpoint_mode]
    trainer.activation_checkpoint = policy() if policy is not None else None
    if simulation.activation_checkpoint_mode == "selective":
        trainer.activation_checkpoint = SimulatorSelectiveAC.Config(save_ops=simulation.selective_ac_save_ops)
    adaptations.append(f"native activation checkpoint policy: {simulation.activation_checkpoint_mode}")
    trainer.debug.moe_force_load_balance = True
    adaptations.append("balanced round-robin routing; actual data-dependent expert imbalance is not simulated")
    if trainer.debug.seed is None:
        trainer.debug.seed = 42
    trainer.override.imports = list(
        _DSV4_OVERRIDES if simulation.override_imports is None else simulation.override_imports
    )
    model.update_from_config(config=trainer)
    trainer.extension.quantization.enable_quantized_training = simulation.enable_quantized_training
    trainer.extension.quantization.recipe = simulation.quantization_recipe
    if simulation.enable_quantized_training:
        from torchtitan_npu.extensions.adaptor.torchao.converter import apply_quantization_converter

        trainer.model_spec = apply_quantization_converter(
            trainer.model_spec,
            trainer.extension.quantization,
            model_compile_enabled=False,
        )
        adaptations.append("native TorchAO-NPU all_mxfp8 recipe; native FQN selection without simulator overrides")
    from torchtitan.config.override import apply_overrides

    adaptations.extend(apply_overrides(trainer.override, trainer))
    return PreparedConfig(simulation, trainer, target, adaptations)


def build_meta_model(prepared: PreparedConfig):
    import torch

    with torch.device("meta"):
        model = prepared.trainer.model_spec.model.build()
        model.init_states(buffer_device=torch.device("meta"))
    return model
