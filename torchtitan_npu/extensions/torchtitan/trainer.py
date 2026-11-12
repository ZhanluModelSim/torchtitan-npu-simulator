# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ``CheckpointManager`` installs its upstream replacement at import time; keep
# this import before the remaining extension dependencies.
# ruff: noqa: I001
import logging
from copy import copy, deepcopy
from dataclasses import dataclass, field

from typing import Annotated, cast

import tyro
from torchtitan.components.checkpointer.base import BaseCheckpointManager
from torchtitan.config import derive
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

from torchtitan_npu.config import manager as config_manager
from torchtitan_npu.config.configs import (
    CompileConfig,
    ExtensionConfig,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.config.converters import TrainerConfigConverter
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.components.gradient_clipping import GradientClippingOptim  # noqa: F401
from torchtitan_npu.extensions.components.sdc import SDC, validate_sdc_replay_config
from torchtitan_npu.extensions.experiment.anticipatory_routing.config import (
    AnticipatoryRoutingConfig,
    validate_anticipatory_config,
)
from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import AnticipatorySchedule

from .tools.profiler import CANNProfiler


logger = logging.getLogger(__name__)


def set_allow_hf32(allow: bool) -> None:
    """Set HF32 consistently on the NPU math backends."""

    import torch_npu

    torch_npu.npu.matmul.allow_hf32 = allow
    torch_npu.npu.conv.allow_hf32 = allow
    torch_npu.npu.aclnn.allow_hf32 = allow


class TrainingEngineEx(TrainingEngine):
    """Upstream ``TrainingEngine`` with NPU microbatch hooks."""

    anticipatory_schedule: AnticipatorySchedule | None = None
    _sdc: SDC | None = None

    def _non_pp_forward_backward_microbatch(self, **kwargs):
        if self.anticipatory_schedule is not None:
            self.anticipatory_schedule.prepare_microbatch()
        result = super()._non_pp_forward_backward_microbatch(**kwargs)
        if self.anticipatory_schedule is not None:
            self.anticipatory_schedule.accumulate_microbatch_loss(result)
        if self._sdc is not None:
            self._sdc.finalize_sdc_step()
        return result


class TrainerEx(Trainer):
    """Base trainer for NPU-specific training features."""

    engine_cls = TrainingEngineEx

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        extension: ExtensionConfig = field(default_factory=ExtensionConfig)
        # These mutable schemas intentionally specialize upstream fields. The
        # registered converter preserves concrete recipe owners, while defaults
        # expose NPU fields to Tyro; recipes explicitly instantiate optional components.
        optim: OptimConfig = field(default_factory=OptimConfig)  # pyrefly: ignore [bad-override]
        checkpointer: Annotated[  # pyrefly: ignore [bad-override]
            BaseCheckpointManager.Config | None, tyro.conf.AvoidSubcommands
        ] = None
        compile: Annotated[CompileConfig | None, tyro.conf.AvoidSubcommands] = None  # pyrefly: ignore [bad-override]
        profiler: CANNProfiler.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=CANNProfiler.Config,
        )
        training: TrainingConfig = field(  # pyrefly: ignore [bad-override]
            default_factory=TrainingConfig,
        )
        sdc: SDC.Config = field(default_factory=SDC.Config)
        anticipatory: AnticipatoryRoutingConfig = field(default_factory=AnticipatoryRoutingConfig)

        def __post_init__(self) -> None:
            # ``slots=True`` dataclasses are recreated by the decorator, so a
            # zero-argument ``super()`` can retain the pre-decoration class cell.
            Trainer.Config.__post_init__(self)
            if type(self.checkpointer).__module__ == "torchtitan.components.checkpointer.torch_checkpointing":
                from torchtitan.components.checkpointer import torch_checkpointing as native

                from torchtitan_npu.extensions.torchtitan.components import torch_checkpointing as npu_native

                if type(self.checkpointer) is native.TorchCheckpointingManager.Config:
                    self.checkpointer = derive(self.checkpointer, npu_native.TorchCheckpointingManager.Config)
            if self.compile is not None:
                raise NotImplementedError("NPU model compilation is not supported in the eager upgrade")
            # Single switch: ``--training.enable-cpu-offload`` drives both the
            # FSDP offload policy (read by parallelize) and, through this
            # carrier field, the optimizer-container selection in the
            # swap_optimizer override branch (applied inside Trainer.__init__).
            # NPU optimizer schemas carry this field, including configs already
            # derived to an offload owner; upstream TorchFT containers do not.
            if hasattr(self.optim.optimizer, "_cpu_offload"):
                setattr(  # noqa: B010 - recipe-selected subtypes own the carrier
                    self.optim.optimizer, "_cpu_offload", self.training.enable_cpu_offload
                )
            self._post_init_optimizer()
            validate_anticipatory_config(self)
            if self.sdc_replayer is not None:
                validate_sdc_replay_config(self.sdc)

        def _post_init_optimizer(self) -> None:
            # Overrides may already have materialized and replaced the CLI schema.
            if not isinstance(self.optim.optimizer, OptimizerConfig):
                return
            self.optim.optimizer.materialize()
            if self.optim.optimizer.name == "Muon" and (
                self.parallelism.tensor_parallel_degree > 1 or self.parallelism.pipeline_parallel_degree > 1
            ):
                raise ValueError(
                    "DeepSeek-V4 DistMuon requires "
                    "tensor_parallel_degree=1 and pipeline_parallel_degree=1; "
                    "TP _StridedShard and PP stage-local parameter groups are not admitted yet"
                )

    def _carve_grad_slots(self, pool) -> None:
        """Pre-carve one pinned D2H gradient slot per offloaded FSDP param.

        The slots are carved from the same chunk pool as the parameters
        and stashed on each ``FSDPParam`` (``_pooled_grad_dest``), so they
        live with the model. Parameters that never use a slot are skipped:
        frozen parameters (no gradient) and parameters whose backward runs
        the synchronous fallback (post-accumulate hooks installed). The
        D2H gradient is the reduce-scatter shard, so the slot shape is the
        unpadded ``sharded_size``; on any shape or dtype mismatch the
        gradient offload path falls back to the original
        ``.to(non_blocking=True)`` allocation.
        """
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )

        seen: set[int] = set()
        for model in self.engine.model_parts:
            for module in model.modules():
                state = _get_module_fsdp_state(module)
                if state is None or id(state) in seen:
                    continue
                seen.add(id(state))
                for group in state._fsdp_param_groups:
                    for fsdp_param in group.fsdp_params:
                        sharded = fsdp_param.sharded_param
                        if (
                            not (fsdp_param.offload_to_cpu and fsdp_param.pin_memory)
                            or sharded is None
                            or not sharded.requires_grad
                            or getattr(sharded, "_post_accumulate_grad_hooks", None)
                        ):
                            continue
                        fsdp_param._pooled_grad_dest = pool.allocate(  # pyrefly: ignore [missing-attribute]
                            tuple(fsdp_param.sharded_size), sharded.dtype
                        )

    def __init__(self, config: Config):
        if getattr(config, "compile", None) is not None:
            raise NotImplementedError("NPU model compilation is not supported in the eager upgrade")
        if getattr(getattr(config, "optim", None), "ema", None) is not None:
            raise NotImplementedError("NPU EMA is not supported in the eager upgrade")
        quantization_config = config.extension.quantization
        if quantization_config.enable_quantized_training:
            from torchtitan_npu.extensions.adaptor.torchao.converter import apply_quantization_converter

            logger.info(
                "Applying TorchAO-NPU quantization recipe=%s before Trainer initialization",
                quantization_config.recipe,
            )
            model_compile_enabled = config.compile is not None and (
                getattr(config.compile, "model", None) is not None or hasattr(config.compile, "pass_pipeline")
            )
            config.model = apply_quantization_converter(
                config.model,
                quantization_config,
                model_compile_enabled=model_compile_enabled,
                override_config=config.override,
            )

        # v0.4.0 removed the upstream ``update_from_config`` call and applies
        # overrides before the engine exists. Run the model-owned preparation on a
        # private copy here so the sharding pass still precedes the overrides, which
        # derive their configs from it.
        model_config = getattr(config, "model", None)
        if model_config is not None:
            config.model = deepcopy(model_config)
            update = getattr(config.model, "update_from_config", None)
            if update is not None:
                update(config=config)

        set_allow_hf32(config.training.extension.allow_hf32)
        if getattr(config.training, "enable_cpu_offload", False):
            # CPU DTensor parameters need NPU-side initialization during
            # model materialization: Module._init_param fires inside
            # init_weights (after parallelize_fn applies the offload
            # policy), which is earlier than the optimizer container
            # exists. Without CPU offload the module is never imported.
            from torchtitan_npu.patches.torch_npu import cpu_dtensor_init

            cpu_dtensor_init.install()
            if hasattr(config.optim.optimizer, "_cpu_offload"):
                # Default the optimizer container to NPU-resident state so
                # ``--training.enable-cpu-offload`` works without listing an
                # optimizer override; a listed ``swap_optimizer`` (applied
                # inside Trainer.__init__) re-derives to the CPU-canonical-
                # state container, keeping optimizer-state offload opt-in.
                # Deriving here rather than in Config.__post_init__ keeps
                # config rebuilds (dataclasses.replace) on the plain schema,
                # and materialize() has already baked the Muon profile into
                # upstream param-group fields; the carrier-field guard keeps
                # foreign schemas (e.g. TorchFT) on their own containers.
                from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
                    HostSparseOptimizersContainer,
                )
                from torchtitan_npu.override.common.optimizer import (
                    CpuOffloadHostSparseNpuStateOptimizersContainer,
                    CpuOffloadNpuStateOptimizersContainer,
                )

                target = CpuOffloadNpuStateOptimizersContainer.Config
                if isinstance(config.optim.optimizer, HostSparseOptimizersContainer.Config):
                    # HostSparse schemas carry the carrier through
                    # OptimizerConfig; deriving them to the plain container
                    # would silently drop the sparse-table lifecycle.
                    target = CpuOffloadHostSparseNpuStateOptimizersContainer.Config
                config = copy(config)
                config.optim = copy(config.optim)
                # pyrefly: ignore [bad-argument-type, bad-assignment]
                config.optim.optimizer = derive(config.optim.optimizer, target)

            # Carve the per-parameter pins (to_empty()'s _apply ->
            # reset_sharded_param pins the offloaded shards there, before
            # the optimizer is built) and the per-parameter D2H gradient
            # slots from one flat pool of power-of-two chunks, defeating
            # the host pinned allocator's per-request size-class rounding.
            # Ownership follows the carved tensors: chunks stay alive via
            # the parameters and gradient slots, so they are released with
            # the model and a rebuilt Trainer does not accumulate pools.
            from torchtitan_npu.extensions.cpu_offload.pinned_pool import (
                PinnedFlatPool,
            )

            pool = PinnedFlatPool("fsdp-offload")
            with pool.pinned_pin_window():
                super().__init__(config)
            self._carve_grad_slots(pool)
        else:
            super().__init__(config)
        engine = cast("TrainingEngineEx", self.engine)
        engine._sdc = config.sdc.build(
            trainer_config=config,
            model_parts=self.engine.model_parts,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        if getattr(config.training, "enable_cpu_offload", False):
            from torchtitan_npu.override.common.optimizer import CpuOffloadOptimizersContainer

            if not isinstance(self.engine.optim.optimizers, CpuOffloadOptimizersContainer):
                raise ValueError(
                    "--training.enable-cpu-offload requires a CPU-offload optimizer "
                    "container, and this optimizer schema does not carry the NPU "
                    "offload carrier field, so no CPU-offload container can be "
                    "derived for it automatically"
                )

        self.anticipatory_schedule = AnticipatorySchedule(self) if config.anticipatory.enable else None
        engine.anticipatory_schedule = self.anticipatory_schedule

    def train_step(self, data_iterator):
        if self.anticipatory_schedule is not None:
            with self.anticipatory_schedule.training_step_context(data_iterator) as batches:
                return super().train_step(self.step_batches(batches))
        return super().train_step(self.step_batches(data_iterator))

    def step_batches(self, batches):
        """Adapt this step's batches after any anticipatory scheduling."""
        return batches

    def microbatch_generator(self, data_iterable):
        if self.anticipatory_schedule is not None:
            if data_iterable is not self.dataloader:
                raise ValueError("Anticipatory mode requires the trainer's own resumable dataloader")
            return self.anticipatory_schedule.data
        return super().microbatch_generator(data_iterable)

    def close(self) -> None:
        super().close()
        # ``train.py`` calls ``trainer.close()`` on both the normal and the
        # exception path; give optimizer containers (e.g. the CPU-offload
        # runtime owner) a chance to release their resources there.
        optimizers = getattr(getattr(getattr(self, "engine", None), "optim", None), "optimizers", None)
        if optimizers is not None and hasattr(optimizers, "close"):
            optimizers.close()


config_manager.register_config_converter(
    Trainer.Config,
    TrainerConfigConverter(
        target_type=TrainerEx.Config,
        component_types={
            "optim": OptimConfig,
            "optim.optimizer": OptimizerConfig,
            "checkpointer": CheckpointManager.Config,
            "profiler": CANNProfiler.Config,
            "training": TrainingConfig,
        },
    ),
)
