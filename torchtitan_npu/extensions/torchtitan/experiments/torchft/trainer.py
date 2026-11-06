# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit NPU TorchFT trainer configuration and normal training lifecycle."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer, FaultTolerantTrainingEngine

from torchtitan_npu.config import OptimConfig
from torchtitan_npu.extensions.torchtitan.experiments.torchft.ft_manager import FTManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.step_recovery import StepRecovery
from torchtitan_npu.extensions.torchtitan.trainer import NPUTrainingEngineMixin, TrainerEx
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model


class FaultTolerantTrainingEngineEx(NPUTrainingEngineMixin, FaultTolerantTrainingEngine):
    """TorchFT training engine with the NPU microbatch lifecycle hooks."""

    def __init__(self, config, *, model_config, max_num_documents, output_dir):
        super().__init__(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
            fault_tolerance=config.fault_tolerance,
        )


class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer):
    engine_cls = FaultTolerantTrainingEngineEx

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):  # pyrefly: ignore [bad-override]
        optim: OptimConfig = field(default_factory=lambda: OptimConfig(optimizer=TorchFTOptimizersContainerEx.Config()))
        fault_tolerance: FTManagerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=FTManagerEx.Config
        )
        checkpointer: TorchFTCheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=lambda: TorchFTCheckpointManager.Config(enable_ft_dataloader_checkpoints=True)
        )

        def _post_init_optimizer(self) -> None:
            # The TorchFT optimizer config has neither materialize() nor NPU Muon settings.
            pass

    def __init__(self, config: Config) -> None:
        self._validate_supported_config(config)
        super().__init__(config)
        engine = cast("FaultTolerantTrainingEngineEx", self.engine)
        ft_manager = engine.ft_manager
        if not isinstance(ft_manager, FTManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires FTManagerEx")
        manager = ft_manager.manager
        if not isinstance(manager, ManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires ManagerEx")
        self.recovery = StepRecovery(
            ft_manager.process_group.recovery,
            manager=manager,
            model_parts=cast("list[nn.Module]", engine.model_parts),
            train_state=engine,
            lr_schedulers=engine.optim.lr_schedulers,
            max_consecutive_recoveries=config.fault_tolerance.max_consecutive_hccl_recoveries,
            reset_auxiliary_state=self._reset_auxiliary_state,
        )

    @staticmethod
    def _validate_supported_config(config: Config) -> None:
        FTManagerEx.validate_config(config.fault_tolerance)
        if config.sdc_replayer is not None:
            raise ValueError("The pinned TorchFT trainer does not support the upstream SDC replay engine")
        if config.checkpointer is None or not config.checkpointer.enable_ft_dataloader_checkpoints:
            raise ValueError("NPU TorchFT requires checkpoint and per-replica dataloader checkpointing")

        if not isinstance(config.model, DeepSeekV4Model.Config):
            raise ValueError("NPU TorchFT currently supports DeepSeek-V4 Flash recipes only")

        parallelism = config.parallelism
        if parallelism.data_parallel_replicate_degree != 1:
            raise ValueError("TorchFT manages elastic DP replicas; data_parallel_replicate_degree must be 1")
        if parallelism.data_parallel_shard_degree <= 1 or parallelism.expert_parallel_degree <= 1:
            raise ValueError("NPU TorchFT currently requires both FSDP and expert parallelism")
        if (
            parallelism.tensor_parallel_degree != 1
            or parallelism.context_parallel_degree != 1
            or parallelism.pipeline_parallel_degree != 1
        ):
            raise ValueError("NPU TorchFT currently supports FSDP+EP only; TP, CP and PP are unsupported")

    @torch.no_grad()
    def _reset_auxiliary_state(self) -> None:
        # Auxiliary-loss modules expose a local accumulator and shared step metrics.
        # Inspect that state without coupling recovery to a model's patch class.
        for part in self.engine.model_parts:
            for module in part.modules():
                accumulator = module._buffers.get("instance_acc")
                step_metrics = getattr(type(module), "group_acc", None)
                if isinstance(accumulator, torch.Tensor) and isinstance(step_metrics, dict):
                    accumulator.zero_()
                    step_metrics.clear()

    def train(self) -> None:
        # The NPU trainer owns construction; the FT lifecycle still provides the
        # per-replica profiler and semi-sync context around the custom step.
        FaultTolerantTrainer.train(self)

    def train_step(self, data_iterator):
        return self.recovery.run_step(self._train_step, data_iterator)

    def _train_step(self, data_iterator: Iterator):
        """Keep the v0.3 FT loop, but finish metrics before authorizing updates."""
        engine = cast("FaultTolerantTrainingEngineEx", self.engine)
        parallelism_context = engine.parallelism_context
        current_step = engine.num_completed_steps + 1
        lr = engine.optim.lr_schedulers.schedulers[0].get_last_lr()[0]
        should_log = self.metrics_processor.should_log(current_step)

        microbatch_groups = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pp_microbatches):
                microbatch = next(data_iterator)
                local_valid_tokens += microbatch.num_valid_tokens
                microbatches.append(microbatch)
            microbatch_groups.append(microbatches)

        global_valid_tokens = local_valid_tokens.to(engine.device)
        if parallelism_context.dp_enabled:
            global_valid_tokens = dist_utils.dist_sum_tensor(global_valid_tokens, parallelism_context.get_mesh("dp"))

        forward_backward_result = engine.forward_backward(
            microbatch_groups=microbatch_groups,
            global_valid_tokens=global_valid_tokens,
        )

        if not should_log:
            engine.optim_step()
            return

        # No new cross-replica collective may be issued after a successful vote,
        # so finish the logging metrics before the optimizer commit.
        accumulated_loss = forward_backward_result.loss
        if parallelism_context.dp_cp_enabled:
            ft_pg = engine.ft_manager.loss_sync_pg
            assert ft_pg is not None, "Synchronous TorchFT requires a loss process group"
            loss_mesh = parallelism_context.get_optional_mesh("loss")
            local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
            global_avg_loss = dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg) / ft_pg.size()
            global_max_loss = dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(engine.ntokens_seen, dtype=torch.int64, device=engine.device),
                loss_mesh,
                ft_pg,
            )
        else:
            global_avg_loss = global_max_loss = accumulated_loss.item()
            global_ntokens_seen = engine.ntokens_seen

        grad_norm = engine.optim_step()

        self.metrics_processor.log(
            current_step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics={"n_tokens_seen": global_ntokens_seen, "lr": lr},
        )
