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
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics

from torchtitan_npu.extensions.torchtitan.experiments.torchft.ft_manager import FTManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import (
    TorchFTOptimEx,
    TorchFTOptimizersContainerEx,
)
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.step_recovery import StepRecovery
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx, TrainingEngineEx
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model


class FaultTolerantTrainingEngineEx(TrainingEngineEx, FaultTolerantTrainingEngine):
    """TorchFT training engine with the NPU microbatch lifecycle hooks."""

    def _initialize_optim(self) -> None:
        # Overrides run after recipe validation and may change component owners.
        if not isinstance(self.config.optim, TorchFTOptimEx.Config) or not isinstance(
            self.config.optim.optimizer, TorchFTOptimizersContainerEx.Config
        ):
            raise ValueError("NPU TorchFT requires TorchFTOptimEx and TorchFTOptimizersContainerEx")
        super()._initialize_optim()

    def finish_discarded_step(self) -> None:
        """Advance the consumed attempt without updating weights or EMA.

        Match upstream FT rejection: training.steps counts consumed attempts,
        including rejected ones, rather than successful parameter updates.
        """
        self.optim.lr_schedulers.step()
        self.num_completed_steps += 1


class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):  # pyrefly: ignore [bad-override]
        optim: TorchFTOptimEx.Config = field(default_factory=TorchFTOptimEx.Config)  # pyrefly: ignore [bad-override]
        fault_tolerance: FTManagerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=FTManagerEx.Config
        )
        checkpointer: TorchFTCheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=lambda: TorchFTCheckpointManager.Config(enable_ft_dataloader_checkpoints=True)
        )

        def _post_init_optimizer(self) -> None:
            # The TorchFT optimizer config uses its own container schema.
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
            finish_discarded_step=engine.finish_discarded_step,
            max_consecutive_recoveries=config.fault_tolerance.max_consecutive_hccl_recoveries,
            reset_auxiliary_state=self._reset_auxiliary_state,
        )

    def _initialize_trainer(self, config: TrainerEx.Config) -> None:
        # Native FT initialization accounts for all replicas in dataloader
        # sharding and accumulation; the ordinary Trainer cannot do that.
        FaultTolerantTrainer.__init__(self, cast("FaultTolerantTrainer.Config", config))

    @staticmethod
    def _validate_supported_config(config: Config) -> None:
        FTManagerEx.validate_config(config.fault_tolerance)
        if not isinstance(config.optim, TorchFTOptimEx.Config) or not isinstance(
            config.optim.optimizer, TorchFTOptimizersContainerEx.Config
        ):
            raise ValueError("NPU TorchFT requires TorchFTOptimEx and TorchFTOptimizersContainerEx")
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
        engine = cast("FaultTolerantTrainingEngineEx", self.engine)
        if engine._sdc is not None:
            engine._sdc.discard_step()
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
        """Finish cross-replica metrics collectives before authorizing an update."""
        engine = cast("FaultTolerantTrainingEngineEx", self.engine)
        parallelism_context = engine.parallelism_context
        current_step = engine.num_completed_steps + 1
        lr = engine.optim.lr_schedulers.schedulers[0].get_last_lr()[0]
        should_log = self.metrics_processor.should_log(current_step)
        if self.metrics_processor.step_last_log == engine.num_completed_steps:
            self.metrics_processor.reset()

        microbatch_groups = []
        local_loss_tokens: torch.Tensor | None = None
        local_routing_tokens: torch.Tensor | None = None
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pp_microbatches):
                microbatch = next(data_iterator)
                if local_loss_tokens is None:
                    local_loss_tokens = torch.zeros_like(microbatch.loss_token_counts)
                    local_routing_tokens = torch.zeros_like(microbatch.routing_token_counts)
                local_loss_tokens.add_(microbatch.loss_token_counts)
                assert local_routing_tokens is not None
                local_routing_tokens.add_(microbatch.routing_token_counts)
                microbatches.append(microbatch)
            microbatch_groups.append(microbatches)

        assert local_loss_tokens is not None
        assert local_routing_tokens is not None
        num_loss_objectives = local_loss_tokens.numel()
        global_token_counts = torch.cat((local_loss_tokens.reshape(-1), local_routing_tokens.reshape(-1))).to(
            engine.device
        )
        if parallelism_context.dp_enabled:
            dp_mesh = parallelism_context.get_mesh("dp")
            global_token_counts = dist_utils.dist_sum_tensor(global_token_counts, dp_mesh)
        global_loss_token_counts = global_token_counts[:num_loss_objectives].reshape(local_loss_tokens.shape)
        global_routing_token_counts = global_token_counts[num_loss_objectives:].reshape(local_routing_tokens.shape)

        forward_backward_result = engine.forward_backward(
            microbatch_groups=microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )

        if not should_log:
            engine.optim_step()
            self.recovery.mark_step_finished()
            return

        # No new cross-replica collective may be issued after a successful vote,
        # so finish the logging metrics before the optimizer commit.
        accumulated_loss = forward_backward_result.loss
        if parallelism_context.dp_cp_enabled:
            ft_pg = engine.ft_manager.loss_sync_pg
            assert ft_pg is not None, "Synchronous TorchFT requires a loss process group"
            loss_mesh = parallelism_context.get_optional_mesh("loss")
            local_main_tokens = local_loss_tokens if local_loss_tokens.ndim == 0 else local_loss_tokens[0]
            global_main_tokens = (
                global_loss_token_counts if global_loss_token_counts.ndim == 0 else global_loss_token_counts[0]
            )
            local_avg_loss = accumulated_loss * global_main_tokens / local_main_tokens
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
        self.recovery.mark_step_finished()

        self.metrics_processor.log(
            current_step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics={
                "n_tokens_seen": global_ntokens_seen,
                "lr": lr,
                **collect_aux_loss_metrics(parallelism_context),
            },
        )
