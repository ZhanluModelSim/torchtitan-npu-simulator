# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit NPU TorchFT trainer configuration and normal training lifecycle."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer

from torchtitan_npu.extensions.torchtitan.experiments.torchft.ft_manager import FTManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.manager import ManagerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainerEx
from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.step_recovery import StepRecovery
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model


class FaultTolerantTrainerEx(TrainerEx, FaultTolerantTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):  # pyrefly: ignore [bad-override]
        optimizer: TorchFTOptimizersContainerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=TorchFTOptimizersContainerEx.Config
        )
        fault_tolerance: FTManagerEx.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=FTManagerEx.Config
        )
        checkpoint: TorchFTCheckpointManager.Config = field(  # pyrefly: ignore [bad-override]
            default_factory=lambda: TorchFTCheckpointManager.Config(enable=True, enable_ft_dataloader_checkpoints=True)
        )

        def _post_init_optimizer(self) -> None:
            # The TorchFT optimizer config has neither materialize() nor NPU Muon settings.
            pass

    def __init__(self, config: Config) -> None:
        self._validate_supported_config(config)
        super().__init__(config)
        ft_manager = self.ft_manager
        if not isinstance(ft_manager, FTManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires FTManagerEx")
        manager = ft_manager.manager
        if not isinstance(manager, ManagerEx):
            raise TypeError("FaultTolerantTrainerEx requires ManagerEx")
        self.recovery = StepRecovery(
            ft_manager.process_group.recovery,
            manager=manager,
            model_parts=self.model_parts,
            train_state=self,
            lr_schedulers=self.lr_schedulers,
            max_consecutive_recoveries=config.fault_tolerance.max_consecutive_hccl_recoveries,
            reset_auxiliary_state=self._reset_auxiliary_state,
        )

    @staticmethod
    def _validate_supported_config(config: Config) -> None:
        FTManagerEx.validate_config(config.fault_tolerance)
        if not config.checkpoint.enable or not config.checkpoint.enable_ft_dataloader_checkpoints:
            raise ValueError("NPU TorchFT requires checkpoint and per-replica dataloader checkpointing")

        model_spec = config.model_spec
        if (
            model_spec is None
            or not isinstance(model_spec.model, DeepSeekV4Model.Config)
            or not (model_spec.flavor == "deepseek_v4_flash" or model_spec.flavor.startswith("deepseek_v4_flash_"))
        ):
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
        for part in self.model_parts:
            for module in part.modules():
                accumulator = module._buffers.get("instance_acc")
                step_metrics = getattr(type(module), "group_acc", None)
                if isinstance(accumulator, torch.Tensor) and isinstance(step_metrics, dict):
                    accumulator.zero_()
                    step_metrics.clear()

    def init_distributed(self):
        dist_utils.set_spmd_backend(self.config.parallelism.spmd_backend)
        return super().init_distributed()

    def train_step(self, data_iterator):
        return self.recovery.run_step(self._train_step, data_iterator)

    def _train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
        """Keep the v0.3 FT loop, but finish metrics before authorizing updates."""
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        should_log = self.metrics_processor.should_log(self.step)
        parallel_dims = self.parallel_dims

        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pipeline_parallel_microbatches):
                input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)

        global_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            global_valid_tokens = dist_utils.dist_sum_tensor(global_valid_tokens, parallel_dims.get_mesh("batch"))

        accumulated_loss: torch.Tensor | None = None
        for microbatches in microbatch_groups:
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(self.device)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device))
            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            loss = self.forward_backward_step(
                input_dict=fwd_bwd_input_dict,
                labels=fwd_bwd_labels,
                global_valid_tokens=global_valid_tokens,
            )
            if should_log:
                loss = loss.detach()
                if accumulated_loss is None:
                    accumulated_loss = loss.clone()
                else:
                    accumulated_loss.add_(loss)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        if not should_log:
            self.optimizers.step()
            self.lr_schedulers.step()
            return

        assert accumulated_loss is not None
        if parallel_dims.dp_cp_enabled:
            ft_pg = self.ft_manager.loss_sync_pg
            assert ft_pg is not None, "Synchronous TorchFT requires a loss process group"
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
            global_avg_loss = dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg) / ft_pg.size()
            global_max_loss = dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
                ft_pg,
            )
        else:
            global_avg_loss = global_max_loss = accumulated_loss.item()
            global_ntokens_seen = self.ntokens_seen

        # No new cross-replica collective may be issued after a successful vote.
        self.optimizers.step()
        self.lr_schedulers.step()
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics={"n_tokens_seen": global_ntokens_seen, "lr": lr},
        )
