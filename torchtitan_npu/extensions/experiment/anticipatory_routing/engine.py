# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge jojo's schedule to the monolithic 0.3.0 Trainer."""

import copy
import math
import random
import re
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torchtitan.components.checkpoint import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger

from .data import TrainingDataIterator, copy_batch

REQUIRED_STATES = (MODEL, DATALOADER, LR_SCHEDULER, OPTIMIZER, TRAIN_STATE)


class AnticipatoryTrainingEngine:
    """Own auxiliary recovery state; all optimizer work stays in Trainer."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.data = TrainingDataIterator(trainer.dataloader)
        self.in_step = False
        self.failed = False
        self.suppress_checkpoint_saves = False
        self.last_global_loss = None
        self._step_loss_sum = None
        self._microbatch_index = 0
        if not getattr(trainer.dataloader, "in_order", True):
            raise ValueError("Anticipatory routing requires an ordered resumable dataloader")
        self.install_checkpoint_save_guard()

    @property
    def device(self):
        return self.trainer.device

    @property
    def num_completed_steps(self):
        # 0.3.0 increments before entering train_step; jojo increments afterwards.
        return self.trainer.step - int(self.in_step)

    @property
    def remaining_steps(self):
        return max(0, self.trainer.config.training.steps - self.num_completed_steps)

    def begin_step(self):
        self._microbatch_index = 0
        self._step_loss_sum = None

    def accumulate_microbatch_loss(self, loss):
        """Accumulate detached loss and advance the slot index; detection is step-level."""
        value = loss.detach()
        if self._step_loss_sum is None:
            self._step_loss_sum = value.clone()
        else:
            self._step_loss_sum.add_(value)
        self._microbatch_index += 1

    def reduce_step_loss(self):
        if self._microbatch_index != self.trainer.gradient_accumulation_steps:
            raise RuntimeError("Unexpected number of accumulated microbatch losses")
        self.last_global_loss = float(
            dist_utils.dist_sum(
                self._step_loss_sum,  # pyrefly: ignore [bad-argument-type]
                self.trainer.parallel_dims.get_optional_mesh("loss"),
            )
        )
        if not math.isfinite(self.last_global_loss):
            raise RuntimeError(f"Non-finite training loss at step {self.trainer.step}")

    def _reset_fsdp(self):
        for part in self.trainer.model_parts:
            if isinstance(part, FSDPModule) and part._get_fsdp_state()._is_root is not None:
                part.reset_iter_state()

    @contextmanager
    def _isolated_forward(self):
        from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

        trainer = self.trainer
        # Save registered buffers, including model-specific forward statistics.
        # Parameters are not copied or changed by this no-grad forward.
        buffers = [
            (m, name, value, value.detach().clone())
            for part in trainer.model_parts
            for m in part.modules()
            for name, value in m._buffers.items()
            if value is not None
        ]
        aux = copy.deepcopy(AuxLoss.group_acc)
        tokens = trainer.ntokens_seen
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        devices = [] if self.device.type == "cpu" else [self.device.index or 0]
        try:
            with torch.random.fork_rng(devices=devices, device_type=self.device.type):
                yield
        finally:
            with torch.no_grad():
                for module, name, original, before in buffers:
                    original.copy_(before)
                    module._buffers[name] = original
            AuxLoss.group_acc.clear()
            AuxLoss.group_acc.update(aux)
            trainer.ntokens_seen = tokens
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            self._reset_fsdp()

    def initialize_routing(self, cache, batch):
        """Allocate runtime tensors without changing the real training path.

        Reuse the first already-fetched microbatch and the existing isolation.
        In particular, do not mix eager and compiled FullAC training statistics
        within the first optimizer step just to initialize these tensors.
        """
        if cache.needs_materialization:
            # This one-time shape discovery is not a recorded queue entry.
            try:
                cache.begin_initialization()
                with torch.compiler.set_stance("force_eager"):
                    self.forward_only_microbatch(batch)
            finally:
                cache.reset()
            if cache.needs_materialization:
                raise RuntimeError("The initialization forward did not visit every dynamic router")

    @torch.no_grad()
    def forward_only_microbatch(self, batch):
        # Preprocessing is allowed to mutate its input; keep queued data intact.
        with self._isolated_forward():
            inputs, labels = copy_batch(batch, device=self.device)
            inputs, _, kwargs = self.trainer.post_dataloading_process(inputs, labels)
            with self.trainer.train_context():
                self.trainer.model_parts[0](inputs, **kwargs)

    def install_checkpoint_save_guard(self):
        """Gate this trainer's saves without coupling the checkpoint component to routing."""
        save = self.trainer.checkpointer.save

        @wraps(save)
        def guarded_save(*args, **kwargs):
            if self.suppress_checkpoint_saves or self.failed:
                return False
            return save(*args, **kwargs)

        self.trainer.checkpointer.save = guarded_save

    def _required_states(self):
        required = [key for key in REQUIRED_STATES if key != MODEL]
        if getattr(getattr(self.trainer.config, "ema_weights", None), "enable", False):
            from torchtitan_npu.patches.torchtitan.components.ema import EMA_OPTIMIZER

            required.append(EMA_OPTIMIZER)
        return required

    def find_rollback_target(self, onset_step):
        checkpointer = self.trainer.checkpointer
        checkpointer.maybe_wait_for_staging()
        checkpointer.maybe_wait_for_saving()
        # 0.3.0 flattens model FQNs into the DCP root; there is no model prefix.
        # All ranks materialize state keys before the rank-zero metadata scan,
        # since obtaining an FSDP state dict may involve collectives.
        model_keys = set(checkpointer.states[MODEL].state_dict())
        target = -1
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        if not distributed or dist.get_rank() == 0:
            folder = checkpointer.folder
            if filesystem.isdir(folder):
                for name in filesystem.listdir(folder):
                    match = re.fullmatch(r"step-(\d+)", name)
                    if not match:
                        continue
                    step = int(match.group(1))
                    if not 0 < step < onset_step or step <= target:
                        continue
                    path = filesystem.join(folder, name)
                    try:
                        # Recovery requires native resumable DCP, not HF export.
                        keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
                        missing_states = [
                            key
                            for key in self._required_states()
                            if not any(k == key or k.startswith(key + ".") for k in keys)
                        ]
                        missing_model = model_keys.difference(keys)
                        if missing_states or missing_model:
                            logger.warning(
                                "Skipping checkpoint %s: missing training states=%s, missing model keys=%s",
                                path,
                                missing_states,
                                sorted(missing_model),
                            )
                        else:
                            target = step
                    except Exception as error:
                        logger.warning("Ignoring unreadable checkpoint %s: %s", path, error)
        if distributed:
            selected = torch.tensor(target, dtype=torch.int64, device=self.device)
            dist.broadcast(selected, src=0)
            target = int(selected.item())
        return target if target > 0 else None

    def rollback_to(self, step):
        trainer = self.trainer
        trainer.checkpointer.maybe_wait_for_staging()
        trainer.checkpointer.maybe_wait_for_saving()
        ema = getattr(trainer, "ema_optimizer", None)
        if ema is not None:
            ema.wait_for_param_reads()
        if not trainer.checkpointer.load(step=step) or trainer.step != step:
            raise RuntimeError(f"Failed to restore checkpoint step {step}")
        self.data.reset()
        self._reset_transient_training_state()
        logger.info("Anticipatory routing rolled back to step %d", trainer.step)

    def _reset_transient_training_state(self):
        from torchtitan.models.common.moe import MoE

        from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

        # Ascend uses the eager gradient lifecycle here; no CUDA graph storage must be preserved.
        self.trainer.optimizers.zero_grad(set_to_none=True)
        with torch.no_grad():
            for part in self.trainer.model_parts:
                for module in part.modules():
                    if isinstance(module, MoE):
                        module.tokens_per_expert_E.zero_()
                    if isinstance(module, AuxLoss):
                        module.instance_acc.zero_()
        AuxLoss.group_acc.clear()
        self.last_global_loss = None
        self._step_loss_sum = None
        self._reset_fsdp()
