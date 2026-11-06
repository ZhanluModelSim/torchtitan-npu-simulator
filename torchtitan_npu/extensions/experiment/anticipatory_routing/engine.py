# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge jojo's schedule to the monolithic 0.3.0 Trainer."""

import copy
import logging
import math
import random
import re
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.checkpointer import DATALOADER, LR_SCHEDULER, MODEL, OPTIMIZER, TRAIN_STATE
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import filesystem

from .data import TrainingDataIterator

logger = logging.getLogger(__name__)

REQUIRED_STATES = (MODEL, DATALOADER, LR_SCHEDULER, OPTIMIZER, TRAIN_STATE)


class AnticipatoryTrainingEngine:
    """Own auxiliary recovery state; all optimizer work stays in Trainer."""

    def __init__(self, trainer):
        self.trainer = trainer
        # TorchTitan ebdfc8 moved model execution, optimizer, checkpoint and
        # train-state ownership to ``Trainer.engine``.  Keep this recovery
        # helper as a thin policy object; never reach through the trainer for
        # those resources.
        self.runtime = trainer.engine
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
        return self.runtime.device

    @property
    def num_completed_steps(self):
        return self.runtime.num_completed_steps

    @property
    def remaining_steps(self):
        return max(0, self.runtime.config.training.steps - self.num_completed_steps)

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
        if self._microbatch_index != self.runtime.num_accumulation_steps:
            raise RuntimeError("Unexpected number of accumulated microbatch losses")
        self.last_global_loss = float(
            dist_utils.dist_sum(
                self._step_loss_sum,  # pyrefly: ignore [bad-argument-type]
                self.runtime.parallelism_context.get_optional_mesh("loss"),
            )
        )
        if not math.isfinite(self.last_global_loss):
            raise RuntimeError(f"Non-finite training loss at step {self.runtime.num_completed_steps}")

    def _reset_fsdp(self):
        for part in self.runtime.model_parts:
            if isinstance(part, FSDPModule) and part._get_fsdp_state()._is_root is not None:
                part.reset_iter_state()

    @contextmanager
    def _isolated_forward(self):
        from torchtitan.models.common.aux_loss import AuxLoss

        # Save registered buffers, including model-specific forward statistics.
        # Parameters are not copied or changed by this no-grad forward.
        buffers = [
            (m, name, value, value.detach().clone())
            for part in self.runtime.model_parts
            for m in part.modules()
            for name, value in m._buffers.items()
            if value is not None
        ]
        aux = copy.deepcopy(AuxLoss.group_acc)
        tokens = self.runtime.ntokens_seen
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
            self.runtime.ntokens_seen = tokens
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
    def forward_only_microbatch(self, batch: TrainingMicrobatch):
        # ``TrainingMicrobatch`` owns the new input/loss protocol.  Build a
        # device-local input dict and run the same preprocessing seam as
        # ``TrainingEngine._preprocess_microbatch_groups``; the forward is
        # isolated so router buffers and auxiliary-loss counters are restored.
        with self._isolated_forward():
            model = self.runtime.model_parts[0]
            input_dict = batch.to_input_dict(self.device, non_blocking=True)
            input_dict.pop("num_valid_tokens", None)
            with self.runtime.parallelism_context.activate_spmd():
                inputs, _, kwargs = model.preprocess_inputs(
                    input_dict,
                    parallelism_context=self.runtime.parallelism_context,
                    parallelism=self.runtime.config.parallelism,
                    max_num_documents=self.runtime.max_num_documents,
                    max_context_length=self.runtime.config.training.max_context_length,
                    **self.runtime.preprocess_inputs_kwargs,
                )
                model(inputs, **kwargs)

    def install_checkpoint_save_guard(self):
        """Gate this trainer's saves without coupling the checkpoint component to routing."""
        if not hasattr(self.runtime, "checkpointer"):
            # ``max_rollbacks=0`` is a valid configuration for route-capture
            # experiments without recovery checkpoints.  The validator admits
            # it explicitly, so the schedule must not recreate the old
            # Trainer.checkpointer requirement here.
            return
        save = self.runtime.checkpointer.save

        @wraps(save)
        def guarded_save(*args, **kwargs):
            if self.suppress_checkpoint_saves or self.failed:
                return False
            return save(*args, **kwargs)

        self.runtime.checkpointer.save = guarded_save

    def _required_states(self):
        required = [key for key in REQUIRED_STATES if key != MODEL]
        if self.runtime.config.optim.ema is not None:
            from torchtitan.components.checkpointer import EMA as EMA_OPTIMIZER

            required.append(EMA_OPTIMIZER)
        return required

    def find_rollback_target(self, onset_step):
        checkpointer = self.runtime.checkpointer
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
        runtime = self.runtime
        runtime.checkpointer.maybe_wait_for_staging()
        runtime.checkpointer.maybe_wait_for_saving()
        ema = getattr(runtime.optim, "ema", None)
        if ema is not None:
            ema.wait_for_param_reads()
        if not runtime.checkpointer.load(step=step) or runtime.num_completed_steps != step:
            raise RuntimeError(f"Failed to restore checkpoint step {step}")
        self.data.reset()
        self._reset_transient_training_state()
        logger.info("Anticipatory routing rolled back to step %d", runtime.num_completed_steps)

    def _reset_transient_training_state(self):
        from torchtitan.models.common.aux_loss import AuxLoss
        from torchtitan.models.common.moe import MoE

        # Ascend uses the eager gradient lifecycle here; no CUDA graph storage must be preserved.
        self.runtime.optim.zero_grad(set_to_none=True)
        with torch.no_grad():
            for part in self.runtime.model_parts:
                for module in part.modules():
                    if isinstance(module, MoE):
                        module.router.tokens_per_expert_E.zero_()
                    if isinstance(module, AuxLoss):
                        module.instance_acc.zero_()
        AuxLoss.group_acc.clear()
        self.last_global_loss = None
        self._step_loss_sum = None
        self._reset_fsdp()
