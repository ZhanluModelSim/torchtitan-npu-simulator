# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/4191
# Upstream-Commit: 3dadef77c25d7373e628b7db1cd7345b4af067fc
# Remove this opt-in subclass after the pinned TorchTitan includes #4191, and
# inherit its native manager directly. Do not backport the global Base/DCP
# refactor: dc3985 still dispatches through the _load hook.

"""Native-load backport for TorchTitan's dc3985 _load hook."""

import logging

import torch
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.checkpointer.base import MODEL
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager as UpstreamTorchCheckpointingManager,
)
from torchtitan.observability import structured_logger as sl

logger = logging.getLogger(__name__)


class TorchCheckpointingManager(UpstreamTorchCheckpointingManager):
    """Internal load backport; build through the NPU extension's Config."""

    @sl.log_trace_span("checkpoint_load")
    @torch.no_grad()
    def _load(self, step: int = -1) -> bool:
        model_only = step == 0
        if step >= 0:
            checkpoint_id = self._create_checkpoint_id(step)
        else:
            latest = self._find_load_step()
            if latest >= 0:
                checkpoint_id = self._create_checkpoint_id(latest)
                model_only = latest == 0
            else:
                if not self.initial_load_path:
                    return False
                if self.initial_load_in_hf:
                    raise ValueError("TorchCheckpointingManager does not yet support loading Hugging Face checkpoints.")
                checkpoint_id = self.initial_load_path
                model_only = self.initial_load_model_only
        if not self._storage.isdir(checkpoint_id):
            raise FileNotFoundError(f"Native checkpoint directory does not exist: {checkpoint_id}")
        if not self._is_valid_checkpoint(checkpoint_id):
            raise ValueError(f"Checkpoint {checkpoint_id!r} is not a native torch_checkpointing checkpoint.")
        if model_only:
            selected = {MODEL: self.states[MODEL]}
        else:
            for exclude_key in self.exclude_from_loading:
                if exclude_key not in self.states:
                    raise ValueError(f"{exclude_key} not found in state_dict.")
            selected = {key: value for key, value in self.states.items() if key not in self.exclude_from_loading}
        state_dict = _stateful_to_state_dict(selected)
        loaded = self._manager.load(checkpoint_id, into=state_dict, strict=True)
        for key, target in selected.items():
            if isinstance(target, Stateful):
                target.load_state_dict(state_dict[key])
            elif loaded[key] is not target:
                raise TypeError(f"Cannot restore non-Stateful checkpoint state {key!r} of type {type(target).__name__}")
        logger.info("Loaded native %s checkpoint from %s.", "model-only" if model_only else "full-state", checkpoint_id)
        return True
