# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in local native checkpointing for the fixed TorchTitan API on NPU."""

from dataclasses import dataclass

import torch
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict
from torchtitan.components.checkpointer import torch_checkpointing as upstream
from torchtitan.components.checkpointer.base import MODEL
from torchtitan.components.checkpointer.torch_checkpointing import _default_backend_config, _sync_save_config
from torchtitan.tools.utils import GarbageCollection

from torchtitan_npu.patches.torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager as NativeLoadManager,
)

from .torch_checkpointing_backend import adapt_checkpoint_manager


class TorchCheckpointingManager(NativeLoadManager):
    """Native checkpoint manager with NPU staging and metadata collectives."""

    @dataclass(kw_only=True, slots=True)
    class Config(upstream.TorchCheckpointingManager.Config):
        pass

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if self.enable:
            adapt_checkpoint_manager(self._manager)

    @torch.no_grad()
    def _save_last_step(self, curr_step: int) -> None:
        if self.last_save_model_only:
            model_state = self.states[MODEL].state_dict()
            model_state = {
                key: value.to(self.export_dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != self.export_dtype
                else value
                for key, value in model_state.items()
            }
            states = {MODEL: model_state}
        else:
            states = self.states
        self._manager.close()
        config = _default_backend_config(
            _sync_save_config(),
            storage_config=self._manager_config.storage_config,
        )
        # The fixed upstream final-save path builds a fresh backend directly.
        manager = config.build()
        try:
            adapt_checkpoint_manager(manager)
            manager.save(self._create_checkpoint_id(curr_step), _stateful_to_state_dict(states))
        finally:
            manager.close()
        GarbageCollection.collect("GC collection invoked by native NPU checkpointer.")
