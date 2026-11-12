# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in local native checkpointing for the fixed TorchTitan API on NPU."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict
from torch_checkpointing.checkpoint_layout import LayoutInfo, SafetensorsSerialization
from torch_checkpointing.checkpoint_writer import CallbackArgs
from torch_checkpointing.hf.consolidation import consolidate_hf_safetensors_checkpoint
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.checkpointer import torch_checkpointing as upstream
from torchtitan.components.checkpointer.base import MODEL
from torchtitan.components.checkpointer.torch_checkpointing import (
    _default_backend_config,
    _item_specs,
    _sync_save_config,
)
from torchtitan.tools import filesystem
from torchtitan.tools.garbage_collector import GarbageCollector

from .torch_checkpointing_backend import adapt_checkpoint_manager

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class TorchCheckpointingManager(upstream.TorchCheckpointingManager):
    """Native checkpoint manager with NPU staging and metadata collectives."""

    @dataclass(kw_only=True, slots=True)
    class Config(upstream.TorchCheckpointingManager.Config):
        pass

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        adapt_checkpoint_manager(self._manager)

    @torch.no_grad()
    def _save_last_step(self, curr_step: int) -> None:
        # The final save creates a fresh synchronous backend that also needs NPU adaptation.
        if self.last_save_model_only:
            model_state = self.states[MODEL].state_dict()
            model_state = {
                key: value.to(self.export_dtype)
                if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != self.export_dtype
                else value
                for key, value in model_state.items()
            }
            states: dict[str, Any] = {MODEL: model_state}
            logger.info(f"Saving a model only checkpoint in {self.export_dtype} at last step, step {curr_step}.")
        else:
            states = self.states
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._manager.close()
        storage_config = self._manager_config.storage_config
        input_checkpoint_id = checkpoint_id
        item_specs: dict[str, ItemSpec] | None = None
        pre_finalize_callback: Callable[[CallbackArgs], None] | None = None
        if self.last_save_in_hf:
            assert self.sd_adapter is not None
            states = {MODEL: self.sd_adapter.to_hf(states[MODEL])}
            input_checkpoint_id = filesystem.join(checkpoint_id, "sharded")
            item_specs = _item_specs()
            model_spec = item_specs[MODEL]
            item_specs[MODEL] = ItemSpec(
                requires_copy=model_spec.requires_copy,
                layout=LayoutInfo(
                    f"{MODEL}_{{rank}}.safetensors",
                    SafetensorsSerialization(),
                ),
                resharder=model_spec.resharder,
                required=model_spec.required,
            )
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            hf_storage_config = storage_config or LocalFileSystemStorageConfig(use_direct_io=False)

            # CallbackArgs.path is the shard directory before the backend's atomic rename.
            def pre_finalize_callback(args: CallbackArgs) -> None:
                # The backend's write barrier runs after this callback; synchronize before consolidation.
                if dist.is_initialized():
                    dist.barrier()
                consolidate_hf_safetensors_checkpoint(
                    args.path,
                    output_dir=checkpoint_id,
                    item_key=MODEL,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    storage_config=hf_storage_config,
                )

        manager_config = _default_backend_config(
            _sync_save_config(),
            storage_config=storage_config,
            items=item_specs,
        )
        # torch_checkpointing 0.1.1 uses CallbackArgs; the upstream helper still types the old two-argument API.
        manager_config.pre_finalize_callback = pre_finalize_callback
        manager = manager_config.build()
        try:
            adapt_checkpoint_manager(manager)
            manager.save(
                input_checkpoint_id,
                _stateful_to_state_dict(states),
            )
        finally:
            manager.close()
        GarbageCollector.collect("GC collection invoked by native NPU checkpointer.")
