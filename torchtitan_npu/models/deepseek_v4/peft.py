# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ["DeepSeekV4PEFTCheckpointManager", "select_adapter_only_state"]

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpointer import EMA, MODEL
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.tools import filesystem

from torchtitan_npu.extensions.torchtitan.components import checkpoint as checkpoint_extensions
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager as NPUCheckpointManager

from .lora import DeepSeekPEFTAdapter, is_lora_parameter

logger = logging.getLogger(__name__)


def select_adapter_only_state(
    flattened_state: dict[str, Any],
    non_model_keys: set[str],
    *,
    model_buffer_keys: set[str] | None = None,
    model_trainable_keys: set[str] | None = None,
) -> dict[str, Any]:
    retained_keys = non_model_keys | (model_buffer_keys or set()) | (model_trainable_keys or set())
    return {key: value for key, value in flattened_state.items() if key in retained_keys or is_lora_parameter(key)}


class DeepSeekV4PEFTCheckpointManager(NPUCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(NPUCheckpointManager.Config):
        last_save_in_peft: bool = True
        periodic_save_adapter_only: bool = True
        save_training_state: bool = False
        peft_base_model_name_or_path: str | None = None

        def __post_init__(self) -> None:
            if self.last_save_in_hf:
                raise ValueError("last_save_in_hf cannot preserve LoRA adapters; use PEFT export or native DCP")
            NPUCheckpointManager.Config.__post_init__(self)
            if self.last_save_in_peft and filesystem.is_remote(self.folder):
                raise ValueError("PEFT export requires a local checkpoint folder")

    def __init__(self, config: Config, **kwargs) -> None:
        if config.last_save_in_hf:
            raise ValueError("last_save_in_hf cannot preserve LoRA adapters; use PEFT export or native DCP")
        if config.last_save_in_peft:
            destination = filesystem.join(kwargs.get("base_folder", ""), config.folder)
            if filesystem.is_remote(destination):
                raise ValueError("PEFT export requires a local checkpoint folder")

            adapter = kwargs.get("sd_adapter")
            if not isinstance(adapter, DeepSeekPEFTAdapter):
                raise TypeError("DeepSeekV4 PEFT export requires DeepSeekV4StateDictAdapter")
            adapter.peft_adapter_config()
        super().__init__(config, **kwargs)
        self.peft_base_model_name_or_path = config.peft_base_model_name_or_path
        self.last_save_in_peft = config.last_save_in_peft
        self.periodic_save_adapter_only = config.periodic_save_adapter_only
        self.save_training_state = config.save_training_state
        if self.periodic_save_adapter_only and not self.initial_load_path:
            raise ValueError("periodic_save_adapter_only requires a fixed initial_load_path for the frozen base")

    @staticmethod
    def _finalize_peft_directory(checkpoint_id: str, adapter_config: dict, state_dict: dict) -> None:
        os.makedirs(checkpoint_id, exist_ok=True)
        save_file(state_dict, os.path.join(checkpoint_id, "adapter_model.safetensors"))
        with open(os.path.join(checkpoint_id, "adapter_config.json"), "w", encoding="utf-8") as output:
            json.dump(adapter_config, output, indent=2, sort_keys=True)
            output.write("\n")

    def finalize_peft_checkpoint(self, checkpoint_id: str, adapter_config: dict, state_dict: dict) -> None:
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        status: list[str | None] = [None]
        if rank == 0:
            try:
                adapter = self.sd_adapter
                if not isinstance(adapter, DeepSeekPEFTAdapter):
                    raise TypeError("DeepSeekV4 PEFT export requires DeepSeekV4StateDictAdapter")
                peft_state = adapter.to_peft(state_dict)
                peft_state = {name: value.contiguous() for name, value in peft_state.items()}
                self._finalize_peft_directory(checkpoint_id, adapter_config, peft_state)
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"PEFT checkpoint finalization failed: {status[0]}")

    def _verify_load_source(self, checkpoint_id: str) -> None:
        if self.verify_hash_manifest:
            checkpoint_extensions.verify_checkpoint_manifest(checkpoint_id)

    def _load_hf_base(self, model_state, checkpoint_id: str, *, from_quantized: bool) -> None:
        """Load one layer at a time without restoring LoRA or reseeding EMA."""
        assert self.sd_adapter is not None, "HF loading requires a state-dict adapter"
        groups: dict[str, dict[str, Any]] = {}
        for name, value in model_state.state_dict().items():
            if is_lora_parameter(name):
                continue
            parts = name.split(".", 2)
            prefix = ".".join(parts[:2]) if parts[0] in {"layers", "mtp_layers"} else ""
            groups.setdefault(prefix, {})[name] = value
        for prefix in sorted(groups):
            hf_state = self.sd_adapter.to_hf(groups[prefix])
            reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
            dcp.load(hf_state, storage_reader=reader)
            model_state.load_state_dict(self.sd_adapter.from_hf(hf_state))
            del hf_state, reader

    @staticmethod
    def _load_native_states(states, state_dict, checkpoint_id: str) -> None:
        # DCP restores non-model Stateful objects itself. The flattened model
        # needs an explicit load to commit transformed or hook-produced tensors.
        dcp.load(state_dict, checkpoint_id=checkpoint_id)
        if MODEL in states:
            states[MODEL].load_state_dict(state_dict)

    def _validate_native_resume(self, saved_keys: set[str]) -> None:
        if any(key.endswith(".w13_lora_b") for key in saved_keys):
            raise ValueError("This checkpoint uses the unsupported historical w13_lora_b adapter format")

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf: bool, from_quantized: bool) -> None:
        if MODEL not in states:
            # Preserve upstream's non-model-only consumers, including TorchFT.
            return super()._load_checkpoint(states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized)

        self._verify_load_source(checkpoint_id)
        restored_states = states
        if from_hf:
            self._load_hf_base(states[MODEL], checkpoint_id, from_quantized=from_quantized)
            restored_states = {MODEL: states[MODEL]}
        else:
            saved_keys = set(dcp.FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata)
            initial_base = checkpoint_id == self.initial_load_path
            if not initial_base:
                self._validate_native_resume(saved_keys)
            model_state = states[MODEL].state_dict()
            model_buffer_keys = self._model_buffer_keys()
            has_adapters = any(is_lora_parameter(key) for key in saved_keys)
            has_base = all(
                key in saved_keys for key in model_state if not is_lora_parameter(key) and key not in model_buffer_keys
            )
            if initial_base or not has_adapters:
                flat = {
                    key: value
                    for key, value in self._flattened_model_states_sd(states).items()
                    if not is_lora_parameter(key)
                }
            elif has_base:
                flat = self._flattened_model_states_sd(states)
            else:
                if not self.initial_load_path:
                    raise ValueError("Loading an adapter-only DCP requires initial_load_path for the frozen base")
                missing_adapters = {key for key in model_state if is_lora_parameter(key)} - saved_keys
                if missing_adapters:
                    raise ValueError(f"Adapter checkpoint is missing LoRA tensors: {sorted(missing_adapters)}")
                # Check both sources before either can change live weights.
                self._verify_load_source(self.initial_load_path)
                if self.initial_load_in_hf:
                    self._load_hf_base(
                        states[MODEL], self.initial_load_path, from_quantized=self.initial_load_in_hf_quantized
                    )
                else:
                    base_state = {key: value for key, value in model_state.items() if not is_lora_parameter(key)}
                    self._load_native_states({MODEL: states[MODEL]}, base_state, self.initial_load_path)
                restored_states = {
                    key: value
                    for key, value in states.items()
                    if key == MODEL or any(saved == key or saved.startswith(f"{key}.") for saved in saved_keys)
                }
                if restored_states.keys() == {MODEL}:
                    logger.warning(
                        "Loading LoRA weights only: optimizer, scheduler, dataloader "
                        "and training step are not restored."
                    )
                flat = {
                    key: value
                    for key, value in self._flattened_model_states_sd(restored_states).items()
                    if key in restored_states or key in saved_keys
                }
            self._load_native_states(restored_states, flat, checkpoint_id)

        # Match target upstream: reseed only when a model was restored and EMA
        # was not. Base/layer loads above never reseed partially restored weights.
        if EMA in self.states and EMA not in restored_states:
            self.states[EMA].load_state_dict({})

    def _model_buffer_keys(self) -> set[str]:
        return {
            canonical_fqn(name)
            for model in self.states[MODEL].model
            for name, _ in model.named_buffers(remove_duplicate=False)
        }

    def _ensure_peft_exportable(self) -> None:
        unsupported = sorted(
            name
            for model in self.states[MODEL].model
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad and not is_lora_parameter(name)
        )

        if dist.is_available() and dist.is_initialized():
            gathered: list[list[str]] = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, unsupported)
            unsupported = sorted({name for names in gathered for name in names})
        if unsupported:
            preview = ", ".join(unsupported[:8])
            suffix = "..." if len(unsupported) > 8 else ""
            raise ValueError(
                "PEFT LoRA export cannot preserve trainable non-LoRA parameters: "
                f"{preview}{suffix}. Freeze them or save an adapter DCP instead."
            )

    def _flattened_model_states_sd(self, state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
        flattened = super()._flattened_model_states_sd(state_dict)
        if not self.periodic_save_adapter_only or state_dict is not None:
            return flattened
        non_model_keys = {key for key in self.states if key != MODEL} if self.save_training_state else set()
        trainable_keys = {
            canonical_fqn(name)
            for model in self.states[MODEL].model
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        }
        selected = select_adapter_only_state(
            flattened, non_model_keys, model_buffer_keys=self._model_buffer_keys(), model_trainable_keys=trainable_keys
        )
        if not any(is_lora_parameter(key) for key in selected):
            raise ValueError("No LoRA adapter tensors were found for interval checkpointing")
        return selected

    def _save_last_step(self, curr_step: int) -> None:
        if not self.last_save_in_peft:
            super()._save_last_step(curr_step)
            return

        adapter = self.sd_adapter
        if not isinstance(adapter, DeepSeekPEFTAdapter):
            raise TypeError("DeepSeekV4 PEFT export requires DeepSeekV4StateDictAdapter")

        self._ensure_peft_exportable()
        state_dict = self.states[MODEL].state_dict()
        base_model = self.peft_base_model_name_or_path
        if base_model is None and self.initial_load_in_hf:
            base_model = self.initial_load_path
        adapter_config = adapter.peft_adapter_config(base_model_name_or_path=base_model)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        adapter_state = {}
        for name, value in state_dict.items():
            if not is_lora_parameter(name):
                continue
            # All ranks participate in reconstruction, retaining only one full
            # adapter tensor on device at a time. Rank zero owns the CPU export.
            full_value = value.full_tensor() if isinstance(value, DTensor) else value
            if rank == 0:
                adapter_state[name] = full_value.detach().to(device="cpu", dtype=self.export_dtype)
            del full_value
        checkpoint_id = self._create_checkpoint_id(curr_step)
        logger.info("Saving PEFT LoRA adapter checkpoint at %s", checkpoint_id)
        self.finalize_peft_checkpoint(checkpoint_id, adapter_config, adapter_state)
