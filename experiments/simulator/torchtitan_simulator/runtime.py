# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit runtime activation; imports alone never install simulator patches."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from weakref import WeakValueDictionary

import torch

from torchtitan_simulator import meta_env
from torchtitan_simulator.compat.amp import install_meta_autocast
from torchtitan_simulator.compat.cann_metadata import install_metadata_dispatch
from torchtitan_simulator.compat.optimizer import install_meta_optimizer
from torchtitan_simulator.compat.patches import PatchScope
from torchtitan_simulator.compat.rotary import install_rotary_bridge


@dataclass
class RuntimeSession:
    patches: PatchScope
    capabilities: dict

    def configure_model(self, prepared):
        from torchtitan_simulator.compat.target import install_target_selection

        self.capabilities.update(install_target_selection(self.patches, prepared.simulation.target_npu_device_type))
        if prepared.simulation.enable_quantized_training:
            from torchtitan_simulator.compat.swiglu import install_meta_swiglu_autograd

            self.capabilities["swiglu_meta_autograd"] = install_meta_swiglu_autograd(self.patches)
        if prepared.trainer.model_spec.name == "kimi_k3":
            from torchtitan_simulator.compat.kimi import install_kimi_virtual_fusions

            install_kimi_virtual_fusions(self.patches)
            self.capabilities["kimi_virtual_fusions"] = "dispatcher Meta/autograd; legacy fusion boundaries"
        imports = prepared.trainer.override.imports
        if "torchtitan_npu.override.common.rms_norm.asc" in imports:
            from torchtitan_simulator.compat.norm import install_meta_norm_autograd, install_meta_norm_buffers

            owner = import_module("torchtitan_npu.override.common.rms_norm")
            install_meta_norm_buffers(self.patches, owner.AscRMSNorm)
            self.capabilities["rms_norm_buffer_device"] = "meta"
            self.capabilities["rms_norm_meta_autograd"] = install_meta_norm_autograd(self.patches)
        if "torchtitan_npu.override.common.token_dispatcher.asc" in imports:
            from torchtitan_simulator.compat.permute import install_meta_permute_autograd

            self.capabilities["token_permute_meta_autograd"] = install_meta_permute_autograd(self.patches)
        if any(
            isinstance(entry, str) and entry.startswith("torchtitan_npu.override.common.rope.") for entry in imports
        ):
            owner = import_module("torchtitan_npu.override.common.rope")
            if hasattr(owner, "_INTERLEAVED_CACHE_POOL"):
                # Buffers receive mesh-specific SPMD annotations. Sharing them
                # across captures with different logical worlds is invalid.
                self.patches.replace(owner, "_INTERLEAVED_CACHE_POOL", WeakValueDictionary())
                self.capabilities["rope_cache_scope"] = "capture"
        if any(
            isinstance(entry, str) and entry.startswith("torchtitan_npu.override.deepseek_v4.sparse_attn.asc")
            for entry in imports
        ):
            owner = import_module("torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc")
            self.capabilities["cann_metadata"] = install_metadata_dispatch(self.patches, owner)


@contextmanager
def meta_runtime():
    if meta_env._is_meta_simulation:
        raise RuntimeError("Nested simulator runtime activation is not supported")
    previous_context = dict(meta_env._pp_context)
    previous_fsdp_state = dict(meta_env._fsdp_state)
    previous_comm_layer = meta_env._comm_layer
    previous_metadata_inference = meta_env._in_metadata_inference
    with PatchScope() as scope:
        capabilities = install_rotary_bridge(scope)
        install_meta_autocast(scope)
        install_meta_optimizer(scope)
        previous_multithreading = torch.autograd.is_multithreading_enabled()
        # SPMD contexts carry TLS and must remain on the executing thread.
        torch.autograd.set_multithreading_enabled(False)
        meta_env._is_meta_simulation = True
        try:
            yield RuntimeSession(scope, capabilities)
        finally:
            meta_env._is_meta_simulation = False
            meta_env._pp_context.clear()
            meta_env._pp_context.update(previous_context)
            meta_env._fsdp_state.clear()
            meta_env._fsdp_state.update(previous_fsdp_state)
            meta_env._comm_layer = previous_comm_layer
            meta_env._in_metadata_inference = previous_metadata_inference
            torch.autograd.set_multithreading_enabled(previous_multithreading)
