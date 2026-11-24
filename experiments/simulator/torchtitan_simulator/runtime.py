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
from torchtitan_simulator.compat.fsdp import validate_fsdp_capture
from torchtitan_simulator.compat.loss import install_chunked_loss_policy_restore
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
        if any(group.optimizer_name == "DistMuon" for group in prepared.trainer.optimizer.param_groups):
            from torchtitan_simulator.compat.muon import install_meta_muon

            self.capabilities.update(
                install_meta_muon(self.patches, bucket_count=prepared.simulation.muon_bucket_count)
            )
        if prepared.simulation.enable_quantized_training:
            from torchtitan_simulator.compat.swiglu import install_meta_swiglu_autograd

            self.capabilities["swiglu_meta_autograd"] = install_meta_swiglu_autograd(self.patches)
        if prepared.trainer.model_spec.name in {"kimi_k3", "kimi_k3_1"}:
            from torchtitan_simulator.compat.kimi import install_kimi_virtual_fusions

            install_kimi_virtual_fusions(self.patches)
            self.capabilities["kimi_virtual_fusions"] = "dispatcher Meta/autograd; legacy fusion boundaries"
        if prepared.trainer.model_spec.name == "kimi_k3_1":
            from torchtitan_simulator.compat.kimi_dsa import install_kimi_dsa_virtual_fusions

            install_kimi_dsa_virtual_fusions(self.patches)
            self.capabilities["kimi_dsa_virtual_fusions"] = "selected-key Meta/autograd; explicit saved output/LSE"
        if prepared.trainer.model_spec.name == "system3":
            from torchtitan_simulator.compat.system3_engram import install_system3_engram_meta_support
            from torchtitan_simulator.compat.system3_fusions import install_system3_virtual_fusions

            install_system3_virtual_fusions(self.patches)
            self.capabilities["system3_virtual_fusions"] = (
                "swiglu_clamp, gdn2_chunked, isa_indexer and isa_sparse_attention dispatcher Meta/autograd boundaries"
            )
            self.capabilities["system3_engram_host_table"] = install_system3_engram_meta_support(self.patches)
        if prepared.trainer.model_spec.name == "qwen3_5":
            from torchtitan_simulator.compat.qwen3_5 import install_qwen3_5_meta_support

            install_qwen3_5_meta_support(self.patches)
            self.capabilities["qwen3_5_gdn_meta_bridge"] = (
                "dispatcher Meta/autograd; Triton GDN fusion boundary and text input adapter"
            )
        if prepared.trainer.model_spec.name == "multi_head_moe" and prepared.simulation.enable_sla2_fusion:
            from torchtitan_simulator.compat.multi_head_moe import install_multi_head_moe_meta_support

            install_multi_head_moe_meta_support(self.patches)
            self.capabilities["multi_head_moe_sla2_virtual_fusions"] = (
                "dispatcher Meta/autograd; opt-in sparse and linear SLA2 branch fusion boundaries"
            )
        if prepared.trainer.model_spec.name == "deepseek_v4_1":
            from torchtitan_simulator.compat.deepseek_v4_1 import (
                install_deepseek_v4_1_input_adapter,
                install_deepseek_v4_1_meta_support,
            )

            install_deepseek_v4_1_meta_support(self.patches)
            install_deepseek_v4_1_input_adapter(self.patches)
            self.capabilities["deepseek_v4_1_engram_meta_bridge"] = (
                "Meta init for the host-offload Engram table; real residency stays CPU"
            )
            self.capabilities["deepseek_v4_1_metadata_map"] = (
                "build_attention_masks adapter producing the per-backend AttentionMetadataMap"
            )
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
        if "torchtitan_npu.override.common.fusion_attention.asc" in imports:
            from torchtitan_simulator.compat.attention import install_npu_fusion_attention_autograd

            self.capabilities["npu_fusion_attention_meta_autograd"] = install_npu_fusion_attention_autograd(
                self.patches
            )
        if "torchtitan_npu.override.common.varlen_attention.asc" in imports:
            from torchtitan_simulator.compat.attention import install_meta_fusion_attention_autograd

            self.capabilities["fusion_attention_meta_autograd"] = install_meta_fusion_attention_autograd(self.patches)
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
    fsdp_compatibility = validate_fsdp_capture()
    # Static declarations for simulator-supported main-package custom wrappers.
    import_module("torchtitan_simulator.compat.moe_tensor_semantics")
    previous_context = dict(meta_env._pp_context)
    previous_fsdp_state = dict(meta_env._fsdp_state)
    previous_comm_layer = meta_env._comm_layer
    previous_metadata_inference = meta_env._in_metadata_inference
    with PatchScope() as scope:
        capabilities = install_rotary_bridge(scope)
        capabilities["fsdp_capture_compatibility"] = fsdp_compatibility
        install_meta_autocast(scope)
        install_meta_optimizer(scope)
        install_chunked_loss_policy_restore(scope)
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
