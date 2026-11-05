# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint conversion between the official MindSpeed glm5_next HF layout
and the torchtitan ``state_dict`` naming.

Official (MindSpeed) naming -> torchtitan naming:
- ``model.language_model.embed_tokens.weight``      -> tok_embeddings.weight
- ``model.language_model.layers.{i}.input_layernorm`` -> layers.{i}.attention_norm
- ``model.language_model.layers.{i}.post_attention_layernorm`` -> layers.{i}.ffn_norm
- ``model.language_model.layers.{i}.self_attn.*``   -> layers.{i}.attention.*
  (KDA: qkv_proj/conv1d/forget_gate.*/b_proj/g_a_proj/g_b_proj/o_norm/o_proj;
  DSA: q_a_proj/q_a_layernorm/q_b_proj/kv_a_proj_with_mqa/kv_a_layernorm/
  kv_b_proj/o_proj/indexer.*)
- ``model.language_model.layers.{i}.mlp.{gate_up_proj,down_proj}`` (dense)
  -> ``layers.{i}.feed_forward.*``
- ``model.language_model.layers.{i}.mlp.gate.{weight,e_score_correction_bias}``
  -> ``layers.{i}.moe.router.gate.weight`` / ``layers.{i}.moe.router.e_score_correction_bias``
- ``model.language_model.layers.{i}.mlp.experts.gate_up_proj`` [E, 2I, H]
  -> ``layers.{i}.moe.routed_experts.inner_experts.{w1_EFD, w3_EFD}`` (gate/up split)
- ``model.language_model.layers.{i}.mlp.experts.down_proj`` [E, H, I]
  -> ``layers.{i}.moe.routed_experts.inner_experts.w2_EDF
- ``model.language_model.layers.{i}.mlp.shared_experts.*`` -> layers.{i}.moe.shared_experts.*
- ``model.language_model.layers.{i}.{attn_hc,ffn_hc}.{fn,base,scale}``
                                                      -> same names
- ``model.language_model.norm.weight``                -> norm.weight
- ``lm_head.weight``                                  -> lm_head.weight
- ``model.visual.*``                                  -> vision_encoder.*

Ignored: MTP layers (``model.language_model.layers.{45,46,47}.*`` for the
full checkpoint), ``shared_head.head.weight`` (MTP head).
``tie_word_embeddings`` is False in the deployed config: lm_head has its own
weight.
"""

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

_MTP_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(4[5-9]|[5-9]\d)\.")


class Glm5NextStateDictAdapter(StateDictAdapter):
    def __init__(self, model: torch.nn.Module, base_model_prefix: str | None = None):
        self.model = model
        self.base_model_prefix = base_model_prefix

    @staticmethod
    def to_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
        """torchtitan naming -> official MindSpeed/HF naming."""
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith(("tok_embeddings.", "layers.", "norm.")):
                new_key = "model.language_model." + key
            elif key.startswith("vision_encoder."):
                new_key = "model." + key
            new_key = new_key.replace(".attention.", ".self_attn.", 1) if ".layers." in new_key else new_key
            new_key = new_key.replace(".attention_norm.", ".input_layernorm.", 1)
            new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.", 1)
            new_key = new_key.replace(".feed_forward.", ".mlp.", 1)
            if ".moe.router.gate." in new_key:
                new_key = new_key.replace(".moe.router.gate.", ".mlp.gate.", 1)
            elif ".moe.router." in new_key:
                new_key = new_key.replace(".moe.router.", ".mlp.gate.", 1)
            if ".moe.routed_experts.inner_experts." in new_key:
                # w1/w3 re-fuse into gate_up_proj [E, 2I, H]; w2 -> down_proj.
                new_key = new_key.replace(".moe.routed_experts.inner_experts.", ".mlp.experts.", 1)
                if new_key.endswith(".w1_EFD"):
                    new_key = new_key[: -len(".w1_EFD")] + ".gate_up_proj"
                elif new_key.endswith(".w3_EFD"):
                    new_key = new_key[: -len(".w3_EFD")] + ".gate_up_proj"
                elif new_key.endswith(".w2_EDF"):
                    new_key = new_key[: -len(".w2_EDF")] + ".down_proj"
            new_key = new_key.replace(".moe.shared_experts.", ".mlp.shared_experts.", 1)
            hf_state_dict[new_key] = value
        return hf_state_dict

    @staticmethod
    def from_hf(state_dict: dict[str, Any]) -> dict[str, Any]:
        """Official MindSpeed/HF naming -> torchtitan naming.

        Drops the MTP layers and the MTP shared head. Expert fused weights
        are split into the torchtitan grouped-GEMM layout; everything else is
        renamed per the table above. Values are passed through (device/dtype
        normalization is the caller's job)."""
        titan_state_dict = {}
        for key, value in state_dict.items():
            if _MTP_LAYER_RE.match(key) or key.startswith("shared_head."):
                continue
            if key.startswith("model.visual."):
                new_key = "vision_encoder." + key[len("model.visual.") :]
            elif key.startswith("model.language_model."):
                new_key = key[len("model.language_model.") :]
            else:
                new_key = key
            new_key = new_key.replace(".self_attn.", ".attention.", 1)
            new_key = new_key.replace(".input_layernorm.", ".attention_norm.", 1)
            new_key = new_key.replace(".post_attention_layernorm.", ".ffn_norm.", 1)
            if ".mlp.experts.gate_up_proj" in new_key:
                prefix = new_key.rsplit(".gate_up_proj", 1)[0]
                fused = value
                intermediate = fused.shape[1] // 2
                titan_state_dict[prefix + ".w1_EFD"] = fused[:, :intermediate, :]
                titan_state_dict[prefix + ".w3_EFD"] = fused[:, intermediate:, :]
                continue
            if ".mlp.experts.down_proj" in new_key:
                new_key = new_key.replace(".mlp.experts.down_proj", ".moe.routed_experts.inner_experts.w2_EDF", 1)
                titan_state_dict[new_key] = value
                continue
            if ".mlp.gate." in new_key:
                if new_key.endswith(".mlp.gate.weight"):
                    new_key = new_key[: -len(".mlp.gate.weight")] + ".moe.router.gate.weight"
                else:
                    new_key = new_key.replace(".mlp.gate.", ".moe.router.", 1)
            new_key = new_key.replace(".mlp.shared_experts.", ".moe.shared_experts.", 1)
            if ".mlp." in new_key:
                new_key = new_key.replace(".mlp.", ".feed_forward.", 1)
            titan_state_dict[new_key] = value
        return titan_state_dict

    def get_state_dict(self, names: list[str] | None = None) -> dict[str, Any]:
        sd = self.model.state_dict()
        return self.to_hf(sd)

    def load_from_full_model_state_dict(
        self,
        full_sd: dict[str, Any],
        device_type: str,
        strict: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        sd = self.from_hf(full_sd)
        model_sd = self.model.state_dict()
        missing = [k for k in model_sd if k not in sd]
        unexpected = [k for k in sd if k not in model_sd]
        if strict and (missing or unexpected):
            raise KeyError(f"glm5_next checkpoint mismatch; missing={missing[:8]} unexpected={unexpected[:8]}")
        for key in model_sd:
            if key not in sd:
                continue
            value = sd[key]
            param = model_sd[key]
            if dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype)
            if isinstance(param, DTensor):
                param.to_local().copy_(value)
            else:
                param.copy_(value)

    def load_from_hf(self, full_sd: dict[str, Any], device_type: str) -> None:
        self.load_from_full_model_state_dict(full_sd, device_type)
