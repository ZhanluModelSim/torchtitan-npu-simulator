# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for MAGI-2-preview HF checkpoint mapping.

The official MAGI-2-preview checkpoint keys are identical to the torchtitan
internal module paths (the reference repo loads with strict=True and zero
renaming), so this adapter keeps every key name unchanged. ``from_hf`` keeps
only the keys expected for the given model config, and ``to_hf`` emits every
expected key.

The single layout rule (the only shape change the adapter performs) is for
the multi-expert grouped linear weights (see ``grouped_linear.py``): the
official checkpoint stores them fused expert-major as
``(num_experts * out_features, in_features)``, while the model stores them
with a per-expert leading dim ``(num_experts, out_features, in_features)``.
``from_hf`` reshapes the 2D official tensor to the 3D internal layout and
``to_hf`` reshapes back, bijectively. ``linear_qkv`` additionally reorders
its out dim between the official section-major ``[q | k | v]`` layout and the
internal head-major ``[head_i(q, k, v)]`` layout so its TP head shard is a
single honest placement (see ``grouped_linear.py``); that reorder is part of
the same bijective rule. Single-expert weights (``num_experts == 1``) are 2D
in both formats and pass through unchanged, as do all non-grouped tensors.

Reference: MAGI-2-preview official inference/model/magi2_preview.py (Apache-2.0)
"""

import logging
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import StateDictAdapter

logger = logging.getLogger(__name__)

_PRE_ADAPTER_KEYS = (
    "video_embedder.weight",
    "video_embedder.bias",
    "text_embedder.weight",
    "text_embedder.bias",
    "audio_embedder.weight",
    "audio_embedder.bias",
    "rope.bands",
)

_ATTENTION_KEYS = (
    "pre_norm.weight",
    "linear_g.weight",
    "linear_qkv.weight",
    "linear_proj.weight",
    "sinks",
    "q_norm.weight",
    "k_norm.weight",
)

_DENSE_MLP_KEYS = (
    "pre_norm.weight",
    "up_gate_proj.weight",
    "down_proj.weight",
)

_MOE_MLP_KEYS = (
    "pre_norm.weight",
    "split_linear.weight",
    "merge_linear.weight",
    "moe_mlp.gate",
    "moe_mlp.W_gate",
    "moe_mlp.W_up",
    "moe_mlp.W_down",
    "moe_mlp.router.expert_bias",
    "moe_mlp.router.expert_bias_ema",
    "shared_expert_fc1.weight",
    "shared_expert_fc2.weight",
    "modality_specific_shared_expert_fc1.weight",
    "modality_specific_shared_expert_fc2.weight",
)

# 6 alphas + 4 vector biases + 2 matrix biases + 2 fused phis + MHC norm.
_MHC_KEYS = (
    "mhc_alpha_pre_attn",
    "mhc_alpha_post_attn",
    "mhc_alpha_res_attn",
    "mhc_alpha_pre_mlp",
    "mhc_alpha_post_mlp",
    "mhc_alpha_res_mlp",
    "mhc_bias_pre_attn",
    "mhc_bias_post_attn",
    "mhc_bias_pre_mlp",
    "mhc_bias_post_mlp",
    "mhc_bias_res_attn",
    "mhc_bias_res_mlp",
    "mhc_phi_fused_attn",
    "mhc_phi_fused_mlp",
    "mhc_norm.weight",
)

_POST_ADAPTER_KEYS = (
    "final_norm_video.weight",
    "final_norm_audio.weight",
    "final_linear_video.weight",
    "final_linear_audio.weight",
)

# GroupedLinear weight suffixes subject to the 2D <-> 3D layout rule.
_GROUPED_SUFFIXES = (
    "attention.linear_g.weight",
    "attention.linear_qkv.weight",
    "attention.linear_proj.weight",
    "mlp.up_gate_proj.weight",
    "mlp.down_proj.weight",
    "mlp.split_linear.weight",
    "mlp.merge_linear.weight",
    "mlp.shared_expert_fc1.weight",
    "mlp.shared_expert_fc2.weight",
    "mlp.modality_specific_shared_expert_fc1.weight",
    "mlp.modality_specific_shared_expert_fc2.weight",
)

# GroupedLinear weights that always have a single expert (no layout change).
_SINGLE_EXPERT_SUFFIXES = (
    "mlp.split_linear.weight",
    "mlp.merge_linear.weight",
    "mlp.shared_expert_fc1.weight",
    "mlp.shared_expert_fc2.weight",
)

# GroupedLinear weights that always have num_modality (3) experts, even on
# MoE layers (whose attention is single-expert).
_MOE_MODEXPERT_SUFFIXES = (
    "mlp.modality_specific_shared_expert_fc1.weight",
    "mlp.modality_specific_shared_expert_fc2.weight",
)


def _unwrap(tensor: Any) -> torch.Tensor:
    """Return a plain tensor, gathering a DTensor to its full form first."""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class Magi2PreviewStateDictAdapter(StateDictAdapter):
    """Adapts official MAGI-2-preview checkpoints to torchtitan Magi2PreviewModel format.

    Identity scheme for key names: the official checkpoint stores every tensor
    under exactly the same key as the internal module path (``pre_adapter.*``,
    ``block.layers.{i}.*``, ``post_adapter.*``), so no renaming is performed.
    ``from_hf`` filters the incoming keys against the expected key set built
    from the model config (dropping anything else), and ``to_hf`` emits every
    expected key.

    Layout rule: multi-expert grouped linear weights are converted between the
    official 2D expert-major ``(E * out, in)`` shape and the internal 3D
    ``(E, out, in)`` per-expert layout (plus the head-major ``linear_qkv``
    out-dim reorder); see the module docstring. The routed MoE expert tensors
    (``block.layers.{i}.mlp.moe_mlp.{gate, W_gate, W_up, W_down}``) already
    arrive stacked expert-major with shape ``(moe_num_heads * num_experts,
    ...)`` and pass through unchanged.
    """

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.expected_keys = self._build_expected_keys(model_config)
        self._mm_layers = set(model_config.mm_layers)
        self._moe_layers = set(model_config.moe_layers)
        self._num_heads = model_config.hidden_size // model_config.head_dim
        self._head_dim = model_config.head_dim

    @staticmethod
    def _build_expected_keys(model_config) -> set[str]:
        keys = {f"pre_adapter.{name}" for name in _PRE_ADAPTER_KEYS}
        moe_layers = set(model_config.moe_layers)
        for layer_id in range(model_config.num_layers):
            prefix = f"block.layers.{layer_id}"
            keys.update(f"{prefix}.attention.{name}" for name in _ATTENTION_KEYS)
            mlp_keys = _MOE_MLP_KEYS if layer_id in moe_layers else _DENSE_MLP_KEYS
            keys.update(f"{prefix}.mlp.{name}" for name in mlp_keys)
            keys.update(f"{prefix}.{name}" for name in _MHC_KEYS)
        keys.update(f"post_adapter.{name}" for name in _POST_ADAPTER_KEYS)
        return keys

    def _grouped_num_experts(self, key: str) -> int:
        """Number of modality experts of the GroupedLinear named by ``key``.

        Returns 1 for weights that are always single-expert (or not grouped),
        in which case the layout rule does not apply.
        """
        parts = key.split(".")
        if len(parts) < 5 or parts[0] != "block" or parts[1] != "layers":
            return 1
        try:
            layer_id = int(parts[2])
        except ValueError:
            return 1
        suffix = ".".join(parts[3:])
        if suffix in _SINGLE_EXPERT_SUFFIXES:
            return 1
        if suffix in _MOE_MODEXPERT_SUFFIXES:
            return 3
        if suffix not in _GROUPED_SUFFIXES:
            return 1
        return 3 if layer_id in self._mm_layers else 1

    def _is_qkv(self, key: str) -> bool:
        return key.endswith("attention.linear_qkv.weight")

    def _hf_to_internal(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape an official 2D grouped weight to the internal 3D layout."""
        num_experts = self._grouped_num_experts(key)
        if num_experts <= 1:
            return tensor
        if tensor.ndim != 2:
            return tensor  # already in the internal layout
        in_features = tensor.shape[1]
        if self._is_qkv(key):
            head_dim = self._head_dim
            num_heads = self._num_heads
            return (
                tensor.view(num_experts, 3, num_heads, head_dim, in_features)
                .permute(0, 2, 1, 3, 4)
                .reshape(num_experts, num_heads * 3 * head_dim, in_features)
            )
        out_features = tensor.shape[0] // num_experts
        return tensor.reshape(num_experts, out_features, in_features)

    def _internal_to_hf(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape an internal 3D grouped weight back to the official 2D form."""
        num_experts = self._grouped_num_experts(key)
        if num_experts <= 1:
            return tensor
        if tensor.ndim != 3:
            return tensor  # already in the official layout
        tensor = _unwrap(tensor)
        out_features = tensor.shape[1]
        in_features = tensor.shape[2]
        if self._is_qkv(key):
            head_dim = self._head_dim
            num_heads = self._num_heads
            return (
                tensor.view(num_experts, num_heads, 3, head_dim, in_features)
                .permute(0, 2, 1, 3, 4)
                .reshape(num_experts * 3 * num_heads * head_dim, in_features)
            )
        return tensor.reshape(num_experts * out_features, in_features)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        for hf_key, tensor in hf_state_dict.items():
            if hf_key in self.expected_keys:
                state_dict[hf_key] = self._hf_to_internal(hf_key, tensor)
            else:
                logger.debug("from_hf: skipping unexpected key: %s", hf_key)

        missing_keys = self.expected_keys - state_dict.keys()
        if missing_keys:
            logger.warning(
                "from_hf: %d expected keys missing from the HF checkpoint, "
                "e.g. %s (loading will fail unless they are re-initialized)",
                len(missing_keys),
                sorted(missing_keys)[:5],
            )
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}
        for key, value in state_dict.items():
            if key not in self.expected_keys:
                logger.debug("to_hf: passing through unexpected key: %s", key)
            hf_state_dict[key] = self._internal_to_hf(key, _unwrap(value))
        return hf_state_dict
