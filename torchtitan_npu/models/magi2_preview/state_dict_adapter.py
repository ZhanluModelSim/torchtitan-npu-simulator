# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for MAGI-2-preview HF checkpoint mapping.

The official MAGI-2-preview checkpoint keys are identical to the torchtitan
internal module paths (the reference repo loads with strict=True and zero
renaming), so this adapter implements an explicit identity mapping:
``from_hf`` keeps only the keys expected for the given model config and
``to_hf`` passes every key through unchanged.
Reference: MAGI-2-preview official inference/model/magi2_preview.py (Apache-2.0)
"""

import logging
from typing import Any

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


class Magi2PreviewStateDictAdapter(StateDictAdapter):
    """Adapts official MAGI-2-preview checkpoints to torchtitan Magi2PreviewModel format.

    Identity scheme: the official checkpoint stores every tensor under exactly
    the same key as the internal module path (``pre_adapter.*``,
    ``block.layers.{i}.*``, ``post_adapter.*``), so no renaming is performed.
    ``from_hf`` filters the incoming keys against the expected key set built
    from the model config (dropping anything else, e.g. VAE/text-encoder or
    optimizer leftovers), and ``to_hf`` is a plain pass-through.

    The routed MoE expert tensors (``block.layers.{i}.mlp.moe_mlp.{gate,
    W_gate, W_up, W_down}``) already arrive stacked expert-major with shape
    ``(moe_num_heads * num_experts, ...)``, so no per-expert
    stacking/unstacking is needed. Expert parallelism is deferred; once EP is
    enabled these tensors get sliced along the leading flattened expert dim.
    """

    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.expected_keys = self._build_expected_keys(model_config)

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

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        for hf_key, tensor in hf_state_dict.items():
            if hf_key in self.expected_keys:
                state_dict[hf_key] = tensor
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
            hf_state_dict[key] = value
        return hf_state_dict
