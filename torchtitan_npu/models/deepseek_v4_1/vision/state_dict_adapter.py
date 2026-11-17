# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections.abc import Mapping
from typing import Any

import torch


class DeepSeekV41VisionStateDictAdapter:
    """Explicit adapter for the V4.1 vision namespace.

    Owns the tower and aligner under ``vision.*`` / ``aligner.*``, the marker vectors
    ``image_start`` / ``image_newline`` / ``image_end``, and the per-layer routing bias
    ``layers.{}.ffn.gate.bias_vl``.

    That last one is not a vision parameter -- it lives on the language MoE's router --
    but it only exists when that router is built for a modality, so it belongs to the
    same "this model has a vision half" fact as the tower.  Keeping it here is what lets
    the text half of the composed adapter stay free of vision knowledge: a text-only
    model builds no ``bias_vl``, and needs no rule that would have to name one.

    Text-backbone keys are delegated to the inherited V4 adapter; the V4.1-only
    text-side mappings are registered by the composed adapter above.
    """

    _FROM_HF = {
        "vision.patch_embed.proj.weight": "vision_encoder.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias": "vision_encoder.patch_embed.proj.bias",
        "vision.norm.weight": "vision_encoder.norm.weight",
        "aligner.w1.weight": "vision_encoder.aligner.w1.weight",
        "aligner.w1.bias": "vision_encoder.aligner.w1.bias",
        "aligner.w2.weight": "vision_encoder.aligner.w2.weight",
        "aligner.w2.bias": "vision_encoder.aligner.w2.bias",
        "image_start": "image_marker_embeddings.image_start",
        "image_newline": "image_marker_embeddings.image_newline",
        "image_end": "image_marker_embeddings.image_end",
        "layers.{}.ffn.gate.bias_vl": "layers.{}.moe.router.bias_vl",
    }

    _HF_PREFIXES = ("vision.", "aligner.")
    _HF_EXACT = {"image_start", "image_newline", "image_end"}
    # ``bias_vl`` sits inside the layer namespace the text half otherwise owns, so it is
    # claimed by pattern rather than by prefix -- on both sides of the mapping.
    _HF_VL_BIAS = re.compile(r"^layers\.\d+\.ffn\.gate\.bias_vl$")
    _LOCAL_VL_BIAS = re.compile(r"^layers\.\d+\.moe\.router\.bias_vl$")
    _LOCAL_PREFIXES = ("vision_encoder.", "image_marker_embeddings.")

    def __init__(self, *, expected_shapes: Mapping[str, tuple[int, ...]] | None = None):
        self.expected_shapes = dict(expected_shapes or {})

    @classmethod
    def owns_hf_key(cls, key: str) -> bool:
        return key in cls._HF_EXACT or key.startswith(cls._HF_PREFIXES) or cls._HF_VL_BIAS.match(key) is not None

    @classmethod
    def owns_local_key(cls, key: str) -> bool:
        return key.startswith(cls._LOCAL_PREFIXES) or cls._LOCAL_VL_BIAS.match(key) is not None

    @staticmethod
    def _to_local_key(key: str) -> str:
        if key.startswith("vision."):
            return "vision_encoder." + key[len("vision.") :]
        if key.startswith("aligner."):
            return "vision_encoder." + key
        if key in ("image_start", "image_newline", "image_end"):
            return "image_marker_embeddings." + key
        return key

    @staticmethod
    def _to_hf_key(key: str) -> str:
        if key.startswith("vision_encoder.aligner."):
            return "aligner." + key[len("vision_encoder.") :]
        if key.startswith("vision_encoder."):
            return "vision." + key[len("vision_encoder.") :]
        if key.startswith("image_marker_embeddings."):
            return key[len("image_marker_embeddings.") :]
        return key

    def from_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            local_key = self._FROM_HF.get(key)
            if local_key is None:
                # A layer-scoped key never equals its template, so it is resolved by
                # abstraction and then spelled back with the key's own layer index.
                template = self._FROM_HF.get(self._abstract_key(key))
                if template is not None and template != self._abstract_key(key):
                    local_key = template.format(self._layer_index(key))
            if local_key is None:
                local_key = self._to_local_key(key)
            self._validate(local_key, value)
            result[local_key] = value
        return result

    def to_hf(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        reverse = {local: hf for hf, local in self._FROM_HF.items()}
        result: dict[str, Any] = {}
        for key, value in state_dict.items():
            abstract = self._abstract_key(key)
            hf_key = reverse.get(abstract)
            if hf_key is not None and abstract != key:
                hf_key = hf_key.format(self._layer_index(key))
            elif hf_key is None:
                hf_key = self._to_hf_key(key) if self.owns_local_key(key) else key
            self._validate(key, value)
            result[hf_key] = value
        return result

    @staticmethod
    def _abstract_key(key: str) -> str:
        """``layers.3.moe.router.bias_vl`` -> ``layers.{}.moe.router.bias_vl``.

        A key that is not layer-scoped is returned unchanged, so the abstraction is only
        ever a *different* string for a layer-scoped key -- which is what the callers
        test to decide whether they have one.
        """
        if not key.startswith("layers."):
            return key
        return re.sub(r"^layers\.\d+\.", "layers.{}.", key, count=1)

    @staticmethod
    def _layer_index(key: str) -> int:
        match = re.match(r"^layers\.(\d+)\.", key)
        if match is None:
            raise ValueError(f"layer-scoped key without a layer index: {key!r}")
        return int(match.group(1))

    def _validate(self, key: str, value: Any) -> None:
        expected = self.expected_shapes.get(key)
        if expected is None or not isinstance(value, torch.Tensor):
            return
        if tuple(value.shape) != expected:
            raise ValueError(f"shape mismatch for {key}: expected {expected}, got {tuple(value.shape)}")
