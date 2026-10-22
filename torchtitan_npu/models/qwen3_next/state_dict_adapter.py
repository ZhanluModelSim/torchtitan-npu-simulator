# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict adapter for Qwen3-Next (v1 scope).

The framework state dict uses the native module FQNs directly (identity
passthrough for DCP save/load). HF weight conversion is out of the v1
declared support matrix and fails fast with an explicit message, per the
onboarding rules (no silent mis-mapped weights).
"""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter


class Qwen3NextStateDictAdapter(BaseStateDictAdapter):
    """Identity DCP schema; HF conversion is explicitly unsupported in v1."""

    fqn_to_index_mapping: dict[Any, int] | None = None
    hf_assets_path: str | None = None

    def __init__(self, model_config: Any, hf_assets_path: str | None = None):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "Qwen3-Next HF weight export is not supported in the v1 "
            "simulator scope; the framework state dict (native FQNs) is "
            "the only declared checkpoint schema."
        )

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "Qwen3-Next HF weight loading is not supported in the v1 "
            "simulator scope; the framework state dict (native FQNs) is "
            "the only declared checkpoint schema."
        )

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        raise NotImplementedError(
            "Qwen3-Next HF storage reader is not supported in the v1 "
            "simulator scope."
        )
