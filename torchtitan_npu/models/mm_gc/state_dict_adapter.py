# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict adapter for mm_gc.

No public checkpoint exists for this model yet, so the adapter is a native
identity passthrough: the native state-dict schema is the single source of
truth for DCP save/load round-trips. HF conversion must be added together
with the external weight mapping and must preserve the w1/w2/w3 expert
layout documented in MODEL_CONTRACT.md.
"""

from typing import Any

from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter


class MMGcStateDictAdapter(BaseStateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.fqn_to_index_mapping: dict[Any, int] | None = None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        return hf_state_dict

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        raise NotImplementedError(
            "mm_gc has no HF checkpoint mapping yet; native DCP save/load is "
            "the only supported checkpoint path"
        )
