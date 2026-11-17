# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 checkpoint mapping adds indexer weights to the common K3 mapping."""

from torchtitan_npu.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


class KimiK31StateDictAdapter(KimiK3StateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None = None):
        super().__init__(model_config, hf_assets_path)
        for name in ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"):
            self.from_hf_layer_map[f"language_model.model.layers.{{}}.self_attn.indexer.{name}"] = (
                f"layers.{{}}.attention.indexer.{name}"
            )
