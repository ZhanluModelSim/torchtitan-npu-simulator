# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.deepseek_v3.model import DeepSeekV3Model


class DeepSeekV3ModelNpu(DeepSeekV3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            original_use_grouped_mm = {}
            for i, layer_cfg in enumerate(self.layers):
                if layer_cfg.moe is not None:
                    original_use_grouped_mm[i] = layer_cfg.moe.experts.use_grouped_mm

            DeepSeekV3Model.Config.update_from_config(
                self, trainer_config=trainer_config, **kwargs
            )

            for i, layer_cfg in enumerate(self.layers):
                if layer_cfg.moe is not None and i in original_use_grouped_mm:
                    layer_cfg.moe.experts.use_grouped_mm = original_use_grouped_mm[i]
