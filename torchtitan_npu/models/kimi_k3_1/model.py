# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 decoder reusing K3 blocks, loss protocol and parallel execution."""

from dataclasses import dataclass

from torchtitan_npu.models.kimi_k3.model import KimiK3Model

from .sharding import set_indexer_sharding_config


class KimiK31Model(KimiK3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        def update_from_config(self, *, config, **kwargs) -> None:
            KimiK3Model.Config.update_from_config(self, config=config, **kwargs)
            set_indexer_sharding_config(self)
