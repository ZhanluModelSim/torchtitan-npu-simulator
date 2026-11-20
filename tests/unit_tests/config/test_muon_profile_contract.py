# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real model recipes select the typed Muon replica-dedup configuration."""

import importlib
import sys
import types

import pytest
from torchtitan.components.optim import AdamW
from torchtitan.config.loader import ConfigLoader

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu_recipes.models.deepseek_v4 import _dsv4_muon_profile
from torchtitan_npu_recipes.models.deepseek_v4_1 import _v41_muon_profile


@pytest.mark.parametrize(
    ("module", "recipe"),
    [
        ("torchtitan_npu_recipes.tests.models.deepseek_v4", "deepseek_v4_debugmodel"),
        ("torchtitan_npu_recipes.models.deepseek_v4_1", "deepseek_v4_1_debugmodel_text"),
    ],
)
@pytest.mark.parametrize("dedup", [False, True])
def test_real_muon_recipe_selects_npu_config_and_propagates_replica_dedup(module, recipe, dedup, monkeypatch, tmp_path):
    def configured_recipe():
        config = getattr(importlib.import_module(module), recipe)()
        config.hf_assets_path = str(tmp_path)
        profile = (_v41_muon_profile if "v4_1" in module else _dsv4_muon_profile)(config.model)
        config.optim.optimizer = OptimizerConfig(
            optimizers=[
                DistMuon.Config(
                    pattern=profile.muon_pattern,
                    compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
                    bucket_configs=profile.bucket_configs,
                    lr=1e-5,
                    enable_hsdp_replica_dedup=dedup,
                ),
                AdamW.Config(pattern=r".*"),
            ]
        )
        return config

    registry = types.ModuleType("_optimizer_recipe")
    registry.recipe = configured_recipe
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    config = ConfigLoader().load(["--module", registry.__name__, "--config", "recipe"])

    muons = [optimizer for optimizer in config.optim.optimizer.optimizers if type(optimizer) is DistMuon.Config]
    assert len(muons) == 1
    muon = muons[0]
    assert type(muon) is DistMuon.Config
    assert muon.enable_hsdp_replica_dedup is dedup
    assert muon.compute_sharding_by_fqn
    assert muon.bucket_configs
