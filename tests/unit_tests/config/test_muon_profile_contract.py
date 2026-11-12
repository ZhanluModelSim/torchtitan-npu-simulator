# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real model recipes select the typed Muon replica-dedup configuration."""

import pytest
from torchtitan.config import ConfigManager

from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon


@pytest.mark.parametrize(
    ("module", "recipe"),
    [
        ("torchtitan_npu_recipes.models.deepseek_v4", "deepseek_v4_debugmodel"),
        ("torchtitan_npu_recipes.models.deepseek_v4_1", "deepseek_v4_1_debugmodel_text"),
    ],
)
@pytest.mark.parametrize("dedup", [False, True])
def test_real_muon_recipe_selects_npu_config_and_propagates_replica_dedup(module, recipe, dedup, tmp_path):
    args = [
        "--module",
        module,
        "--config",
        recipe,
        "--hf_assets_path",
        str(tmp_path),
        "--optim.optimizer.name",
        "Muon",
    ]
    if dedup:
        args.append("--optim.optimizer.muon_enable_hsdp_replica_dedup")

    config = ConfigManager().parse_args(args)

    muons = [optimizer for optimizer in config.optim.optimizer.optimizers if type(optimizer) is DistMuon.Config]
    assert len(muons) == 1
    muon = muons[0]
    assert type(muon) is DistMuon.Config
    assert muon.enable_hsdp_replica_dedup is dedup
    assert config.optim.optimizer._muon_profile.optimizer_config_type is DistMuon.Config
    assert muon.pattern == config.optim.optimizer._muon_profile.muon_pattern
    assert muon.compute_sharding_by_fqn
    assert muon.bucket_configs
