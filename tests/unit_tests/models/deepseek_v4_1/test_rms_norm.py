# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torchtitan.config import OverrideConfig
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.nn_modules import RMSNorm

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.vision import DeepSeekV41VisionEncoder
from torchtitan_npu.override.common.rms_norm import AscRMSNorm

PLAIN_OVERRIDE = "torchtitan_npu.override.common.rms_norm.asc"


def test_override_reaches_vision_and_preserves_parameter_names():
    cfg = DeepSeekV41VisionEncoder.Config(dim=8, num_layers=2, num_heads=2, inter_dim=16, text_dim=8)
    ref = cfg.build()
    fused_cfg = copy.deepcopy(cfg)
    apply_overrides(OverrideConfig(imports=[PLAIN_OVERRIDE]), fused_cfg)
    fused = fused_cfg.build()
    norms = [m for m in fused.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 5
    assert all(isinstance(m, AscRMSNorm) for m in norms)
    assert ref.state_dict().keys() == fused.state_dict().keys()
    fused.load_state_dict(ref.state_dict(), strict=True)


def test_model_config_uses_common_rms_norm():
    cfg = model_registry("deepseek_v4_1_debugmodel").model
    assert type(cfg.layers[0].attention.q_norm) is RMSNorm.Config
    assert type(cfg.layers[2].attention.compressor.norm) is RMSNorm.Config
    assert type(cfg.layers[2].attention.indexer.k_norm) is RMSNorm.Config
    assert type(cfg.layers[20].attention.compressor.norm) is RMSNorm.Config

    apply_overrides(OverrideConfig(imports=[PLAIN_OVERRIDE]), cfg)
    assert type(cfg.layers[0].attention.q_norm) is AscRMSNorm.Config
    assert type(cfg.layers[2].attention.compressor.norm) is AscRMSNorm.Config
    assert type(cfg.layers[2].attention.indexer.k_norm) is AscRMSNorm.Config
    assert type(cfg.layers[20].attention.compressor.norm) is AscRMSNorm.Config
