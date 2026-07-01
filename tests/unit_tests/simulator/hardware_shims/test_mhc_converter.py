# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.converters.kernels.mhc_prepost import MHCPostModelConfig, MHCPrePostModelConfig
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.simulator.hardware_shims.mhc_converter import apply_mhc_shims, unapply_mhc_shims
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcHead, SimHcPost, SimHcPre


class _FakeModelSpec:
    name = "deepseek_v4"


def _make_hc_pre() -> HcPre:
    config = HcPre.Config(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6)
    return HcPre(config)


def test_apply_mhc_shims_replaces_converter_target_classes():
    original_pre = MHCPrePostModelConfig.model_converter
    original_post = MHCPostModelConfig.model_converter
    try:
        apply_mhc_shims()
        assert MHCPrePostModelConfig.model_converter is not original_pre
        assert MHCPostModelConfig.model_converter is not original_post
    finally:
        unapply_mhc_shims()
        assert MHCPrePostModelConfig.model_converter is original_pre
        assert MHCPostModelConfig.model_converter is original_post


def test_applied_mhc_pre_converter_replaces_hc_pre_with_sim_hc_pre():
    apply_mhc_shims()
    try:
        model = nn.Sequential()
        model.add_module("hc_pre", _make_hc_pre())
        converter = MHCPrePostModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.hc_pre, SimHcPre)
        assert model.hc_pre.hc_mult == 4
    finally:
        unapply_mhc_shims()


def test_applied_mhc_post_converter_replaces_hc_post_and_hc_head():
    apply_mhc_shims()
    try:
        model = nn.Sequential()
        model.add_module("hc_post", HcPost(HcPost.Config()))
        model.add_module("hc_head", HcHead(HcHead.Config(norm_eps=1e-6, hc_eps=1e-6, hc_mult=4, dim=8)))
        converter = MHCPostModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.hc_post, SimHcPost)
        assert isinstance(model.hc_head, SimHcHead)
    finally:
        unapply_mhc_shims()


def test_unapply_is_idempotent_when_not_applied():
    unapply_mhc_shims()  # must not raise even if apply_mhc_shims was never called
    unapply_mhc_shims()  # calling twice must also not raise
