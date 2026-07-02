# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.converters.kernels.npu_smla import NpuSMLAModelConfig
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model, LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.hardware_shims.smla_converter import apply_smla_shims, unapply_smla_shims
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute, SimNpuLiLoss, SimNpuSparseAttention


class _FakeModelSpec:
    name = "deepseek_v4"


def test_apply_smla_shims_replaces_converter_target_class():
    original = NpuSMLAModelConfig.model_converter
    try:
        apply_smla_shims()
        assert NpuSMLAModelConfig.model_converter is not original
    finally:
        unapply_smla_shims()
        assert NpuSMLAModelConfig.model_converter is original


def test_applied_smla_converter_replaces_all_three_submodule_types():
    apply_smla_shims()
    try:
        args = DeepSeekV4Model.Config(n_heads=4, head_dim=8, compress_ratios=(4,), window_size=2, n_layers=1)
        model = nn.Sequential()
        model.add_module("sparse_attn", SparseAttention(SparseAttention.Config(layer_id=0, args=args)))
        model.add_module("li_compute", LiCompute(LiCompute.Config(ratio=4, index_topk=5)))
        model.add_module("li_loss", LiLoss(LiLoss.Config(n_heads=4, softmax_scale=0.1, compress_ratio=4, window_size=2, layer_id=0, n_layers=1)))
        converter = NpuSMLAModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.sparse_attn, SimNpuSparseAttention)
        assert isinstance(model.li_compute, SimNpuLiCompute)
        assert isinstance(model.li_loss, SimNpuLiLoss)
    finally:
        unapply_smla_shims()


def test_unapply_is_idempotent_when_not_applied():
    unapply_smla_shims()  # must not raise even if apply_smla_shims was never called
    unapply_smla_shims()  # calling twice must also not raise
