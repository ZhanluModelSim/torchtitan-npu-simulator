# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torchtitan.distributed.activation_checkpoint as activation_checkpoint

from torchtitan_npu.simulator.selective_ac import (
    _FULL_CHOICES,
    _normalized_selection,
    selective_ac_save_ops_context,
)


def test_none_is_exclusive_selective_ac_save_ops():
    with pytest.raises(ValueError, match="must be used alone"):
        with selective_ac_save_ops_context(["none", "mm"]):
            pass


def test_full_expands_to_all_selectable_save_op_categories():
    assert _normalized_selection(["full"]) == set(_FULL_CHOICES)
    assert _normalized_selection(["full", "gmm"]) == set(_FULL_CHOICES)

    original_context_factory = activation_checkpoint.create_selective_checkpoint_contexts
    with selective_ac_save_ops_context(["full"]):
        save_ops = activation_checkpoint._get_save_ops()
        assert (
            activation_checkpoint.create_selective_checkpoint_contexts
            is original_context_factory
        )

    assert torch.ops.aten.mm.default in save_ops
    assert torch.ops.aten.linear.default in save_ops
    assert torch.ops.aten._grouped_mm.default in save_ops
    assert torch.ops.npu.npu_quant_matmul.default in save_ops


def test_mm_uses_upstream_policy_while_gmm_is_added_to_save_ops():
    original_context_factory = activation_checkpoint.create_selective_checkpoint_contexts

    with selective_ac_save_ops_context(["mm", "gmm"]):
        save_ops = activation_checkpoint._get_save_ops()
        assert torch.ops.aten.mm.default in save_ops
        assert torch.ops.aten._grouped_mm.default in save_ops
        assert activation_checkpoint.create_selective_checkpoint_contexts is original_context_factory
