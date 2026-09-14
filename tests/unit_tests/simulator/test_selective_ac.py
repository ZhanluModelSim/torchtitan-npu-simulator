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
    synthetic_ac_save_patterns,
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


@pytest.mark.parametrize("selection", [None, ["default"], ["attention"], ["full"]])
def test_attention_choices_save_synthetic_attention_ops(selection):
    patterns = synthetic_ac_save_patterns(selection)

    assert "aclnn.npu_sparse_attn_sharedkv" in patterns
    assert "triton_ascend_kernels.chunk_kda" in patterns
    assert "fusion_attention" in patterns


@pytest.mark.parametrize("selection", [["none"], ["mm"], ["gmm", "quant-mm"]])
def test_non_attention_choices_recompute_synthetic_attention_ops(selection):
    assert synthetic_ac_save_patterns(selection) == ()
