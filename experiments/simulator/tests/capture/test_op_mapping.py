# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_simulator.capture.op_mapping import (
    OP_MAPPING,
    is_metadata_view_op,
    to_canonical_op_type,
)


def test_known_aten_op_maps_to_canonical_type():
    assert to_canonical_op_type("aten.addmm.default") == "matmul"
    assert to_canonical_op_type("aten.bmm.default") == "bmm"


def test_known_npu_op_maps_to_canonical_type():
    assert to_canonical_op_type("npu.npu_rms_norm.default") == "rms_norm"
    assert to_canonical_op_type("npu.npu_rms_norm_backward.default") == "rms_norm_backward"
    assert to_canonical_op_type("npu.npu_moe_token_permute.default") == "moe_token_permute"
    assert to_canonical_op_type("npu.npu_moe_token_permute_grad.default") == "moe_token_permute"
    assert to_canonical_op_type("npu.npu_rotary_mul.default") == "rope"
    assert to_canonical_op_type("npu.npu_swiglu_backward.default") == "swiglu"
    assert to_canonical_op_type("npu.npu_quant_matmul.default") == "matmul"
    assert to_canonical_op_type("npu.npu_grouped_matmul.default") == "grouped_mm"
    assert to_canonical_op_type("aten._grouped_mm.default") == "grouped_mm"


def test_unknown_op_maps_to_unknown():
    assert to_canonical_op_type("aten.some_brand_new_op.default") == "unknown"


def test_metadata_view_ops_are_classified_without_hiding_materializing_ops():
    assert to_canonical_op_type("aten.t.default") == "transpose"
    assert to_canonical_op_type("aten.slice.Tensor") == "metadata_view"
    assert is_metadata_view_op("aten.view.default")
    assert is_metadata_view_op("aten.t.default")
    assert is_metadata_view_op("aten.split_with_sizes.default")
    assert not is_metadata_view_op("aten.cat.default")
    assert not is_metadata_view_op("aten.clone.default")
    assert not is_metadata_view_op("aten.contiguous.default")


def test_op_mapping_has_no_duplicate_canonical_names_missing():
    # sanity: every value should be a non-empty string
    assert all(isinstance(v, str) and v for v in OP_MAPPING.values())
