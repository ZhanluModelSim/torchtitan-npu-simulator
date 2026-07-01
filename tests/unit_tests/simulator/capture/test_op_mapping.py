# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.op_mapping import OP_MAPPING, display_op_label, to_canonical_op_type


def test_known_aten_op_maps_to_canonical_type():
    assert to_canonical_op_type("aten.addmm.default") == "matmul"
    assert to_canonical_op_type("aten.bmm.default") == "bmm"


def test_known_npu_op_maps_to_canonical_type():
    assert to_canonical_op_type("npu.npu_rms_norm.default") == "rms_norm"
    assert to_canonical_op_type("npu.npu_moe_token_permute.default") == "moe_token_permute"
    assert to_canonical_op_type("npu.npu_rotary_mul.default") == "rope"


def test_unknown_op_maps_to_unknown():
    assert to_canonical_op_type("aten.some_brand_new_op.default") == "unknown"


def test_op_mapping_has_no_duplicate_canonical_names_missing():
    # sanity: every value should be a non-empty string
    assert all(isinstance(v, str) and v for v in OP_MAPPING.values())


def test_display_op_label_returns_canonical_type_when_known():
    # A recognized canonical op_type is already human-readable -- shown as-is,
    # real dispatcher name stays available only in annotations.
    assert display_op_label("matmul", {"raw_op_type": "aten.mm.default"}) == "matmul"


def test_display_op_label_falls_back_to_raw_op_type_when_unknown():
    # op_type == "unknown" must never be shown verbatim in visualizations --
    # it must fall back to the real dispatcher name so op identity is never
    # hidden from graph review (see dispatch_capture.py's raw_op_type).
    assert display_op_label("unknown", {"raw_op_type": "aten.embedding.default"}) == "aten.embedding.default"


def test_display_op_label_falls_back_to_literal_unknown_when_raw_op_type_missing():
    assert display_op_label("unknown", {}) == "unknown"
