# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from pathlib import Path


def test_qwen_gdn_uses_the_override_refactor_kernel():
    root = Path(__file__).resolve().parents[4]
    gdn = root / "torchtitan_npu/ops/triton/gdn"
    override = root / "torchtitan_npu/override/qwen3_5/gated_delta.py"

    assert (gdn / "gated_delta.py").is_file()
    assert (gdn / "__init__.py").is_file()
    text = override.read_text(encoding="utf-8")
    assert "torchtitan_npu.ops.triton.gdn" in text
    assert "torchtitan_npu.ops.triton.gated_delta_rule" not in text


def test_qwen_gdn_legacy_vendor_tree_is_not_tracked():
    root = Path(__file__).resolve().parents[4]
    legacy = root / "torchtitan_npu/ops/triton/gated_delta_rule"
    assert not any(legacy.rglob("*.py"))
