# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the architecture-boundary ratchet."""

from pathlib import Path

import pytest

from scripts.check_architecture_boundaries import check

pytestmark = pytest.mark.tooling


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_baseline(root: Path, content: str = "{}") -> Path:
    baseline = root / ".ci/architecture_import_baseline.json"
    _write(root, ".ci/architecture_import_baseline.json", content)
    return baseline


def test_allows_registered_legacy_import_but_rejects_new_one(tmp_path):
    _write(tmp_path, "torchtitan_npu/extensions/legacy.py", "from torchtitan_npu.models import legacy\n")
    _write(tmp_path, "torchtitan_npu/extensions/new.py", "from torchtitan_npu.models import new\n")
    baseline = _write_baseline(
        tmp_path,
        """{
  "import": [
    {
      "source": "torchtitan_npu/extensions/legacy.py",
      "target_module": "torchtitan_npu.models"
    }
  ]
}""",
    )

    assert check(tmp_path, baseline) == [
        "new forbidden import: torchtitan_npu/extensions/new.py -> torchtitan_npu.models"
    ]


def test_rejects_patch_to_extension_import(tmp_path):
    _write(
        tmp_path,
        "torchtitan_npu/patches/torchtitan/example.py",
        "# Upstream-PR: https://example.invalid/1\nfrom torchtitan_npu.extensions import example\n",
    )
    baseline = _write_baseline(tmp_path)

    assert check(tmp_path, baseline) == [
        "new forbidden import: torchtitan_npu/patches/torchtitan/example.py -> torchtitan_npu.extensions"
    ]


def test_rejects_new_patch_without_header_metadata(tmp_path):
    _write(tmp_path, "torchtitan_npu/patches/torchtitan/example.py", "VALUE = 1\n")
    baseline = _write_baseline(tmp_path)

    assert check(tmp_path, baseline) == [
        "new patch missing upstream metadata: torchtitan_npu/patches/torchtitan/example.py"
    ]


def test_rejects_activation_order_change(tmp_path):
    _write(
        tmp_path,
        "torchtitan_npu/extensions/__init__.py",
        "from .torchtitan import second\nfrom .torchtitan import first\n",
    )
    baseline = _write_baseline(
        tmp_path,
        """{
  "activation_order": {
    "torchtitan_npu/extensions/__init__.py": [".torchtitan.first", ".torchtitan.second"]
  }
}""",
    )

    assert check(tmp_path, baseline) == [
        "activation order changed: torchtitan_npu/extensions/__init__.py; "
        "expected ['.torchtitan.first', '.torchtitan.second'], "
        "got ['.torchtitan.second', '.torchtitan.first']"
    ]
