# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import annotations

import os
import sys
import tomllib
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

EXPECTED_TORCHVISION = "0.29.0+cpu"


def test_multimodal_dependency_and_package_discovery_are_explicit():
    root = Path(__file__).resolve().parents[4]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = project["project"]["optional-dependencies"]["multimodal"]
    assert f"torchvision=={EXPECTED_TORCHVISION}" in dependencies
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    torchvision_requirement = f"torchvision=={EXPECTED_TORCHVISION.removesuffix('+cpu')}"
    assert f"{torchvision_requirement}\n" in requirements

    # The UT runner installs requirements-dev.txt directly, so the runtime
    # dependency must be declared there as well as in the base requirements.
    package_find = project.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find")
    assert package_find["include"] == ["torchtitan_npu*"]


def test_qwen3_5_uses_the_merged_gdn_override():
    root = Path(__file__).resolve().parents[4]
    override = (root / "torchtitan_npu/override/qwen3_5/gated_delta.py").read_text(encoding="utf-8")
    assert "torchtitan_npu.ops.triton.gdn" in override
    assert "gated_delta_rule" not in override.split("torchtitan_npu.ops.triton.gdn", 1)[0]
    assert (root / "torchtitan_npu/ops/triton/gdn/gated_delta.py").is_file()
    old_ops = root / "torchtitan_npu/ops/triton/gated_delta_rule"
    assert not any(old_ops.rglob("*.py"))


def test_qwen3_5_launcher_selects_the_current_gdn_override():
    script = (Path(__file__).resolve().parents[4] / "examples" / "qwen3_5" / "qwen3_5_debugmodel_1p_cpt_512.sh").read_text(encoding="utf-8")
    assert "bash scripts/run_train.sh" in script
    assert "torchtitan_npu.override.qwen3_5.gated_delta.npu" in script
    assert "override.common.attention" not in script
    assert "triton_gated_delta_override" not in script


def test_torchvision_native_image_ops_and_v2_transforms_are_available():
    import torchvision
    import torchvision.transforms.v2.functional as TVF

    assert torchvision.extension._has_ops()
    buffer = BytesIO()
    Image.new("RGB", (16, 12), color=(64, 128, 192)).save(buffer, format="JPEG")
    encoded = torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)
    decoded = torchvision.io.decode_image(
        encoded,
        mode=torchvision.io.ImageReadMode.RGB,
    )
    resized = TVF.resize(
        decoded,
        [8, 8],
        interpolation=TVF.InterpolationMode.BICUBIC,
        antialias=True,
    )
    assert decoded.shape == (3, 12, 16)
    assert resized.shape == (3, 8, 8)
    assert resized.dtype == torch.uint8


def test_qwen3_5_adapter_uses_real_torchvision_without_compat_modules():
    import subprocess

    root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util
import sys
import torchvision
from pathlib import Path
assert torchvision.__version__ in ('0.29.0+cpu', '0.30.0.dev20260918+cpu')
assert torchvision.__file__ is not None and Path(torchvision.__file__).is_file()
from torchtitan_npu.models.qwen3_5.config_registry import qwen35_debugmodel
qwen35_debugmodel()
assert sys.modules['torchvision'] is torchvision
assert 'torchtitan_npu.models.qwen3_5._torchvision_compat' not in sys.modules
# FLA is imported normally when installed; the placeholder cleanup
# is only observable in environments without FLA (e.g. the CI runner).
assert 'fla.ops.gated_delta_rule' not in sys.modules or importlib.util.find_spec('fla') is not None
""",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys([str(root), *sys.path]))},
    )
