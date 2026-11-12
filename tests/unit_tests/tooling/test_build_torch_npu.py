# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the candidate wheel verifier without building or loading an NPU runtime."""

import os
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".ci/build_torch_npu.sh"


def _candidate_wheel(tmp_path, *, version="2.15.0+git848b8e7", torch_requirement="torch==2.15.0.*", tag=None):
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    tag = tag or f"cp312-cp312-{platform}"
    wheel = tmp_path / f"torch_npu-{version}-{tag}.whl"
    metadata = f"Metadata-Version: 2.1\nName: torch-npu\nVersion: {version}\nRequires-Dist: {torch_requirement}\n\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"torch_npu-{version}.dist-info/METADATA", metadata)
    return wheel


def _verify(wheel):
    return subprocess.run(
        [str(SCRIPT), "--verify-wheel", str(wheel)],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )


def test_candidate_wheel_requires_matching_python_and_torch_metadata(tmp_path):
    wheel = _candidate_wheel(tmp_path)
    result = _verify(wheel)
    assert result.returncode == 0, result.stderr
    assert "torch==2.15.0.*" in result.stdout


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"torch_requirement": "torch==2.14.0"}, "compatible torch requirement"),
        ({"torch_requirement": "torch"}, "compatible torch requirement"),
        ({"version": "2.15.0+git0000000"}, "unexpected wheel name/version"),
        ({"tag": "py3-none-any"}, "incompatible with this CPython 3.12"),
    ],
    ids=["torch-mismatch", "torch-unpinned", "revision-mismatch", "platform-mismatch"],
)
def test_candidate_wheel_rejects_wrong_metadata_or_tag(tmp_path, kwargs, expected):
    result = _verify(_candidate_wheel(tmp_path, **kwargs))
    assert result.returncode != 0
    assert expected in result.stderr


def test_build_helper_rejects_invalid_arguments():
    result = subprocess.run([str(SCRIPT), "--wheelhouse"], text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_runtime_requirements_match_package_dependencies():
    import tomllib

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    requirements = {
        canonicalize_name(item.name): item
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith(("#", "--"))
        for item in (Requirement(line),)
    }
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {}
    for value in project["dependencies"]:
        item = Requirement(value)
        declared[canonicalize_name(item.name)] = item
    for name, expected in {
        "torch": "2.15.0+cpu",
        "torchvision": "0.29.2+cpu",
        "torch-npu": "2.15.0.dev20261105",
    }.items():
        assert str(requirements[name].specifier) == str(declared[name].specifier) == f"=={expected}"
    extras = [Requirement(value) for value in project["optional-dependencies"]["multimodal"]]
    vision = next(item for item in extras if canonicalize_name(item.name) == "torchvision")
    assert str(vision.specifier) == "==0.29.2+cpu"
