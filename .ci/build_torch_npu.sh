#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build a candidate wheel from a fresh checkout. Never install or publish it.
set -euo pipefail

readonly TORCH_NPU_REPOSITORY="https://github.com/Ascend/pytorch.git"
readonly TORCH_NPU_REVISION="848b8e7271069d5a9947bf75fd02a162092b41a6"
readonly TORCH_NPU_VERSION="2.15.0+git848b8e7"
readonly TORCH_VERSION="2.15.0+cpu"
readonly -a SUBMODULES=(
    third_party/op-plugin
    third_party/nlohmann
    third_party/fmt
    third_party/dvm/dvm
    third_party/acl_src/runtime
    third_party/acl_src/ge
    third_party/acl_src/graph-autofusion
)

usage() {
    echo "Usage: PYTHON_BIN=/path/to/python $0 --wheelhouse DIR"
    echo "       PYTHON_BIN=/path/to/python $0 --verify-wheel FILE"
}

if [[ $# != 2 ]] || [[ "$1" != "--wheelhouse" && "$1" != "--verify-wheel" ]]; then
    if [[ $# == 1 && "$1" == "--help" ]]; then usage; exit 0; fi
    usage >&2
    exit 2
fi
mode="$1"
output="$2"
if [[ -z "$output" ]]; then
    echo "A non-empty output path is required." >&2
    exit 2
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")" || { echo "Python interpreter not found." >&2; exit 1; }
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0

verify_wheel() {
    "${PYTHON_BIN}" - "$1" "$TORCH_NPU_VERSION" "$TORCH_VERSION" <<'PY'
import email
import sys
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

if sys.version_info[:2] != (3, 12):
    raise ValueError("Python 3.12 is required to verify this candidate")
path = Path(sys.argv[1])
name, version, _, tags = parse_wheel_filename(path.name)
if canonicalize_name(name) != "torch-npu" or version != Version(sys.argv[2]):
    raise ValueError(f"unexpected wheel name/version: {path.name}")
compatible = set(tags).intersection(sys_tags())
if not any(tag.interpreter == "cp312" and tag.abi == "cp312" and tag.platform != "any" for tag in compatible):
    raise ValueError(f"wheel is incompatible with this CPython 3.12 interpreter: {path.name}")
with zipfile.ZipFile(path) as wheel:
    metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata_files) != 1:
        raise ValueError("wheel must contain exactly one METADATA file")
    metadata = email.message_from_bytes(wheel.read(metadata_files[0]))
if canonicalize_name(metadata["Name"]) != "torch-npu" or Version(metadata["Version"]) != version:
    raise ValueError("wheel metadata disagrees with its filename")
requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
torch_requirements = [
    requirement
    for requirement in requirements
    if canonicalize_name(requirement.name) == "torch"
    and (requirement.marker is None or requirement.marker.evaluate({"extra": ""}))
]
if (
    len(torch_requirements) != 1
    or not torch_requirements[0].specifier
    or not torch_requirements[0].specifier.contains(Version(sys.argv[3]))
):
    raise ValueError(f"wheel does not declare a compatible torch requirement: {torch_requirements}")
print(f"Verified {path.name}: {torch_requirements[0]}")
PY
}

if [[ "$mode" == "--verify-wheel" ]]; then
    verify_wheel "$output"
    exit 0
fi

"${PYTHON_BIN}" - "$TORCH_VERSION" <<'PY'
import sys
from importlib.metadata import version

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version}")
if version("torch") != sys.argv[1]:
    raise SystemExit(f"Expected torch {sys.argv[1]}, got {version('torch')}")
PY

# The upstream build invokes `python3.12`; ensure it selects this environment,
# without replacing or installing an interpreter in the environment.
python_dir="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)"
export PATH="${python_dir}:${PATH}"
if [[ "$(python3.12 -c 'import sys; print(sys.prefix)')" != "$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')" ]]; then
    echo "python3.12 does not resolve to the requested Python environment." >&2
    exit 1
fi
mkdir -p "$output"
wheelhouse="$(cd "$output" && pwd -P)"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-wheel.XXXXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT
checkout="${build_dir}/pytorch"
git clone --no-checkout "$TORCH_NPU_REPOSITORY" "$checkout"
git -C "$checkout" checkout --detach "$TORCH_NPU_REVISION"
[[ "$(git -C "$checkout" rev-parse HEAD)" == "$TORCH_NPU_REVISION" ]]
git -C "$checkout" submodule update --init -- "${SUBMODULES[@]}"
for path in "${SUBMODULES[@]}"; do
    expected="$(git -C "$checkout" ls-tree HEAD -- "$path" | awk '{print $3}')"
    actual="$(git -C "$checkout/$path" rev-parse HEAD)"
    if [[ -z "$expected" || "$expected" != "$actual" ]]; then
        echo "Submodule does not match gitlink: $path" >&2
        exit 1
    fi
done
{
    printf 'repository=%s\nrevision=%s\n' "$TORCH_NPU_REPOSITORY" "$TORCH_NPU_REVISION"
    printf 'python=%s\n' "$("$PYTHON_BIN" --version 2>&1)"
    printf 'torch=%s\n' "$TORCH_VERSION"
    printf 'command=ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair --disable_fxrt --disable_rpc\n'
    git -C "$checkout" submodule status -- "${SUBMODULES[@]}"
} > "${build_dir}/manifest"
(
    cd "$checkout"
    bash ci/build.sh --python=3.12 --torch=2.15.0 --disable_torchair --disable_fxrt --disable_rpc
)
shopt -s nullglob
wheels=("${checkout}"/dist/torch_npu-*.whl)
if [[ ${#wheels[@]} != 1 ]]; then
    echo "Expected exactly one torch_npu wheel; found ${#wheels[@]}." >&2
    exit 1
fi
verify_wheel "${wheels[0]}"
wheel_name="$(basename "${wheels[0]}")"
destination="${wheelhouse}/${wheel_name}"
if [[ -e "$destination" ]]; then
    if ! cmp -s -- "${wheels[0]}" "$destination"; then
        echo "Refusing to overwrite a different wheel: $destination" >&2
        exit 1
    fi
else
    cp -- "${wheels[0]}" "$destination"
fi
checksum="$(sha256sum "$destination" | awk '{print $1}')"
printf 'sha256=%s  %s\n' "$checksum" "$wheel_name" >> "${build_dir}/manifest"
manifest="$(mktemp "${wheelhouse}/${wheel_name}.build-info.XXXXXXXX")"
cp -- "${build_dir}/manifest" "$manifest"
printf 'Wheel: %s\nManifest: %s\n' "$destination" "$manifest"
