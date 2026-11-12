# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Use one interpreter for pip, TorchTitan setup, and the test process.
if ! PYTHON_BIN="$(command -v "${PYTHON_BIN:-python3.12}")"; then
    echo "Python interpreter not found" >&2
    exit 1
fi
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
readonly PYTHON_BIN
readonly PYTHON_SHIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-python.XXXXXX")"
ln -s "${PYTHON_BIN}" "${PYTHON_SHIM_DIR}/python"
ln -s "${PYTHON_BIN}" "${PYTHON_SHIM_DIR}/python3"
export PATH="${PYTHON_SHIM_DIR}:${PATH}"

_cleanup_python_shims() {
    rm -f -- \
        "${PYTHON_SHIM_DIR}/python" \
        "${PYTHON_SHIM_DIR}/python3"
    rmdir -- "${PYTHON_SHIM_DIR}"
}
trap _cleanup_python_shims EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
source "${SCRIPT_DIR}/setup_torchtitan.sh"
TORCHTITAN_COMMIT="$(_torchtitan_revision)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${PROJECT_ROOT}/third_party/torchtitan}"

# v0.4.0 pairs torch_npu 2.15 with CANN 9.3; keep 9.2 as the image fallback.
if [[ -f /home/jenkins/Ascend/cann-9.3.0/set_env.sh ]]; then
    source /home/jenkins/Ascend/cann-9.3.0/set_env.sh
else
    source /home/jenkins/Ascend/cann-9.2.0/set_env.sh
fi
# Prevent torch from eagerly importing torch_npu while spmd_types is initializing.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# torch 2.15 headers require C++20 when cann_ops_nn builds its JIT extensions.
export CXXFLAGS="${CXXFLAGS:-} -std=c++20"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m pip install -r requirements-dev.txt

_setup_torchtitan "${TORCHTITAN_DIR}"
export TORCHTITAN_DIR
# Resolve the runtime dependencies pinned by the target TorchTitan release.
"${PYTHON_BIN}" -m pip install torch_remat==0.2.2
# The UT image is x86_64 CPython 3.12. Pull the pinned wheel set from the CI wheel
# mirror; the PTA daily bundle is too slow from the restricted CI network.
test "$(uname -m)" = "x86_64" || {
    echo "The CI wheel set is published for x86_64 only" >&2
    exit 1
}
CI_WHEELS_TAG="${CI_WHEELS_TAG:-torch-2.15.0-cpu-cp312-x86_64}"
CI_WHEELS_BASE="${CI_WHEELS_BASE:-https://gitcode.com/hitwdy/torchtitan-npu-ci-wheels/releases/download/${CI_WHEELS_TAG}}"
TORCH_WHEELHOUSE="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-torch.XXXXXX")"
trap '_cleanup_python_shims; rm -rf -- "${TORCH_WHEELHOUSE}"' EXIT
TORCH_VERSION="$(sed -n 's/^torch==//p' requirements.txt)"
TORCHVISION_VERSION="$(sed -n 's/^torchvision==//p' requirements.txt)"
TORCH_NPU_VERSION="$(sed -n 's/^torch_npu==//p' requirements.txt)"
TORCH_WHEEL="torch-${TORCH_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCHVISION_WHEEL="torchvision-${TORCHVISION_VERSION}-py3-none-manylinux_2_28_x86_64.whl"
TORCH_NPU_WHEEL="torch_npu-${TORCH_NPU_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
curl -fL --retry 3 "${CI_WHEELS_BASE}/${TORCH_WHEEL//+/%2B}" -o "${TORCH_WHEELHOUSE}/${TORCH_WHEEL}"
curl -fL --retry 3 "${CI_WHEELS_BASE}/${TORCHVISION_WHEEL//+/%2B}" -o "${TORCH_WHEELHOUSE}/${TORCHVISION_WHEEL}"
curl -fL --retry 3 "${CI_WHEELS_BASE}/${TORCH_NPU_WHEEL//+/%2B}" -o "${TORCH_WHEELHOUSE}/${TORCH_NPU_WHEEL}"
(
    cd "${TORCH_WHEELHOUSE}"
    sha256sum -c <<'EOF'
0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3  torch-2.15.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
87631236a47be98ab21301e8cb4fc1cbb9325fe0aa7a022e18d2f894087b68c3  torchvision-0.29.2+cpu-py3-none-manylinux_2_28_x86_64.whl
45fff2691b918f8591801c3bbae4bc4c0799e38f2bf811766dd49b35cc5a9ffb  torch_npu-2.15.0.dev20261105-cp312-cp312-manylinux_2_28_x86_64.whl
EOF
)
"${PYTHON_BIN}" -m pip install --no-cache-dir --force-reinstall --no-deps \
    "${TORCH_WHEELHOUSE}/${TORCH_WHEEL}" \
    "${TORCH_WHEELHOUSE}/${TORCHVISION_WHEEL}" \
    "${TORCH_WHEELHOUSE}/${TORCH_NPU_WHEEL}"
"${PYTHON_BIN}" -m pip install -e "${TORCHTITAN_DIR}"
# Keep the torch stack installed above and the target TorchTitan attention version.
"${PYTHON_BIN}" -m pip install --no-deps \
    --find-links https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torchft/index.html \
    'torchft==0.2.0' 'attn-gym==0.0.17'
"${PYTHON_BIN}" -m pip install --no-deps -e .

python -c 'import grain, renderers, spmd_types, torch_remat, torchtitan, torchtitan_npu'
python -m pytest -v --tb=short tests/unit_tests
