# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# set python3.12 and pip3.12 as default
if ! PYTHON_BIN="$(command -v python3.12)"; then
    echo "python3.12 not found" >&2
    exit 1
fi
if ! PIP_BIN="$(command -v pip3.12)"; then
    echo "pip3.12 not found" >&2
    exit 1
fi
readonly PYTHON_BIN PIP_BIN
readonly PYTHON_SHIM_DIR="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-python.XXXXXX")"
ln -s "${PYTHON_BIN}" "${PYTHON_SHIM_DIR}/python"
ln -s "${PYTHON_BIN}" "${PYTHON_SHIM_DIR}/python3"
ln -s "${PIP_BIN}" "${PYTHON_SHIM_DIR}/pip"
export PATH="${PYTHON_SHIM_DIR}:${PATH}"

_cleanup_python_shims() {
    rm -f -- \
        "${PYTHON_SHIM_DIR}/python" \
        "${PYTHON_SHIM_DIR}/python3" \
        "${PYTHON_SHIM_DIR}/pip"
    rmdir -- "${PYTHON_SHIM_DIR}"
}
trap _cleanup_python_shims EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
source "${SCRIPT_DIR}/setup_torchtitan.sh"
TORCHTITAN_COMMIT="$(_torchtitan_revision)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${PROJECT_ROOT}/third_party/torchtitan}"

source /home/jenkins/Ascend/cann-9.2.0/set_env.sh
# Prevent torch from eagerly importing torch_npu while spmd_types is initializing.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
pip install -r requirements-dev.txt

_setup_torchtitan "${TORCHTITAN_DIR}"
export TORCHTITAN_DIR
# Resolve upstream runtime dependencies using the transport configured above.
pip install -e "${TORCHTITAN_DIR}"
# The NPU package overrides upstream's attn-gym pin; keep the image's torch stack.
pip install --no-deps \
    --find-links https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torchft/index.html \
    'torchft==0.2.0' 'attn-gym==0.0.16'
pip install --no-deps -e .

pip install --no-deps torchvision==0.29.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

python -c 'import grain, renderers, spmd_types, torch_remat, torchtitan, torchtitan_npu'
python -m pytest -v --tb=short tests/unit_tests
