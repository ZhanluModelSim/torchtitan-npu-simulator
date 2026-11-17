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
TORCHTITAN_VERSION="$(grep -E '^torchtitan==' "${PROJECT_ROOT}/requirements.txt" | head -1 | cut -d= -f3 | tr -d '[:space:][:cntrl:]')"
TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-v${TORCHTITAN_VERSION}}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${PROJECT_ROOT}/third_party/torchtitan}"

source /home/jenkins/Ascend/cann-9.2.0/set_env.sh
# Prevent torch from eagerly importing torch_npu while spmd_types is initializing.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
pip install -r requirements-dev.txt

if [[ ! -d "${TORCHTITAN_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${TORCHTITAN_DIR}")"
    git clone --filter=blob:none --no-checkout "${TORCHTITAN_REPO}" "${TORCHTITAN_DIR}"
fi
git -C "${TORCHTITAN_DIR}" fetch --depth 1 origin "${TORCHTITAN_COMMIT}"
git -C "${TORCHTITAN_DIR}" checkout --detach --quiet "${TORCHTITAN_COMMIT}"
pip install --no-deps -e "${TORCHTITAN_DIR}"
pip install \
    --find-links https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torchft/index.html \
    "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')" \
    'torchft==0.2.0'
pip install --no-deps -e .

# The image predates wheels added to requirements.txt.  Install only those wheels:
# the image pins a *different* torch/torch_npu build than requirements.txt does, so
# re-running the whole file would upgrade the CANN stack mid-job (it broke the
# fused-op smoke tests once).  New pure-Python runtime wheels go here by name.
pip install --no-deps attn-gym==0.0.16

pip install --no-deps torchvision==0.30.0.dev20260918 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu

python -c 'from torchtitan.distributed.context_parallel.api import cp_shard; print(cp_shard.__module__)'
python -m pytest -v --tb=short tests/unit_tests
