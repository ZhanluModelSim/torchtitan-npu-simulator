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

source /home/jenkins/Ascend/cann-9.2.0/set_env.sh

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
pip install -r requirements-dev.txt

python -m pytest -v --tb=short tests/unit_tests
