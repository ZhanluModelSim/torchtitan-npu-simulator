# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly TORCHTITAN_BRANCH="main"
readonly TORCHTITAN_COMMIT="cc286a63599e42480a07928cc362e514ae448a85"
readonly TORCHTITAN_DIR="/tmp/torchtitan"

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


echo "============== preparing torchtitan =============="
echo "torchtitan target directory: ${TORCHTITAN_DIR}"
echo "torchtitan expected commit: ${TORCHTITAN_COMMIT}"
if [[ -d "${TORCHTITAN_DIR}" ]] && \
    git -C "${TORCHTITAN_DIR}" rev-parse --verify HEAD 2>/dev/null; then
    git -C "${TORCHTITAN_DIR}" fetch origin "${TORCHTITAN_BRANCH}"
    git -C "${TORCHTITAN_DIR}" checkout "${TORCHTITAN_COMMIT}"
else
    mkdir -p "$(dirname "${TORCHTITAN_DIR}")"
    git clone --branch "${TORCHTITAN_BRANCH}" \
        https://gitcode.com/GitHub_Trending/to/torchtitan.git \
        "${TORCHTITAN_DIR}"
    git -C "${TORCHTITAN_DIR}" checkout "${TORCHTITAN_COMMIT}"
fi

echo "============== installing packages =============="
cd "${PROJECT_ROOT}"
export PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements-dev.txt
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "============== begin lint =============="
python3 -m pre_commit run --all-files --show-diff-on-failure
