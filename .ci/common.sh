#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared preparation for the CI test entrypoints. `.ci/smoke_test.sh` and
# `.ci/integration_test.sh` both source this file; the smoke and integration
# phases run as separate CI jobs, so each entrypoint must be able to prepare the
# interpreter, the CANN environment and the torchtitan checkout on its own.
#
# CI supplies the interpreter, dependencies, accelerator runtime, and data
# assets. This file only applies contract defaults on top of them.
#
# Sourced, not executed: the caller keeps ownership of its exit status, and the
# EXIT trap registered here removes the interpreter shims when the caller exits.
# Sourcing twice in one shell is a no-op.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a test entrypoint; do not execute it directly." >&2
    exit 1
fi

if [[ -n "${TORCHTITAN_NPU_CI_SETUP_DONE:-}" ]]; then
    return 0
fi

set -euo pipefail

readonly CANN_ENV_PATH=/usr/local/Ascend/cann/set_env.sh
source "${CANN_ENV_PATH}"
echo "[CANN] sourced ${CANN_ENV_PATH}"
echo "[CANN] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "[CANN] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
source "${PROJECT_ROOT}/.ci/setup_torchtitan.sh"
TORCHTITAN_COMMIT="$(_torchtitan_revision)"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${PROJECT_ROOT}/third_party/torchtitan}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/test_reports/smoke}"
NGPU="${NGPU:-4}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Required interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

# Binding torchrun/python launchers to the CI's python 3.12 interpreter.
PYTHON_SHIMS="$(mktemp -d)"
trap 'rm -rf "${PYTHON_SHIMS}"' EXIT
ln -s "$(command -v "${PYTHON_BIN}")" "${PYTHON_SHIMS}/python3"
cat >"${PYTHON_SHIMS}/torchrun" <<EOF
#!/usr/bin/env bash
exec "$(command -v "${PYTHON_BIN}")" -m torch.distributed.run "\$@"
EOF
chmod +x "${PYTHON_SHIMS}/torchrun"
export PATH="${PYTHON_SHIMS}:${PATH}"

# Clone and install torchtitan.
_setup_torchtitan "${TORCHTITAN_DIR}"

"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir -e "${TORCHTITAN_DIR}"
"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir -e "${PROJECT_ROOT}"

# The image predates wheels added to requirements.txt; install only the missing
# wheels, never the whole file, whose torch/torch_npu pins differ from the image
# build.
"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir attn-gym==0.0.16 spmd_types==0.2.5 \
    'torch_checkpointing==0.1.1' 'torch_remat>=0.2.0'
# Grain needs its data-pipeline dependencies (for example etils and array-record).
# Its dependency graph does not include torch or torch_npu.
"${PYTHON_BIN}" -m pip install --break-system-packages --no-cache-dir grain==0.2.18
# The CI image supplies torch 2.14.0.dev; the matching torchvision CPU wheel
# is installed with --no-deps (resolving its torch pin would fight the image).
"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torchvision==0.29.0+cpu"
"${PYTHON_BIN}" -c 'import torchtitan, torchtitan_npu, torchvision'

export MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
export CONFIG="${CONFIG:-deepseek_v4_debugmodel}"
export NGPU
export LOG_RANK="${LOG_RANK:-0}"
export PYTHON_BIN
# The Qwen3.5 launcher resolves tokenizer and cc12m-test assets from the
# torchtitan checkout; expose the clone location to child processes.
export TORCHTITAN_DIR

TORCHTITAN_NPU_CI_SETUP_DONE=1
