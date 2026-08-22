#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI supplies the interpreter, dependencies, accelerator runtime, and data
# assets. This entrypoint is a thin adapter that applies contract defaults and
# delegates execution to the integration-test runner.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-$(grep '^torchtitan @ ' requirements.txt | cut -d@ -f3)}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${PROJECT_ROOT}/third_party/torchtitan}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/test_reports/smoke}"
NGPU="${NGPU:-8}"

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
if [[ ! -d "${TORCHTITAN_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${TORCHTITAN_DIR}")"
    git clone --filter=blob:none --no-checkout "${TORCHTITAN_REPO}" "${TORCHTITAN_DIR}"
fi
git -C "${TORCHTITAN_DIR}" fetch --depth 1 origin "${TORCHTITAN_COMMIT}"
git -C "${TORCHTITAN_DIR}" checkout --detach --quiet "${TORCHTITAN_COMMIT}"

"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir -e "${TORCHTITAN_DIR}"
"${PYTHON_BIN}" -m pip install --break-system-packages --no-deps --no-cache-dir -e "${PROJECT_ROOT}"
"${PYTHON_BIN}" -c 'import torchtitan, torchtitan_npu'

export MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
export CONFIG="${CONFIG:-deepseek_v4_debugmodel}"
export NGPU
export LOG_RANK="${LOG_RANK:-0}"
export PYTHON_BIN

"${PYTHON_BIN}" -m tests.integration_tests.run_tests \
    "${OUTPUT_DIR}" \
    --test_suite models \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    --ngpu "${NGPU}"

echo "smoke test passed."
