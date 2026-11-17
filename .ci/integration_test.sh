#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI integration entrypoint. Runs only the model ST suite through the existing
# `tests.integration_tests` runner; the pytest smoke phase lives in
# `.ci/smoke_test.sh` and runs as its own CI job.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# The CI Triton launcher still uses C++17, but PyTorch ATen requires C++20.
# Restore this GDN-dependent case once the CI toolchain is fixed.
INTEGRATION_TESTS_START="$(date +%s)"
"${PYTHON_BIN}" -m tests.integration_tests.run_tests \
    "${OUTPUT_DIR}" \
    --test_suite models \
    --exclude qwen3_5_debugmodel_1rank \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    --ngpu "${NGPU}"
INTEGRATION_TESTS_END="$(date +%s)"

echo "tests.integration_tests finished in $((INTEGRATION_TESTS_END - INTEGRATION_TESTS_START))s"
echo "integration test passed."
