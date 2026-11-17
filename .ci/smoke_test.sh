#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CI smoke entrypoint. Runs only the pytest smoke suite in tests/smoke_tests.
# The integration model ST phase lives in `.ci/integration_test.sh` and runs as
# its own CI job.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Tests that the 2026-09-29 CI image cannot run yet. They are deselected here
# instead of patched in the repo; drop a deselect once its toolchain issue is
# fixed.
#   * tests/smoke_tests/ops/test_mhc_triton.py (26 cases): triton-ascend
#     hardcodes -std=c++17 when it builds its NPU host stub / precompiled
#     header, while torch >= 2.14 refuses anything below C++20
#     (ATen/ATen.h:5), so every Triton kernel launch fails to compile.
#   * test_qwen3_5_npu_runtime.py::test_merged_gdn_forward_backward uses the
#     same Triton host-stub compiler and fails with the same C++17/C++20 error.
SMOKE_DESELECT=(
    --deselect tests/smoke_tests/ops/test_mhc_triton.py
    --deselect tests/smoke_tests/ops/test_qwen3_5_npu_runtime.py::test_merged_gdn_forward_backward
)

SMOKE_TESTS_START="$(date +%s)"
"${PYTHON_BIN}" -m pytest -v --tb=short "${SMOKE_DESELECT[@]}" tests/smoke_tests
SMOKE_TESTS_END="$(date +%s)"

echo "tests.smoke_tests finished in $((SMOKE_TESTS_END - SMOKE_TESTS_START))s"
echo "smoke test passed."
