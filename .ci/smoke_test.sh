# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source /usr/local/Ascend/cann/set_env.sh

export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

pip install -r requirements_dev.txt
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple

# Global variable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output"
REPORT_DIR="${PROJECT_ROOT}/test_reports"
INTEGRATION_REPORT_DIR="${PROJECT_ROOT}/test_reports/integration_tests"
TORCHTITAN_BRANCH="main"
TORCHTITAN_COMMIT="ac13e536c84e7f6647b14fa9375c3c8a8a2b8578"
TORCHTITAN_DIR="${PROJECT_ROOT}/third_party/torchtitan"
TORCHAIR_COMMIT="3c9418c2"
TORCHAIR_DIR="${PROJECT_ROOT}/third_party/torchair"
DEEPSEEK_TOKENIZER_REPO="${DEEPSEEK_TOKENIZER_REPO:-https://gitcode.com/hitwdy/deepseekv4.git}"
DEEPSEEK_V4_TOKENIZER_DIR="${PROJECT_ROOT}/tests/assets/tokenizer/deepseekv4_tokenizer"
DEEPSEEK_V32_TOKENIZER_DIR="${PROJECT_ROOT}/tests/assets/tokenizer/deepseekv32_tokenizer"
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
SMOKE_STEPS=${SMOKE_STEPS:-1}
# Known false-positive patterns to exclude from error detection
ERROR_EXCLUDE_PATTERNS=("TORCH_NCCL_ASYNC_ERROR_HANDLING")

export PYTHONPATH="${PROJECT_ROOT}/tests/smoke_tests/npu_bypass_triton_codegen:${PYTHONPATH:-}"

# Prepare environment: install packages and clone torchtitan source.
_setup_env() {
    cd "$PROJECT_ROOT"

    # Ensure torchtitan_npu is installed
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Ensure the torch.compile NPU codegen extension is installed.
    echo "Installing inductor_npu_ext from torchair ${TORCHAIR_COMMIT}..."
    mkdir -p third_party
    if [[ ! -d "$TORCHAIR_DIR/.git" ]]; then
        git clone https://gitcode.com/Ascend/torchair.git "$TORCHAIR_DIR"
    fi
    git -C "$TORCHAIR_DIR" fetch origin
    git -C "$TORCHAIR_DIR" checkout "$TORCHAIR_COMMIT"
    pip3 install -e "$TORCHAIR_DIR/experimental/_inductor_npu_ext/python/"

    # Clone torchtitan source if not exists
    if [[ ! -d "$TORCHTITAN_DIR/.git" ]]; then
        echo "Cloning torchtitan source..."
        git clone --branch "$TORCHTITAN_BRANCH" \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git "$TORCHTITAN_DIR"
    fi

    git -C "$TORCHTITAN_DIR" fetch origin "$TORCHTITAN_BRANCH"
    git -C "$TORCHTITAN_DIR" checkout "$TORCHTITAN_COMMIT"
}

_copy_deepseek_tokenizer() {
    local name="$1"
    local source_dir="$2"
    local target_dir="$3"
    local tokenizer_json="${target_dir}/tokenizer.json"
    local tokenizer_config="${target_dir}/tokenizer_config.json"

    if [[ -s "$tokenizer_json" && -s "$tokenizer_config" ]]; then
        echo "${name} tokenizer already exists: ${target_dir}"
        return 0
    fi

    if [[ ! -s "${source_dir}/tokenizer.json" || ! -s "${source_dir}/tokenizer_config.json" ]]; then
        echo "${name} tokenizer is incomplete in ${source_dir}"
        find "$source_dir" -maxdepth 1 -type f -print 2>/dev/null || true
        return 1
    fi

    echo "Preparing ${name} tokenizer..."
    mkdir -p "$target_dir"
    cp "${source_dir}/tokenizer.json" "${source_dir}/tokenizer_config.json" \
        "$target_dir"/

    if [[ ! -s "$tokenizer_json" || ! -s "$tokenizer_config" ]]; then
        echo "${name} tokenizer is incomplete in ${target_dir}"
        find "$target_dir" -maxdepth 1 -type f -print || true
        return 1
    fi

    echo "${name} tokenizer prepared: ${target_dir}"
}

_prepare_deepseek_tokenizers() {
    local tokenizer_repo="$DEEPSEEK_TOKENIZER_REPO"
    local source_dir="$tokenizer_repo"
    local tmp_dir=""

    if [[ ! -d "$source_dir" ]]; then
        echo "Downloading DeepSeek tokenizers..."
        tmp_dir=$(mktemp -d)
        git clone --depth 1 --filter=blob:none --no-checkout \
            "$tokenizer_repo" "$tmp_dir"
        git -C "$tmp_dir" sparse-checkout set --no-cone \
            "/deepseekv4/tokenizer.json" \
            "/deepseekv4/tokenizer_config.json" \
            "/deepseekv32/tokenizer.json" \
            "/deepseekv32/tokenizer_config.json"
        git -C "$tmp_dir" checkout
        source_dir="$tmp_dir"
    fi

    _copy_deepseek_tokenizer \
        "DeepSeek V4" \
        "${source_dir}/deepseekv4" \
        "$DEEPSEEK_V4_TOKENIZER_DIR"
    _copy_deepseek_tokenizer \
        "DeepSeek V3.2" \
        "${source_dir}/deepseekv32" \
        "$DEEPSEEK_V32_TOKENIZER_DIR"

    if [[ -n "$tmp_dir" ]]; then
        rm -rf "$tmp_dir"
    fi
}

# Run integrated test: end-to-end training configurations.
run_torchtitan_npu_smoke() {
    echo "Running torchtitan-npu smoke test..."

    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/smoke_test.log"
    local start_time=$(date +%s)
    local integration_test="${PROJECT_ROOT}/tests/smoke_tests/integration_test.py"
    echo "Verifying torchtitan..."
    python -c "import torchtitan; print('torchtitan ok')"

    echo "Done."

    cd "$PROJECT_ROOT"

    echo "Starting torchtitan-npu integration test..."

    set +e
    timeout $TIMEOUT_SECONDS bash -c "
        python "${integration_test}" "${INTEGRATION_REPORT_DIR}" --ngpu 2
    " 2>&1 | tee "$smoke_log"
    local exit_code
    exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "torchtitan-npu smoke test finished in ${duration}s"
    if ! analyse_smoke_result "$smoke_log" "$exit_code"; then
        echo "torchtitan-npu smoke test failed."
        echo "--- Error details ---"
        grep -iE "error|exception|traceback" "$smoke_log" 2>/dev/null \
            | grep -vE "$(IFS='|'; echo "${ERROR_EXCLUDE_PATTERNS[*]}")" || true
        grep -iE "loss:\s*(nan|inf)" "$smoke_log" 2>/dev/null || true
        echo "--- End error details ---"
        exit 1
    fi
}

analyse_smoke_result() {
    local log_file="$1"
    local exit_code="$2"

    echo "Analyzing smoke test results..."
    local has_error=false

    if [[ "$exit_code" -eq 124 ]]; then
        echo "Timeout!"
        has_error=true
    fi

    if grep -iE "error|exception|traceback" "$log_file" 2>/dev/null \
       | grep -qvE "$(IFS='|'; echo "${ERROR_EXCLUDE_PATTERNS[*]}")"; then
        has_error=true
    fi

    if grep -qiE "loss:\s*(nan|inf)" "$log_file" 2>/dev/null; then
        echo "loss error (NAN/Inf)"
        has_error=true
    fi

    # grepping mstep to match colored output. Example log line:
    # [2024-06-17 10:00:00] INFO: \e[0;31mstep: 10
    local complete_steps=$(grep -oP "mstep[:\s]*\K\d+" "$log_file" 2>/dev/null | tail -1)
    complete_steps=${complete_steps:-0}
    if [[ "$complete_steps" -ge "$SMOKE_STEPS" ]]; then
        echo "Completed $complete_steps steps"
    elif [[ "$complete_steps" -gt 0 ]]; then
        echo "Only finished $complete_steps/$SMOKE_STEPS steps"
    else
        echo "No output steps detected"
    fi

    if [[ "$has_error" == "true" ]] || [[ "$exit_code" -ne 0 ]]; then
        return 1
    fi

    return 0
}

# Run upstream integration tests with NPU.
run_torchtitan_smoke() {
    echo "Running torchtitan upstream integration tests..."

    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/upstream_smoke.log"
    local upstream_output_dir="${REPORT_DIR}/upstream_integration_output"
    local start_time=$(date +%s)
    local saved_pythonpath="$PYTHONPATH"

    # Disable tests that use torch.compile (Autofuse takes too long to compile on CI).
    local features_py="${TORCHTITAN_DIR}/tests/integration_tests/features.py"
    for name in "1d_compile" "1d_compile_sac_op" "2d_compile" "3d_compile"; do
        if ! grep -q "\"$name\",.*disabled=True" "$features_py" 2>/dev/null; then
            sed -i "/\"$name\",/s/$/ disabled=True,/" "$features_py"
        fi
    done

    cd "$TORCHTITAN_DIR"

    rm -rf "$upstream_output_dir"
    mkdir -p "$upstream_output_dir"

    export PYTHONPATH="${TORCHTITAN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

    # Run upstream integration tests with torchtitan_npu patches applied.
    local cmd=(
        python3 -m tests.integration_tests.run_tests
        "$upstream_output_dir"
        --test_suite features
        --test_name all
        --ngpu 2
    )

    set +e
    "${cmd[@]}" 2>&1 | tee "$smoke_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    cd "$PROJECT_ROOT"
    export PYTHONPATH="$saved_pythonpath"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "torchtitan integration test duration: ${duration}s"

    if [[ $exit_code -eq 0 ]]; then
        echo "torchtitan integration tests passed!"
    else
        echo "torchtitan integration tests failed (exit_code=$exit_code)"
        exit $exit_code
    fi
}


_setup_env
_prepare_deepseek_tokenizers

_wait_npu_idle() {
    local max_wait=${1:-10}
    local threshold=${2:-500}
    for i in $(seq 1 "$max_wait"); do
        local used
        used=$(npu-smi info 2>/dev/null | grep -oP '\d+(?=\s+/\s+\d+)' | sort -n | tail -1)
        used=${used:-0}
        if [ "$used" -lt "$threshold" ]; then
            echo "NPU idle (max HBM: ${used}MB)"
            return 0
        fi
        echo "Waiting for NPU memory to free... (${used}MB used, attempt $i/$max_wait)"
        sleep 1
    done
    echo "Warning: NPU memory still not idle after ${max_wait}s"
    return 1
}

# run_torchtitan_smoke
_wait_npu_idle 10 5000
run_torchtitan_npu_smoke
pytest -v --tb=short tests/smoke_tests

# Smoke test success sentinel, grepped by gitcode ci. Do not modify.
echo "smoke test passed."
