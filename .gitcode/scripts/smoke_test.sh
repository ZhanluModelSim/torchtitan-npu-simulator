#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

readonly TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
readonly TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-cc286a63599e42480a07928cc362e514ae448a85}"
readonly TOKENIZER_REPO="${DEEPSEEK_TOKENIZER_REPO:-https://gitcode.com/hitwdy/deepseekv4.git}"
readonly NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-2}"
readonly SMOKE_STEPS="${SMOKE_STEPS:-5}"
readonly SMOKE_SEQ_LEN="${SMOKE_SEQ_LEN:-2048}"
readonly SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-900}"
readonly DRY_RUN="${SMOKE_DRY_RUN:-0}"

readonly -a ALL_CASES=(
    dsv32_fsdp
    dsv32_ep
    dsv4_fsdp
    dsv4_ep
)

readonly DSV32_OVERRIDES="\
torchtitan_npu.override.deepseek_v3_2.sparse_attention.kernel,\
torchtitan_npu.override.deepseek_v3_2.sparse_attention.mask_handler,\
torchtitan_npu.override.common.rope.npu_rope_override,\
torchtitan_npu.override.common.rope.npu_single_complex_rope_override,\
torchtitan_npu.override.common.rms_norm.npu_rms_norm_override"

readonly DSV4_OVERRIDES="\
torchtitan_npu.override.deepseek_v4.fused_dsa.npu_smla_tnd_override,\
torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_rope_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_single_rope_override,\
torchtitan_npu.override.deepseek_v4.golden.rms_norm_golden"

WORK_DIR=""
TORCHTITAN_DIR=""
REPORT_DIR=""
DSV32_TOKENIZER_DIR=""
DSV4_TOKENIZER_DIR=""

_usage() {
    cat <<'EOF'
Usage: .gitcode/scripts/smoke_test.sh [--list] [CASE ...]

Cases:
  dsv32_fsdp     DeepSeek-V3.2 FSDP
  dsv32_ep       DeepSeek-V3.2 FSDP + EP
  dsv4_fsdp      DeepSeek-V4 FSDP
  dsv4_ep        DeepSeek-V4 FSDP + EP

With no CASE, all cases run in the order above. Set SMOKE_DRY_RUN=1 to print
the generated torchrun commands without preparing dependencies or using NPUs.
The default workload uses 2 NPUs, a sequence length of 2048, and 5 steps.
EOF
}

_contains_case() {
    local expected="$1"
    local item
    for item in "${ALL_CASES[@]}"; do
        if [[ "${item}" == "${expected}" ]]; then
            return 0
        fi
    done
    return 1
}

_select_cases() {
    if [[ $# -eq 0 ]]; then
        SELECTED_CASES=("${ALL_CASES[@]}")
        return
    fi

    SELECTED_CASES=()
    local item
    for item in "$@"; do
        if ! _contains_case "${item}"; then
            echo "Unknown smoke case: ${item}" >&2
            _usage >&2
            exit 2
        fi
        SELECTED_CASES+=("${item}")
    done
}

_source_cann_env() {
    local env_script
    for env_script in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/cann/set_env.sh; do
        if [[ -f "${env_script}" ]]; then
            # shellcheck disable=SC1090
            source "${env_script}"
            return
        fi
    done
    echo "CANN environment script not found" >&2
    return 1
}

_clone_at_commit() {
    local repo="$1"
    local commit="$2"
    local target="$3"
    local sparse_path="${4:-}"

    git -c init.defaultBranch=main init --quiet "${target}"
    git -C "${target}" remote add origin "${repo}"
    git -C "${target}" fetch --filter=blob:none --depth 1 origin "${commit}"
    if [[ -n "${sparse_path}" ]]; then
        git -C "${target}" sparse-checkout set "${sparse_path}"
    fi
    git -C "${target}" checkout --detach FETCH_HEAD
}

_prepare_tokenizers() {
    local tokenizer_source="${WORK_DIR}/tokenizer-source"
    git clone --depth 1 --filter=blob:none --no-checkout \
        "${TOKENIZER_REPO}" "${tokenizer_source}"
    git -C "${tokenizer_source}" sparse-checkout set --no-cone \
        /deepseekv32/tokenizer.json \
        /deepseekv32/tokenizer_config.json \
        /deepseekv4/tokenizer.json \
        /deepseekv4/tokenizer_config.json
    git -C "${tokenizer_source}" checkout

    DSV32_TOKENIZER_DIR="${tokenizer_source}/deepseekv32"
    DSV4_TOKENIZER_DIR="${tokenizer_source}/deepseekv4"

    local file
    for file in \
        "${DSV32_TOKENIZER_DIR}/tokenizer.json" \
        "${DSV32_TOKENIZER_DIR}/tokenizer_config.json" \
        "${DSV4_TOKENIZER_DIR}/tokenizer.json" \
        "${DSV4_TOKENIZER_DIR}/tokenizer_config.json"; do
        if [[ ! -s "${file}" ]]; then
            echo "Tokenizer asset is missing or empty: ${file}" >&2
            return 1
        fi
    done
}

_install_requirements() {
    local requirements_file="${PROJECT_ROOT}/requirements.txt"
    local requirements_without_torch_npu="${WORK_DIR}/requirements_without_torch_npu.txt"
    local torch_npu_requirement

    # torch_npu declares a stable torch dependency in its wheel metadata,
    # while requirements.txt intentionally selects the matching torch nightly.
    # Install the other requirements first, then install torch_npu without
    # dependencies, matching the image build in .gitcode/gitcode.dockerfile.
    sed '/^torch_npu @ /d' "${requirements_file}" > "${requirements_without_torch_npu}"
    "${PYTHON_BIN}" -m pip install --no-cache-dir --prefer-binary \
        --retries 10 --timeout 120 \
        -r "${requirements_without_torch_npu}"

    torch_npu_requirement="$(sed -n '/^torch_npu @ /p' "${requirements_file}")"
    test -n "${torch_npu_requirement}"
    "${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps \
        --retries 10 --timeout 120 \
        "${torch_npu_requirement}"
}

_prepare_environment() {
    _source_cann_env

    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_BIN="python3.12"
    else
        PYTHON_BIN="python3"
    fi
    readonly PYTHON_BIN

    "${PYTHON_BIN}" -c \
        'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-smoke.XXXXXX")"
    REPORT_DIR="${SMOKE_REPORT_DIR:-${WORK_DIR}/reports}"
    TORCHTITAN_DIR="${WORK_DIR}/torchtitan"
    mkdir -p "${REPORT_DIR}"

    _install_requirements
    "${PYTHON_BIN}" -m pip install --no-deps -e "${PROJECT_ROOT}"
    _clone_at_commit "${TORCHTITAN_REPO}" "${TORCHTITAN_COMMIT}" "${TORCHTITAN_DIR}"
    _prepare_tokenizers

    "${PYTHON_BIN}" -c "import torch, torch_npu, torchtitan, torchtitan_npu; print(torch.__version__, torch_npu.__version__)"
}

_prepare_dry_run_environment() {
    WORK_DIR="/tmp/torchtitan-npu-smoke.dry-run"
    REPORT_DIR="${WORK_DIR}/reports"
    TORCHTITAN_DIR="${WORK_DIR}/torchtitan"
    DSV32_TOKENIZER_DIR="${WORK_DIR}/tokenizers/deepseekv32"
    DSV4_TOKENIZER_DIR="${WORK_DIR}/tokenizers/deepseekv4"
}

_print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

_run_case() {
    local case_name="$1"
    local model="${case_name%%_*}"
    local feature="${case_name#*_}"
    local module
    local config
    local tokenizer_dir
    local overrides
    local ep_degree=1

    case "${model}" in
        dsv32)
            module="torchtitan_npu.models.deepseek_v3_2"
            config="deepseek_v3_2_debugmodel"
            tokenizer_dir="${DSV32_TOKENIZER_DIR}"
            overrides="${DSV32_OVERRIDES}"
            ;;
        dsv4)
            module="torchtitan_npu.models.deepseek_v4"
            config="deepseek_v4_debugmodel"
            tokenizer_dir="${DSV4_TOKENIZER_DIR}"
            overrides="${DSV4_OVERRIDES}"
            ;;
        *)
            echo "Unsupported model in case ${case_name}" >&2
            return 2
            ;;
    esac

    if [[ "${feature}" == "ep" ]]; then
        ep_degree="${NPROC_PER_NODE}"
    elif [[ "${feature}" != "fsdp" ]]; then
        echo "Unsupported feature in case ${case_name}" >&2
        return 2
    fi

    local -a train_args=(
        --module "${module}"
        --config "${config}"
        --hf-assets-path "${tokenizer_dir}"
        --dataloader.dataset c4_test
        --dataloader.dataset-path "${TORCHTITAN_DIR}/tests/assets/c4_test"
        --override.imports "${overrides}"
        --training.local-batch-size 1
        --training.seq-len "${SMOKE_SEQ_LEN}"
        --training.steps "${SMOKE_STEPS}"
        --parallelism.data-parallel-shard-degree "${NPROC_PER_NODE}"
        --parallelism.expert-parallel-degree "${ep_degree}"
        --checkpoint.no-enable
        --metrics.log-freq 1
        --dump-folder "${REPORT_DIR}/${case_name}/dump"
    )

    local pythonpath="${TORCHTITAN_DIR}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

    local -a command=(
        timeout --signal=TERM --kill-after=30 "${SMOKE_TIMEOUT_SECONDS}"
        env
        "PYTHONPATH=${pythonpath}"
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"
        "TORCHINDUCTOR_FX_GRAPH_CACHE=0"
        "TORCHINDUCTOR_AUTOGRAD_CACHE=0"
        "HCCL_CONNECT_TIMEOUT=300"
        "CLOSE_MATMUL_K_SHIFT=1"
        torchrun
        --nproc-per-node "${NPROC_PER_NODE}"
        --rdzv-backend c10d
        --rdzv-endpoint localhost:0
        --local-ranks-filter 0
        --role rank
        --tee 3
        -m torchtitan.train
        "${train_args[@]}"
    )

    echo "===== ${case_name} ====="
    _print_command "${command[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi

    local log_dir="${REPORT_DIR}/${case_name}"
    local log_file="${log_dir}/test.log"
    mkdir -p "${log_dir}"

    set +e
    "${command[@]}" 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    if [[ ${exit_code} -ne 0 ]]; then
        echo "${case_name} failed with exit code ${exit_code}" >&2
        return "${exit_code}"
    fi
    if grep -qiE 'loss[^[:alnum:]]*(nan|inf)' "${log_file}"; then
        echo "${case_name} produced a non-finite loss" >&2
        return 1
    fi
    echo "${case_name} passed"
}

main() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        _usage
        return
    fi
    if [[ "${1:-}" == "--list" ]]; then
        printf '%s\n' "${ALL_CASES[@]}"
        return
    fi

    _select_cases "$@"
    if [[ "${DRY_RUN}" == "1" ]]; then
        _prepare_dry_run_environment
    else
        _prepare_environment
    fi

    local -a failed_cases=()
    local item
    for item in "${SELECTED_CASES[@]}"; do
        if ! _run_case "${item}"; then
            failed_cases+=("${item}")
        fi
    done

    if [[ ${#failed_cases[@]} -gt 0 ]]; then
        printf 'Failed smoke cases: %s\n' "${failed_cases[*]}" >&2
        return 1
    fi
    echo "All selected smoke cases passed."
}

main "$@"
