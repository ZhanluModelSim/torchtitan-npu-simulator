# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

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

readonly TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://gitcode.com/GitHub_Trending/to/torchtitan.git}"
TORCHTITAN_REQUIREMENT="$(grep -E '^torchtitan .*@[0-9a-f]{40}$' "${PROJECT_ROOT}/requirements.txt")"
readonly TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-${TORCHTITAN_REQUIREMENT##*@}}"
[[ "${TORCHTITAN_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid TorchTitan commit: ${TORCHTITAN_COMMIT}" >&2; exit 1; }
readonly TOKENIZER_REPO="${DEEPSEEK_TOKENIZER_REPO:-https://gitcode.com/hitwdy/deepseekv4.git}"
readonly NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-4}"
readonly SMOKE_STEPS="${SMOKE_STEPS:-5}"
readonly SMOKE_SEQ_LEN="${SMOKE_SEQ_LEN:-4096}"
readonly SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-900}"
readonly DRY_RUN="${SMOKE_DRY_RUN:-0}"

readonly -a ALL_CASES=(
    deepseek_v3_2_ep2
    deepseek_v3_2_cp2_ep2
    deepseek_v4_ep2
)

# The v3.2 fused path: TND mask handler + fused attention (sparse_attn
# package), the CANN fused complex rope, and the CANN RMSNorm.  The legacy
# npu_*_override / single-complex names are gone (the unified rope API).
readonly -a DSV32_OVERRIDES=(
    torchtitan_npu.override.deepseek_v3_2.sparse_attn.cann_metadata
    torchtitan_npu.override.deepseek_v3_2.sparse_attn.cann
    torchtitan_npu.override.common.rope.cann_complex
    torchtitan_npu.override.common.rms_norm.cann
)

# The v4 fused path (debugmodel geometry, kept in sync with run_train.sh):
# CANN RMSNorm + fused rope, the CANN metadata layer (geometry JSON — a
# single --override.imports argument, hence the array form), and the fused
# TND attention.
readonly DSV4_GEOMETRY='{"num_heads":16,"head_dim":512,"index_n_heads":8,"index_head_dim":128,"index_topk":512}'
readonly -a DSV4_OVERRIDES=(
    torchtitan_npu.override.common.rms_norm.cann
    torchtitan_npu.override.common.rope.cann_complex
    "torchtitan_npu.override.deepseek_v4.sparse_attn.cann_metadata=${DSV4_GEOMETRY}"
    torchtitan_npu.override.deepseek_v4.sparse_attn.cann
)

WORK_DIR=""
TORCHTITAN_DIR=""
REPORT_DIR=""
DSV32_TOKENIZER_DIR=""
DSV4_TOKENIZER_DIR=""

_usage() {
    cat <<'EOF'
Usage: .ci/smoke_test.sh [-s] [--list] [CASE ...]

Cases:
  deepseek_v3_2_ep2          DeepSeek-V3.2 FSDP + EP2
  deepseek_v3_2_cp2_ep2      DeepSeek-V3.2 FSDP + CP2 + EP2
  deepseek_v4_ep2            DeepSeek-V4 FSDP + EP2

With no CASE, all cases run. The default workload uses 4 NPUs, a sequence
length of 4096, and 5 training steps. Set SMOKE_DRY_RUN=1 to only print the
generated torchrun commands. The -s option is accepted for CI runner
compatibility and has no effect.
EOF
}

_contains_case() {
    local expected="$1"
    local item
    for item in "${ALL_CASES[@]}"; do
        [[ "${item}" == "${expected}" ]] && return 0
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
    local env_script="/usr/local/Ascend/ascend-toolkit/set_env.sh"
    if [[ ! -f "${env_script}" ]]; then
        echo "CANN environment script not found: ${env_script}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${env_script}"
}

_clone_at_commit() {
    local repo="$1"
    local commit="$2"
    local target="$3"

    git -c init.defaultBranch=main init --quiet "${target}"
    git -C "${target}" remote add origin "${repo}"
    git -C "${target}" fetch --filter=blob:none --depth 1 origin "${commit}"
    git -C "${target}" checkout --detach --quiet FETCH_HEAD
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
    git -C "${tokenizer_source}" checkout --quiet

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

_prepare_environment() {
    _source_cann_env
    export PATH="${PYTHON_SHIM_DIR}:${PATH}"

    python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        python -c 'import torch, torch_npu, torchtitan, torchtitan_npu'

    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-smoke.XXXXXX")"
    TORCHTITAN_DIR="${WORK_DIR}/torchtitan"
    REPORT_DIR="${SMOKE_REPORT_DIR:-${PROJECT_ROOT}/test_reports/smoke}"
    mkdir -p "${REPORT_DIR}"

    _clone_at_commit "${TORCHTITAN_REPO}" "${TORCHTITAN_COMMIT}" "${TORCHTITAN_DIR}"
    _prepare_tokenizers
}

_prepare_dry_run_environment() {
    WORK_DIR="/tmp/torchtitan-npu-smoke.dry-run"
    TORCHTITAN_DIR="${WORK_DIR}/torchtitan"
    REPORT_DIR="${WORK_DIR}/reports"
    DSV32_TOKENIZER_DIR="${WORK_DIR}/tokenizers/deepseekv32"
    DSV4_TOKENIZER_DIR="${WORK_DIR}/tokenizers/deepseekv4"
}

_wait_npu_idle() {
    local attempts="${SMOKE_IDLE_WAIT_ATTEMPTS:-10}"
    local threshold_mb="${SMOKE_IDLE_THRESHOLD_MB:-5000}"
    local attempt used

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        used="$(npu-smi info 2>/dev/null | grep -oP '\\d+(?=\\s+/\\s+\\d+)' | sort -n | tail -1 || true)"
        used="${used:-0}"
        if [[ "${used}" -lt "${threshold_mb}" ]]; then
            echo "NPU is ready (maximum HBM usage: ${used}MB)"
            return
        fi
        echo "Waiting for NPU memory to free (${used}MB, ${attempt}/${attempts})"
        sleep 1
    done

    echo "NPU memory did not fall below ${threshold_mb}MB" >&2
    return 1
}

_print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

_run_case() {
    local case_name="$1"
    local module config tokenizer_dir
    local -a overrides=()
    local cp_degree=1
    local ep_degree=2
    local -a spmd_args=()

    case "${case_name}" in
        deepseek_v3_2_*)
            module="torchtitan_npu.models.deepseek_v3_2"
            config="deepseek_v3_2_debugmodel"
            tokenizer_dir="${DSV32_TOKENIZER_DIR}"
            overrides=("${DSV32_OVERRIDES[@]}")
            ;;
        deepseek_v4_*)
            module="torchtitan_npu.models.deepseek_v4"
            config="deepseek_v4_debugmodel"
            tokenizer_dir="${DSV4_TOKENIZER_DIR}"
            overrides=("${DSV4_OVERRIDES[@]}")
            ;;
        *)
            echo "Unsupported model in case ${case_name}" >&2
            return 2
            ;;
    esac

    case "${case_name}" in
        deepseek_v3_2_ep2 | deepseek_v4_ep2)
            ;;
        deepseek_v3_2_cp2_ep2)
            cp_degree=2
            spmd_args=(--parallelism.spmd-backend spmd_types)
            ;;
        *)
            echo "Unsupported parallelism in case ${case_name}" >&2
            return 2
            ;;
    esac

    if ((NPROC_PER_NODE % cp_degree != 0 || NPROC_PER_NODE % ep_degree != 0)); then
        echo "NPU count ${NPROC_PER_NODE} is incompatible with CP${cp_degree}/EP${ep_degree}" >&2
        return 2
    fi
    local dp_shard_degree=$((NPROC_PER_NODE / cp_degree))

    # ``--override.imports`` is a single tyro nargs="*" flag: repeated flags
    # do NOT append (only the last survives).  One flag carries all targets —
    # the plain ones comma-joined into one token, the kwargs target (the v4
    # geometry JSON, containing commas/quotes) as its own token.
    local -a override_tokens=()
    local -a plain_overrides=()
    local ov
    for ov in "${overrides[@]}"; do
        if [[ "${ov}" == *"="* ]]; then
            override_tokens+=("${ov}")
        else
            plain_overrides+=("${ov}")
        fi
    done
    if [[ ${#plain_overrides[@]} -gt 0 ]]; then
        override_tokens=("$(IFS=,; echo "${plain_overrides[*]}")" "${override_tokens[@]}")
    fi
    local -a override_args=(--override.imports "${override_tokens[@]}")

    local -a train_args=(
        --module "${module}"
        --config "${config}"
        --hf-assets-path "${tokenizer_dir}"
        --dataloader.dataset c4_test
        --dataloader.dataset-path "${TORCHTITAN_DIR}/tests/assets/c4_test"
        --training.local-batch-size 1
        --training.seq-len "${SMOKE_SEQ_LEN}"
        --training.steps "${SMOKE_STEPS}"
        --parallelism.data-parallel-shard-degree "${dp_shard_degree}"
        --parallelism.context-parallel-degree "${cp_degree}"
        --parallelism.expert-parallel-degree "${ep_degree}"
        "${spmd_args[@]}"
        "${override_args[@]}"
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
        python
        -m torch.distributed.run
        --nproc-per-node "${NPROC_PER_NODE}"
        --rdzv-backend c10d
        --rdzv-endpoint localhost:0
        --local-ranks-filter 0
        --role rank
        --tee 3
        -m torchtitan.train
        "${train_args[@]}"
    )

    echo "==================== Test case: ${case_name} ===================="
    _print_command "${command[@]}"
    [[ "${DRY_RUN}" == "1" ]] && return

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
    if [[ "${1:-}" == "-s" ]]; then
        shift
    fi

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
        _wait_npu_idle
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
    echo "smoke test passed."
}

main "$@"
