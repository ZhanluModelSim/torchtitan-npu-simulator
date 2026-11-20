#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODULE="${MODULE:-torchtitan_npu_recipes.models.qwen3_5}"
NGPU="${NGPU:-16}"
TRAIN_FILE=torchtitan_npu_recipes.qwen_sft

usage() {
    cat <<'HELP'
Usage: bash examples/qwen3_5/run_qwen_sft.sh --hf-assets-path DIR --dataset-path FILE --output-dir DIR
Presets: --config run_qwen_sft (default), run_qwen3_5_32k_cp8, run_qwen3_5_64k_cp8, run_qwen3_5_128k_cp8
Optional FLA 32K preset: --config run_qwen3_5_32k_cp8_fla
Custom recipe: --module FILE.py --config FUNCTION
Data: JSONL with messages. Optional: --chat-template FILE, --loss-mask POLICY
Training: --smoke-test, --max-steps N, --resume-step N, --init-checkpoint DIR
HELP
}

data_override() {
    python3 - "$@" <<'PY'
import json
from pathlib import Path
import sys

dataset, template, mask = sys.argv[1:]
if not dataset or not Path(dataset).is_file():
    sys.exit("--dataset-path must name an existing JSONL file")
options = {"dataset_path": str(Path(dataset).resolve())}
if template:
    if not Path(template).is_file():
        sys.exit("--chat-template must name an existing Jinja file")
    options["chat_template"] = str(Path(template).resolve())
if mask:
    options["loss_mask"] = mask
print(json.dumps(options))
PY
}

main() {
    local dataset= assets= template= mask= option value kwargs
    local -a args=() data_args=()
    while (( $# )); do
        case "${1%%=*}" in
            --help|-h) usage; return ;;
            --print-config) export COMM_BACKEND=fake; args+=("$1") ;;
            --module|--dataset-path|--hf-assets-path|--chat-template|--loss-mask|--init-checkpoint)
                option="${1%%=*}"
                if [[ "$1" == *=* ]]; then
                    value="${1#*=}"
                else
                    value="${2:-}"
                    if (( $# > 1 )); then shift; fi
                fi
                if [[ -z "$value" || "$value" == --* ]]; then
                    echo "$option requires a value" >&2
                    return 2
                fi
                case "$option" in
                    --dataset-path) dataset="$value" ;;
                    --hf-assets-path) assets="$value" ;;
                    --chat-template) template="$value" ;;
                    --loss-mask) mask="$value" ;;
                    --module|--init-checkpoint)
                        if [[ "$value" != /* && ( "$option" == --init-checkpoint || "$value" == *.py ) ]]; then
                            value="$PWD/$value"
                        fi
                        args+=("$option" "$value")
                        ;;
                esac
                ;;
            *) args+=("$1") ;;
        esac
        shift
    done

    if [[ -n "$dataset$assets$template$mask" ]]; then
        dataset="${dataset:-${DATA_FILES:-}}"
        assets="${assets:-${HF_ASSETS_PATH:-}}"
        if [[ ! -d "$assets" ]]; then
            echo '--hf-assets-path must name an existing model directory' >&2
            return 2
        fi
        export HF_ASSETS_PATH="$(cd -- "$assets" && pwd)"
        unset DATA_FILES
        kwargs=$(data_override "$dataset" "$template" "$mask")
        data_args=(--override "torchtitan_npu.models.qwen3_5.dataloader.chat_sft=$kwargs")
    fi

    CONFIG="${CONFIG:-run_qwen_sft}"
    export MODULE CONFIG NGPU TRAIN_FILE
    exec bash "$SCRIPT_DIR/../../scripts/run_train.sh" "${data_args[@]}" "${args[@]}"
}

main "$@"
