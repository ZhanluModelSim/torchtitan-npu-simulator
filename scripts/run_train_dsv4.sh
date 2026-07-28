#!/usr/bin/bash
# usage:
#
#   NGPU=1 CONFIG=deepseek_v4_debugmodel ./scripts/run_train_dsv4.sh --training.steps 5
#
# steps/seq_len/local_batch_size come from the config; override them with flags.
#
# PICKING THE DSA kernel (the usual loss/grad_norm precision comparison):
#
#   ATTENTION=golden   ./scripts/run_train_dsv4.sh   # Golden reference (default)
#   ATTENTION=smla ./scripts/run_train_dsv4.sh       # CANN fused TND kernel
#
# OVERRIDE_IMPORTS, if set explicitly, wins over ATTENTION and is the full-control
# escape hatch; set it empty to disable all overrides.

set -ex

source /usr/local/Ascend/ascend-toolkit/set_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# [TODO] change to your path
TORCHTITAN_DIR=${TORCHTITAN_DIR:-"/path/to/torchtitan"}

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.deepseek_v4"}
CONFIG=${CONFIG:-"deepseek_v4_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

# Non-attention overrides that bit-match the dsv4-infer-npu inference baseline
BASE_OVERRIDES="\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_rope_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_single_rope_override,\
torchtitan_npu.override.deepseek_v4.golden.rms_norm_golden,\
torchtitan_npu.override.deepseek_v4.golden.dsv4_moe_golden"

MASK_HANDLER_OVERRIDE="torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override"
ATTENTION=${ATTENTION:-golden}
case "${ATTENTION}" in
    golden)   ATTN_OVERRIDE="torchtitan_npu.override.deepseek_v4.golden.dsa_sparse_attention_golden" ;;
    smla) ATTN_OVERRIDE="torchtitan_npu.override.deepseek_v4.fused_dsa.npu_smla_tnd_override" ;;
    *) echo "unknown ATTENTION='${ATTENTION}' (want: golden | smla)" >&2; exit 2 ;;
esac
ATTN_OVERRIDE="${ATTN_OVERRIDE},${MASK_HANDLER_OVERRIDE}"
# ${VAR-default} (no colon) so an explicit empty value disables all overrides.
OVERRIDE_IMPORTS=${OVERRIDE_IMPORTS-"${ATTN_OVERRIDE},${BASE_OVERRIDES}"}

export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}

export PYTHONPATH="${TORCHTITAN_DIR}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
    --module "${MODULE}"
    --config "${CONFIG}"
    --hf-assets-path "${HF_ASSETS_PATH:-/data/tokenizer/dsv4_tokenizer}"
    --dataloader.dataset "${DATASET:-c4_test}"
    --dataloader.dataset-path "${DATASET_PATH:-${TORCHTITAN_DIR}/tests/assets/c4_test}"
)
if [ -n "${OVERRIDE_IMPORTS}" ]; then
    ARGS+=(--override.imports "${OVERRIDE_IMPORTS}")
fi

if [ -n "${COMM_MODE}" ]; then
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train \
        "${ARGS[@]}" --comm.mode="${COMM_MODE}" "$@"
else
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m torchtitan.train "${ARGS[@]}" "$@"
fi
