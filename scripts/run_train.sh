#!/usr/bin/bash
# usage:
#
#   NGPU=1 CONFIG=deepseek_v4_debugmodel ./scripts/run_train.sh --training.steps 5
#
# steps/seq_len/local_batch_size come from the config; override them with flags.
#
# Comparing the fixed Golden baseline with the test overrides:
#
#   USE_GOLDEN=1 ./scripts/run_train.sh  # Golden reference
#   ./scripts/run_train.sh               # Test configuration (default)
#
# Keep GOLDEN_OVERRIDES unchanged. Edit TEST_OVERRIDES for the implementation
# under test.

set -e

source /usr/local/Ascend/cann/set_env.sh

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.deepseek_v4"}
CONFIG=${CONFIG:-"deepseek_v4_debugmodel"}
COMM_MODE=${COMM_MODE:-""}
export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# Fixed numerical reference. Do not modify this list.
readonly GOLDEN_OVERRIDES="\
torchtitan_npu.override.deepseek_v4.golden.dsa_sparse_attention_golden,\
torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_rope_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_single_rope_override,\
torchtitan_npu.override.deepseek_v4.golden.rms_norm_golden,\
torchtitan_npu.override.deepseek_v4.golden.dsv4_moe_golden"

# Edit this list to select the implementation under test.
TEST_OVERRIDES="\
torchtitan_npu.override.deepseek_v4.fused_dsa.npu_smla_tnd_override,\
torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_rope_override,\
torchtitan_npu.override.deepseek_v4.rope.npu_dsv4_single_rope_override,\
torchtitan_npu.override.deepseek_v4.golden.rms_norm_golden"

if [ "${USE_GOLDEN:-0}" = "1" ]; then
    OVERRIDE_IMPORTS="${GOLDEN_OVERRIDES}"
else
    OVERRIDE_IMPORTS="${TEST_OVERRIDES}"
fi

ARGS=(
    --module "${MODULE}"
    --config "${CONFIG}"
    --hf-assets-path "${HF_ASSETS_PATH:-/path/to/dsv4_tokenizer}"
    --dataloader.dataset "${DATASET:-c4_test}"
    --dataloader.dataset-path "${DATASET_PATH:-tests/assets/c4_test}"
    --override.imports "${OVERRIDE_IMPORTS}"
)

if [ -n "$COMM_MODE" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train \
        "${ARGS[@]}" --comm.mode=${COMM_MODE} "$@"
else
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CPU_AFFINITY_CONF=1 \
    TASK_QUEUE_ENABLE=2 \
    HCCL_CONNECT_TIMEOUT=3600 \
    STREAMS_PER_DEVICE=32 \
    MULTI_STREAM_MEMORY_RESERVE=1 \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train "${ARGS[@]}" "$@"
fi
