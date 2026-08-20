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
# Optional graph-pattern compile controls:
#
#   TORCHINDUCTOR_NPU_EXT_DEBUG=allfallback \
#   PATTERN_IMPORTS=torchtitan_npu.compile.patterns.deepseek_v4.inplace_partial_rope \
#   COMPILE_BACKEND=inductor ./scripts/run_train.sh
#
# Profiling is off by default (--profiler.no-enable-profiling); enable it
# explicitly with ENABLE_PROFILING=1 (optionally combined with the
# PROFILE_START/PROFILE_END window below):
#
#   ENABLE_PROFILING=1 PROFILE_START=5 PROFILE_END=6 ./scripts/run_train.sh
#
# Keep GOLDEN_OVERRIDES unchanged. Edit TEST_OVERRIDES for the implementation
# under test.
#
# NOTE: MODULE must resolve through ``torchtitan_npu`` so the package is
# imported early (its __init__ activates the patches/torchtitan backports);
# if it were imported only at apply_override time the patches would come in
# too late.

set -e

if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /home/developer/Ascend/ascend-toolkit/set_env.sh
fi

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.deepseek_v4"}
CONFIG=${CONFIG:-"deepseek_v4_debugmodel"}
COMM_MODE=${COMM_MODE:-""}
CP_DEGREE=${CP_DEGREE:-0}
CP_LOAD_BALANCER=${CP_LOAD_BALANCER:-"headtail"}
export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}
export TORCHTITAN_NPU_PATTERN_IMPORTS="${PATTERN_IMPORTS:-${TORCHTITAN_NPU_PATTERN_IMPORTS:-}}"

# The AscendC mask handler needs the attention geometry from the selected model
# config. Keep these values in sync with the DeepSeek-V4 config registry.
case "${CONFIG}" in
    deepseek_v4_debugmodel)
        MASK_HANDLER_GEOMETRY='{"num_heads":16,"head_dim":512,"index_n_heads":8,"index_head_dim":128,"index_topk":512}'
        ;;
    deepseek_v4_flash)
        MASK_HANDLER_GEOMETRY='{"num_heads":64,"head_dim":512,"index_n_heads":64,"index_head_dim":128,"index_topk":512}'
        ;;
    deepseek_v4_pro)
        MASK_HANDLER_GEOMETRY='{"num_heads":128,"head_dim":512,"index_n_heads":64,"index_head_dim":128,"index_topk":1024}'
        ;;
    *)
        MASK_HANDLER_GEOMETRY=""
        ;;
esac

# Fixed numerical reference: Torch RMSNorm from the model config, the torch-
# compatible DSV4 rope (unconditional YaRN, patched generic path), the normal
# bf16 MoE (clamped via the GroupedExperts/FeedForward patch — the same
# dtype and clamp semantics as transformers), and the eager Golden
# sparse-attention implementation.
readonly -a GOLDEN_OVERRIDES=(
    torchtitan_npu.override.common.rope.workaround
    torchtitan_npu.override.deepseek_v4.sparse_attn.golden
)

# Fused implementation under test: AscendC RMSNorm, AscendC fused rope, mask
# metadata, and SMLA. Both paths use the model's normal (clamped) MoE.
readonly -a TEST_OVERRIDES=(
    torchtitan_npu.override.common.rms_norm.asc
    torchtitan_npu.override.common.rope.asc_complex
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata=${MASK_HANDLER_GEOMETRY}"
    torchtitan_npu.override.deepseek_v4.sparse_attn.asc
)

if [ "${USE_GOLDEN:-0}" = "1" ]; then
    OVERRIDE_IMPORTS=("${GOLDEN_OVERRIDES[@]}")
else
    if [ -z "${MASK_HANDLER_GEOMETRY}" ]; then
        echo "Unsupported CONFIG='${CONFIG}' for fused DSV4 overrides" >&2
        exit 2
    fi
    OVERRIDE_IMPORTS=("${TEST_OVERRIDES[@]}")
fi

if [ -n "${PROFILER_OVERRIDE:-}" ]; then
    OVERRIDE_IMPORTS+=("${PROFILER_OVERRIDE}")
fi

# ``--override.imports`` is a single tyro nargs="*" flag: repeated flags do
# NOT append (only the last one survives).  One flag carries all targets —
# the plain ones comma-joined into a single token, the kwargs target (the
# AscendC mask-handler geometry JSON) as its own token.
_override_tokens=()
_plain_overrides=()
for _ov in "${OVERRIDE_IMPORTS[@]}"; do
    if [[ "${_ov}" == *"="* ]]; then
        _override_tokens+=("${_ov}")
    else
        _plain_overrides+=("${_ov}")
    fi
done
if [[ ${#_plain_overrides[@]} -gt 0 ]]; then
    _override_tokens=("$(IFS=,; echo "${_plain_overrides[*]}")" "${_override_tokens[@]}")
fi

ARGS=(
    --module "${MODULE}"
    --config "${CONFIG}"
    --hf-assets-path "${HF_ASSETS_PATH:-/path/to/dsv4_tokenizer}"
    --dataloader.dataset "${DATASET:-c4_test}"
    --dataloader.dataset-path "${DATASET_PATH:-tests/assets/c4_test}"
    --override.imports "${_override_tokens[@]}"
)

# An absolute profiling window is expressed with TorchTitan's native profiler
# schedule.  For example, PROFILE_START=5 PROFILE_END=6 PROFILE_WARMUP=3
# becomes skip_first=1, warmup=3, active=1, repeat=1.  The CANN override only
# supplies CANN-specific options; it does not need a second step scheduler.
ENABLE_PROFILING=${ENABLE_PROFILING:-0}
if [ "${ENABLE_PROFILING}" = "1" ]; then
    ARGS+=(--profiler.enable-profiling)
else
    ARGS+=(--profiler.no-enable-profiling)
fi

if [ -n "${PROFILE_START:-}" ] || [ -n "${PROFILE_END:-}" ]; then
    if [[ ! "${PROFILE_START:-}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${PROFILE_END:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROFILE_START and PROFILE_END must be positive integers" >&2
        exit 2
    fi
    PROFILE_WARMUP=${PROFILE_WARMUP:-3}
    if [[ ! "${PROFILE_WARMUP}" =~ ^[0-9]+$ ]]; then
        echo "PROFILE_WARMUP must be a non-negative integer" >&2
        exit 2
    fi
    if [ "${PROFILE_END}" -le "${PROFILE_START}" ]; then
        echo "PROFILE_END must be greater than PROFILE_START" >&2
        exit 2
    fi

    PROFILE_SKIP_FIRST=$(( PROFILE_START > PROFILE_WARMUP ? PROFILE_START - PROFILE_WARMUP - 1 : 0 ))
    PROFILE_WARMUP_STEPS=$(( PROFILE_START - 1 - PROFILE_SKIP_FIRST ))
    PROFILE_ACTIVE=$(( PROFILE_END - PROFILE_START ))
    PROFILE_FREQ=$(( PROFILE_WARMUP_STEPS + PROFILE_ACTIVE ))
    ARGS+=(
        --profiler.profile-freq "${PROFILE_FREQ}"
        --profiler.profiler-warmup "${PROFILE_WARMUP_STEPS}"
        --profiler.profiler-active "${PROFILE_ACTIVE}"
        --profiler.profiler-repeat 1
        --profiler.profiler-skip-first "${PROFILE_SKIP_FIRST}"
    )
fi

if [ -n "${COMPILE_BACKEND:-}" ]; then
    ARGS+=(
        --compile.enable
        --compile.components model
        --compile.backend "${COMPILE_BACKEND}"
    )
fi

# Context parallel (fused path only): CP_DEGREE=2 etc. enables the CP
# parallelism flags + the spmd_types backend (the ShardingConfig emits the
# (idx_k, cmp_k) all-gather; the CP flow itself lives in the model dir).
# seq_len must divide by the load-balancer divisibility rule (2 * cp for
# headtail).
if [ "${CP_DEGREE:-0}" -gt 1 ]; then
    ARGS+=(
        --parallelism.context_parallel_degree "${CP_DEGREE}"
        --parallelism.context_parallel_load_balancer "${CP_LOAD_BALANCER}"
        --parallelism.spmd_backend spmd_types
    )
fi

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
