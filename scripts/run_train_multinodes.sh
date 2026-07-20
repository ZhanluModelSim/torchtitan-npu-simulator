# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ps -ef |grep -i python |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i torchrun |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i ray |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef |grep -i vllm |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}

# NOTE: Source the CANN env scripts in your shell before running this script.
# See docs/user-guides/quickstart.md "配置 CANN 环境变量".

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32
export MULTI_STREAM_MEMORY_RESERVE=1

Network_Interface=${Network_Interface:-$(ip -o -4 addr show scope global | awk '$2 != "docker0" {print $2; exit}')}
export GLOO_SOCKET_IFNAME=${Network_Interface}
export HCCL_SOCKET_IFNAME=${Network_Interface}
export HCCL_IF_BASE_PORT=30000

export LOG_RANK=${LOG_RANK:-0}  # rank to show log
export PYTHONUNBUFFERED=1

LOCAL_HOST=${LOCAL_HOST:-$(ip addr show "${Network_Interface}" | grep "inet " | awk '{print $2}' | cut -d'/' -f1 | head -n1)}
LOCAL_HOST=${LOCAL_HOST:-$(hostname -I | awk '{print $1}')}
echo $LOCAL_HOST
if [[ -n "${NODE_IPS:-}" ]]; then
    read -r -a IPs <<< "${NODE_IPS//,/ }"
else
    IPs=("${LOCAL_HOST}")
fi
NGPU=${NGPU:-8}
NPUS_PER_NODE=${NGPU}
MASTER_ADDR=${MASTER_ADDR:-${IPs[0]}}
MASTER_PORT=${MASTER_PORT:-6300}
NNODES=${NNODES:-${#IPs[@]}}
NODE_RANK=${NODE_RANK:-}
for i in "${!IPs[@]}";
do
    if [[ "$LOCAL_HOST" == "${IPs[$i]}" ]];
    then
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done
if [[ $NODE_RANK == "" ]];then
    echo "[Error] Variable \"NODE_RANK\" must be configured"
    exit 1
fi
set -exo pipefail

RDZV_ID="dsv32_train_$(date +%Y%m%d)"
MODULE=${MODULE:-"torchtitan_npu.models.deepseek_v32"}
CONFIG=${CONFIG:-"deepseek_v32_671b_128npus"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan_npu.entry"}
time=$(date +%Y%m%d%H%M)
logfile=dsv32_128die_${time}_node${NODE_RANK}_${LOCAL_HOST//./_}.log
mkdir -p logs


TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${NPUS_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@" 2>&1 | tee -a logs/${logfile}
