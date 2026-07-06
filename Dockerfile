# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Dockerfile for torchtitan-npu-simulator
#
# Builds a self-contained image that can run the simulator without real NPU
# hardware.  Based on the CANN image (provides libhccl.so etc. for torch_npu
# meta-kernel registration), with torch (CPU), torch_npu, torchtitan (pinned
# commit), and torchtitan_npu (simulator branch) installed on top.
#
# Supports both x86_64 and arm64 (aarch64) architectures.
#
# Usage:
#   docker build -t torchtitan-npu-simulator:v1.0 .
#   docker run -d --name titan-sim \
#     -v $(pwd):/workspace -w /workspace \
#     torchtitan-npu-simulator:v1.0 sleep infinity
#   docker exec -it titan-sim bash
#   # Inside container:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   NGPU=384 python3 -m torchtitan_npu.entry \
#       --module torchtitan_npu.simulator \
#       --config deepseek_v4_pro_simulate_61_layers \
#       --comm.mode=fake_backend --training.steps=1 \
#       --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer

# ------------------------------------------------------------------------------
# Stage 1: CANN base (provides libhccl.so, libascend_hal.so, ATB, Python 3.12)
# ------------------------------------------------------------------------------
# This image is multi-arch (amd64 + arm64), so the same Dockerfile works on
# both x86 servers and ARM64 machines (e.g. Kunpeng / Ascend servers).
FROM quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12

# ------------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# ------------------------------------------------------------------------------
# The CANN image ships Python 3.12 at /usr/local/python3.12.13/bin/python3.
# pip is already available.  We install torch (CPU-only build, no CUDA), 
# torch_npu (provides NPU op meta-kernels for meta-device simulation),
# torchtitan (pinned to the exact commit torchtitan-npu depends on),
# and the simulator's own requirements.

# Install system build tools (some Python packages need compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU) + torch_npu first (large wheels, rarely change)
# The --extra-index-url is needed for the +cpu variant of torch.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.12.0+cpu" \
        "torch_npu==2.12.0rc1"

# Install torchtitan at the pinned commit (torchtitan-npu's requirements.txt
# pins this exact SHA; using a different commit may break API compatibility).
RUN pip install --no-cache-dir \
        "torchtitan @ git+https://gitcode.com/GitHub_Trending/to/torchtitan.git@ac13e536c84e7f6647b14fa9375c3c8a8a2b8578"

# Install remaining Python dependencies from requirements.txt
# (numpy, PyYAML, pybind11, ninja, triton-ascend, scipy, safetensors, torchao)
# We do this in a separate layer so changes to requirements.txt don't
# re-download the large torch/torch_npu wheels.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ------------------------------------------------------------------------------
# Stage 3: Install torchtitan_npu (simulator branch)
# ------------------------------------------------------------------------------
# Clone the repo and checkout the simulator branch, then install in editable
# mode so the simulator code is directly editable inside the container.
# The repo is cloned to /opt/torchtitan-npu-simulator so it persists in the
# image; users mount their own working copy at /workspace for development.

ARG SIM_BRANCH=feat/npu-simulator
RUN git clone --branch ${SIM_BRANCH} --depth 1 \
        https://github.com/ZhanluModelSim/torchtitan-npu-simulator.git \
        /opt/torchtitan-npu-simulator \
    && cd /opt/torchtitan-npu-simulator \
    && pip install --no-cache-dir -e .

# ------------------------------------------------------------------------------
# Stage 4: Environment configuration
# ------------------------------------------------------------------------------
# The CANN set_env.sh script configures LD_LIBRARY_PATH, PYTHONPATH, PATH,
# ASCEND_TOOLKIT_HOME, etc.  We source it in the shell profile so every
# interactive bash session has the CANN environment loaded automatically.
# This mirrors what was done manually in the titan-npu-sim-e2e container.

# Auto-load CANN environment on every bash session
RUN echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null' >> /etc/bash.bashrc \
    && echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null' >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
