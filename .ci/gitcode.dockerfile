# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This Dockerfile is used only to build CI images.

# for smoke
FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu24.04_arm:lv6_v1.1095
# for ut and lint
# FROM swr.cn-north-4.myhuaweicloud.com/ci_cann/ubuntu24.04_x86_64:lv6_v1.1095

ENV PATH="/usr/local/python/python312/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/python/python312/lib:${LD_LIBRARY_PATH}"

RUN mkdir -p /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "index-url=https://repo.huaweicloud.com/repository/pypi/simple" >> /root/.pip/pip.conf \
    && echo "trusted-host=repo.huaweicloud.com" >> /root/.pip/pip.conf \
    && echo "timeout=120" >> /root/.pip/pip.conf

RUN python3 -m pip install --no-cache-dir \
    --index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
    --trusted-host mirrors.huaweicloud.com \
    esdk-obs-python

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
