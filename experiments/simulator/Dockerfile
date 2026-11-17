# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# BASE_TRAINING_IMAGE must already contain the complete official CANN
# environment and the chosen TorchTitan / NPU dependencies.
ARG BASE_TRAINING_IMAGE
FROM ${BASE_TRAINING_IMAGE}
RUN rm -rf /opt/torchtitan-simulator
COPY pyproject.toml LICENSE /opt/torchtitan-simulator/
COPY torchtitan_simulator /opt/torchtitan-simulator/torchtitan_simulator
COPY scripts /opt/torchtitan-simulator/scripts
COPY README.md /opt/torchtitan-simulator/
COPY docs /opt/torchtitan-simulator/docs/
RUN python -m pip install --no-deps --no-build-isolation --force-reinstall /opt/torchtitan-simulator
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV CANN_ENV_FILE=/usr/local/Ascend/cann/set_env.sh
ENTRYPOINT ["bash", "/opt/torchtitan-simulator/scripts/container-entry.sh"]
CMD ["--help"]
