#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export CONFIG="${CONFIG:-graph_trainer_deepseek_v4_flash_43layers_16experts}"
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
    --parallelism.spmd-backend default \
    "$@"
