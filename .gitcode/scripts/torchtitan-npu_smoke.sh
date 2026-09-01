#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

function LOG_DO() {
    local date_time
    local BPurple='\e[1;35m'
    local Purple='\e[0;35m'
    local Color_Off='\e[0m'
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}

function LOG_HEAD() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo "========================================================"
    echo "${date_time} : $*"
    echo "========================================================"
}

cd "${WORKSPACE}" || exit 1
ls -l
LOG_HEAD "Build torchtitan-npu."
gcc --version
python3 --version

set +e
LOG_DO bash .ci/smoke_test.sh
ret=$?
set -e

if [ $ret -ne 0 ]; then
    echo "Lint failed with exit code $ret"
    exit $ret
fi

LOG_HEAD "Build torchtitan-npu success."
