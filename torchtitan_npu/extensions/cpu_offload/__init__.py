# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU offload: keep canonical weights, gradients, and optimizer state on the
host, execute updates and reductions on the NPU.
"""
