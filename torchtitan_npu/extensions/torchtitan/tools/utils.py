# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ascend peak-FLOPS support for TorchTitan metrics.

TorchTitan's device table only covers NVIDIA/AMD/Intel/TPU/Neuron parts, so
Ascend devices fall back to the A100 peak (312 TFLOPS) and skew the reported
MFU. This module hooks ``torchtitan.tools.utils.get_peak_flops`` (the single
utility seam): known Ascend device names map to their Cube (dense BF16) peak
FLOPS, everything else delegates to the original implementation.

``MetricsProcessor`` resolves the peak-FLOPS value through this seam at
construction time, so installing the hook is sufficient -- no other part of
``MetricsProcessor`` is touched. Importing this module applies the hook.
"""

from torchtitan.tools import utils as titan_utils

_upstream_get_peak_flops = titan_utils.get_peak_flops

# MFU uses the BF16 Cube peak rather than the larger total-throughput figure.
_ASCEND_BF16_PEAK_FLOPS = {
    "Ascend910_9362": 353.8944e12,  # total: 376T Cube: 353T (A3 SoC)
    "Ascend910_9392": 353.8944e12,  # total: 376T Cube: 353T
    "Ascend910B1": 373.88e12,  # total: 400T Cube: 373T
    "Ascend910B2": 353.8944e12,  # total: 376T Cube: 353T
    "Ascend910B3": 294.912e12,  # total: 313T Cube: 294T
    "Ascend910B4": 245.76e12,  # total: 280T Cube: 245T
    "Ascend950PR": 432e12,  # total: 486T Cube: 432T (whitepaper Tab 3-1, 32 cube)
    "Ascend950PR_9582": 432e12,  # same as 950PR
    "Ascend950DT_9591": 486e12,  # total: 547T Cube: 486T (whitepaper Tab 3-1, 36 cube)
    "Ascend950DT_9582": 432e12,  # total: 486T Cube: 432T (32 cube)
}


def get_peak_flops(device_name: str) -> float:
    for model, peak_flops in _ASCEND_BF16_PEAK_FLOPS.items():
        if model in device_name:
            return peak_flops
    return _upstream_get_peak_flops(device_name)


titan_utils.get_peak_flops = get_peak_flops
