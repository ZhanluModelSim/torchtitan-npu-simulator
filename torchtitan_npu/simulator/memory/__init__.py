# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static tensor-liveness memory model for simulator captures."""

from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.records import (
    FSDPResidencyEvent,
    MemoryPlan,
    MemoryTimelineEvent,
    RawMemoryEvent,
    TensorLifetime,
    TensorRef,
)

__all__ = [
    "FSDPResidencyEvent",
    "MemoryPlan",
    "MemoryTimelineEvent",
    "RawMemoryEvent",
    "TensorLifetime",
    "TensorRef",
    "estimate_static_memory",
]
