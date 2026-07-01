# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Side-loaded simulator package: captures the four-layer IR (OpNode ->
StepGraph -> ScheduleGraph -> WorkloadGraph, per
https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec)
of one torchtitan_npu training step, without real NPU hardware or real
memory allocation. See
docs/superpowers/specs/2026-07-01-npu-simulator-design.md for the design.
"""
