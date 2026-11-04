# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operator cost annotations for the captured workload (FLOPs / traffic)."""

from torchtitan_simulator.cost.op_cost_model import (
    annotate_workload_costs,
    compute_cost,
)

__all__ = ["annotate_workload_costs", "compute_cost"]
