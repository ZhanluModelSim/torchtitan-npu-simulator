# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped Meta admission for native DistMuon, without replacing its execution."""

from functools import wraps

import torch
from torch.distributed.tensor import DTensor

from torchtitan_simulator.capture.optimizer import capture_parameter_update


def install_meta_muon(scope) -> dict[str, str]:
    from torchtitan.distributed.flex_shard.dist_muon import DistMuon

    original_storage = DistMuon._validate_parameter_storage
    original_plan = DistMuon._validate_plan_across_ranks
    original_update = DistMuon._apply_update

    def meta_parameters(optimizer):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        return bool(parameters) and all(
            isinstance(parameter, DTensor) and parameter.to_local().device.type == "meta" for parameter in parameters
        )

    @wraps(original_storage)
    def validate_storage(optimizer):
        if meta_parameters(optimizer):
            return torch.device("meta")
        return original_storage(optimizer)

    @wraps(original_plan)
    def validate_plan(optimizer):
        if meta_parameters(optimizer):
            # Native layout/bucket construction still validates the local plan.
            # The hash collective compares values across real peers; a single
            # logical-rank Meta capture cannot perform or certify that comparison.
            return
        return original_plan(optimizer)

    @wraps(original_update)
    def apply_update(optimizer, compute_layout, direction):
        with capture_parameter_update(compute_layout.param, compute_layout.fqn):
            return original_update(optimizer, compute_layout, direction)

    scope.replace(DistMuon, "_validate_parameter_storage", validate_storage)
    scope.replace(DistMuon, "_validate_plan_across_ranks", validate_plan)
    scope.replace(DistMuon, "_apply_update", apply_update)
    return {
        "muon_execution": "native DistMuon compute, redistribution, streams and events on Meta",
        "muon_peer_plan_validation": "not_performed_single_logical_rank; native local plan checks retained",
    }
