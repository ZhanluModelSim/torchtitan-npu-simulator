# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Activate equivalent native planning optimizations after compatibility backports.

This is the sole extension owner of these planning targets. No native tensor
math, communication runtime or optimizer state logic is replaced.
"""

import torchtitan.distributed.flex_shard._optimizer_reshard_schedule as schedule
import torchtitan.distributed.flex_shard.dist_muon as muon

from . import planning


def apply() -> None:
    schedule._validate_disjoint_regions_in_bounds = planning.validate_disjoint_regions_in_bounds
    schedule._validate_redistribution_plan = planning.validate_redistribution_plan
    schedule._build_packed_all_to_all_schedule = planning.build_packed_all_to_all_schedule
    schedule._dtensor_storage_region_for_participant = planning.dtensor_storage_region_for_participant
    schedule._dtensor_storage_regions = planning.dtensor_storage_regions
    schedule._build_bucket_plans = planning.build_bucket_plans
    # The optimizer imports these by value; route both callers through one owner.
    muon._dtensor_storage_regions = planning.dtensor_storage_regions
    muon._build_bucket_plans = planning.build_bucket_plans
    muon._resolve_muon_redistribution_plans = planning.resolve_muon_redistribution_plans


apply()
