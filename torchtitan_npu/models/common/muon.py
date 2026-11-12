# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Muon layouts; each model owns its mesh axes and parameter policy."""

import fnmatch
import re
from collections.abc import Iterable
from dataclasses import replace

import torch
from torch.distributed.tensor import Shard
from torchtitan.distributed.flex_shard import ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import GroupedLinear

from torchtitan_npu.config import MuonOptimizerProfile


def make_owned_layout(axes: Iterable[str]) -> ComputeLayout:
    return ComputeLayout(shardings_by_mesh_axis={axis: Owned() for axis in axes}, compute_dtype=torch.bfloat16)


def make_expert_layout(axes: Iterable[str]) -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={
            axis: Shard(0) for axis in (*axes, MeshAxisName.EDP_SHARD.value, MeshAxisName.EP.value)
        },
        shard_order_by_tensor_dim={0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value)},
        compute_dtype=torch.bfloat16,
    )


def with_lora_compute_layouts(model: Module.Config, profile: MuonOptimizerProfile) -> MuonOptimizerProfile:
    """Extend a model's Muon policy for its actual standard LoRA projections.

    Shared dense A factors are complete matrices. Dense/per-head B factors
    retain their base projection's compute layout, and both routed factors
    retain the expert axis. Projections outside the model's Muon policy,
    including the output head, keep the AdamW fallback.
    """
    kwargs = dict(profile.optimizer_factory_kwargs["DistMuon"])
    layouts = dict(kwargs["compute_sharding_by_fqn"])
    buckets = list(kwargs["bucket_configs"])
    factors = []
    for fqn, config, _, _ in model.traverse(Module.Config, recurse=True):
        if not (hasattr(config, "rank") and hasattr(config, "alpha")):
            continue
        base_key = f"{fqn}.weight"
        if base_key not in layouts:
            continue
        base_layout = layouts[base_key]
        a_layout = (
            base_layout
            if isinstance(config, GroupedLinear.Config)
            else make_owned_layout((MeshAxisName.DP_SHARD.value,))
        )
        names = (f"{fqn}.lora_a.weight", f"{fqn}.lora_b.weight")
        layouts.update(zip(names, (a_layout, base_layout), strict=True))
        matches = [
            index
            for index, bucket in enumerate(buckets)
            if any(fnmatch.fnmatchcase(base_key, pattern) for pattern in bucket.patterns)
        ]
        if len(matches) != 1:
            raise ValueError(f"LoRA base projection {base_key} must match exactly one Muon bucket")
        index = matches[0]
        buckets[index] = replace(buckets[index], patterns=(*buckets[index].patterns, *names))
        factors.extend(names)
    if not factors:
        return profile
    # Layer wrappers preserve projection suffixes in the optimizer's raw FQNs.
    suffixes = {fqn.split(".", 2)[-1] for fqn in factors}
    lora_pattern = "|".join(re.escape(suffix) for suffix in sorted(suffixes))
    kwargs.update(compute_sharding_by_fqn=layouts, bucket_configs=tuple(buckets))
    return replace(
        profile,
        muon_pattern=rf"(?:{profile.muon_pattern}|(?:{lora_pattern})$)",
        optimizer_factory_kwargs={**profile.optimizer_factory_kwargs, "DistMuon": kwargs},
    )
