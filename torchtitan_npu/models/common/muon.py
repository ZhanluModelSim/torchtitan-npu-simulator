# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Muon layouts; each model owns its mesh axes and parameter policy."""

from collections.abc import Iterable

from torch.distributed.tensor import Shard
from torchtitan.distributed.flex_shard import ComputeLayout, Owned
from torchtitan.distributed.parallel_dims import MeshAxisName


def make_owned_layout(axes: Iterable[str]) -> ComputeLayout:
    return ComputeLayout(shardings_by_mesh_axis={axis: Owned() for axis in axes})


def make_expert_layout(axes: Iterable[str]) -> ComputeLayout:
    return ComputeLayout(
        shardings_by_mesh_axis={axis: Shard(0) for axis in (*axes, MeshAxisName.EFSDP.value, MeshAxisName.EP.value)}
    )
