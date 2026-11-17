# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4651
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep EP AllToAll size queries local to each graph chunk.

Some dispatcher size queries carry the MoE module FQN but sit outside the
region body. Route only same-root ``sym_size.int(all_to_all, 0)`` users to the
chunked side after upstream classification. Leave full-value materialization
and its validation entirely to the upstream implementation.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from functools import wraps

import torch
import torch.fx as fx
import torchtitan.experiments.graph_trainer.ep_chunk_pass as ep_chunk_pass
from torchtitan.tools.logging import logger


def _is_dim0_all_to_all_size_user(user: fx.Node) -> bool:
    """Return whether ``user`` is ``aten.sym_size.int(all_to_all, 0)``."""
    if user.op != "call_function" or user.target is not torch.ops.aten.sym_size.int or len(user.args) != 2:
        return False
    producer, dimension = user.args
    return (
        isinstance(dimension, int)
        and dimension == 0
        and isinstance(producer, fx.Node)
        and producer.op == "call_function"
        and producer.target is torch.ops._c10d_functional.all_to_all_single.default
    )


def apply() -> None:
    current = ep_chunk_pass._split_live_out_users
    if getattr(current, "npu_keeps_ep_shapes_chunk_local", False):
        return

    @wraps(current)
    def split_live_out_users(users, plans, producer_region, symbol_hints):
        chunked, full = current(users, plans, producer_region, symbol_hints)
        moved = tuple(
            user
            for user in full
            if _is_dim0_all_to_all_size_user(user)
            and ep_chunk_pass.is_module_fqn_inside_root(ep_chunk_pass._get_module_fqn(user), producer_region.root_fqn)
        )
        if not moved:
            return chunked, full
        moved_set = set(moved)
        return (*chunked, *moved), tuple(user for user in full if user not in moved_set)

    # pyrefly: ignore [missing-attribute]
    split_live_out_users.npu_keeps_ep_shapes_chunk_local = True
    ep_chunk_pass._split_live_out_users = split_live_out_users
    logger.info("Enabled GraphTrainer EP AllToAll chunk-local shape live-out patch")


apply()
