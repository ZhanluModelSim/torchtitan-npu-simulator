# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Defer DistMuon's collective plan validation during peer healing only."""

from contextlib import ExitStack, contextmanager

from torchtitan.distributed.flex_shard.dist_muon import DistMuon


class OptimizerRecovery:
    def __init__(self, optimizers):
        self._optimizers = [optimizer for optimizer in optimizers if isinstance(optimizer, DistMuon)]

    @contextmanager
    def defer_validation(self):
        # The post-load hook still rebuilds the local plan. Only its collective waits.
        with ExitStack() as restore:
            for optimizer in self._optimizers:
                method = "_validate_plan_across_ranks"
                original = getattr(optimizer, method)
                setattr(optimizer, method, lambda: None)
                restore.callback(setattr, optimizer, method, original)
            yield

    def validate(self):
        for optimizer in self._optimizers:
            optimizer._validate_plan_across_ranks()
