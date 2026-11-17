# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4664
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register TorchFT replica averaging on every FSDP parameter group.

Remove this module after the TorchTitan dependency includes the PR.
"""

import torch
import torch.distributed as dist
import torchtitan.experiments.torchft.manager
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.distributed_c10d import ReduceOp


def maybe_set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
    if self.enabled and self.use_async_quorum:

        def all_reduce_hook(output):
            dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

        def apply_set_all_reduce_hook(m):
            if isinstance(m, FSDPModule):
                param_groups = m._get_fsdp_state()._fsdp_param_groups
                for param_group in param_groups:
                    param_group._all_reduce_hook = all_reduce_hook

        for model_part in model_parts:
            model_part.apply(apply_set_all_reduce_hook)


def apply() -> None:
    torchtitan.experiments.torchft.manager.TorchFTManager.maybe_set_all_reduce_hook = maybe_set_all_reduce_hook


apply()
