# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in token dispatcher for symbolic CV batch-chunk tracing."""

from dataclasses import dataclass

import torch_npu
from torch.overrides import TorchFunctionMode
from torchtitan.config import derive, override

from torchtitan_npu.ops.ascendc.moe_token_permute import npu_moe_token_permute
from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


class _SymbolicPermuteMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch_npu.npu_moe_token_permute:
            func = npu_moe_token_permute
        return func(*args, **(kwargs or {}))


class BatchChunkTokenDispatcher(AscAllToAllTokenDispatcher):
    @dataclass(kw_only=True, slots=True)
    class Config(AscAllToAllTokenDispatcher.Config):
        pass

    def dispatch(self, *args, **kwargs):
        # The mode is scoped to this call; shared functions and other threads
        # retain native dispatch. Reuse the full upstream dispatch algorithm.
        with _SymbolicPermuteMode():
            return super().dispatch(*args, **kwargs)


@override(target=AllToAllTokenDispatcher.Config, exact=True, description="Symbolic-size-safe fused token permutation")
def asc_dispatcher(cfg: AllToAllTokenDispatcher.Config) -> BatchChunkTokenDispatcher.Config:
    return derive(cfg, BatchChunkTokenDispatcher.Config)
