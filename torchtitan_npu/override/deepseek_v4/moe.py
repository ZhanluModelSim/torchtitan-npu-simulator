# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TileLang SwiGLU for the first-class FeedForward and RoutedExperts configs."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torchtitan.config import derive, override
from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import RoutedExperts

from torchtitan_npu.ops.tilelang.swiglu import tilelang_swiglu


class TilelangSwiGLU(BinaryActivationFn):
    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        swiglu_limit: float = 0.0

    def __init__(self, config):
        self.swiglu_limit = config.swiglu_limit

    def _activation_fn(self, gate, up):
        raise NotImplementedError("TileLang implements the fused activation")

    def __call__(self, gate_up: torch.Tensor, **kwargs):
        gate, up = gate_up.unbind(dim=-2)
        if isinstance(gate, DTensor) or isinstance(up, DTensor):
            raise ValueError("TileLang SwiGLU requires local SPMD inputs")
        shape = gate.shape
        return tilelang_swiglu(
            gate.reshape(-1, shape[-1]).contiguous(),
            up.reshape(-1, shape[-1]).contiguous(),
            self.swiglu_limit if self.swiglu_limit > 0 else None,
        ).reshape(shape)


def _replace_activation(cfg):
    return derive(
        cfg,
        type(cfg),
        activation_fn=TilelangSwiGLU.Config(
            swiglu_limit=getattr(cfg.activation_fn, "swiglu_limit", 0.0),
        ),
    )


@override(target=RoutedExperts.Config, description="DeepSeek routed experts with TileLang SwiGLU")
def tilelang_swiglu_grouped(cfg: RoutedExperts.Config) -> RoutedExperts.Config:
    return _replace_activation(cfg)


@override(target=FeedForward.Config, description="DeepSeek shared experts with TileLang SwiGLU")
def tilelang_swiglu_shared(cfg: FeedForward.Config) -> FeedForward.Config:
    return _replace_activation(cfg)
