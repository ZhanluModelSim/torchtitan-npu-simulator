# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from torchtitan_npu.models.deepseek_v4 import parallelize


@dataclass(frozen=True)
class _MixedPrecisionPolicy:
    param_dtype: torch.dtype
    reduce_dtype: torch.dtype
    cast_forward_inputs: bool = True


class _FSDPState:
    def __init__(self):
        self._mp_policy = _MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )


def test_disable_transformer_block_cast_forward_inputs():
    layers = (nn.Identity(), nn.Identity())
    states = {layer: _FSDPState() for layer in layers}
    model = SimpleNamespace(layers={"0": layers[0], "1": layers[1]})

    disable_transformer_block_cast_forward_inputs = getattr(
        parallelize,
        "_disable_transformer_block_cast_forward_inputs",
    )
    disable_transformer_block_cast_forward_inputs(
        model,
        states.get,
    )

    for layer in layers:
        mp_policy = getattr(states[layer], parallelize.FSDP_MP_POLICY_ATTR)
        assert mp_policy.cast_forward_inputs is False
        assert mp_policy.param_dtype is torch.bfloat16
        assert mp_policy.reduce_dtype is torch.float32
