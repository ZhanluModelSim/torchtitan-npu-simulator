# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import nn
from torchao.prototype.moe_training.utils import unwrap_weight
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat.fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
)
from torchao.utils import TorchAOBaseTensor

from ..ops import float8_rowwise_fake_quantize
from ..quantization.transform import register_parameter_swap_handler
from . import BaseTrainingWeightWrapperTensor


class Float8TrainingWeightWrapperTensor(BaseTrainingWeightWrapperTensor):
    """
    Applies FP8 row-wise fake-quantization.

    Intercepts computation ops via :meth:`__torch_function__`, applies per-row
    FP8 fake-quantization (quantize → dequantize in high precision with STE
    gradient) to the weights and optionally the activations, and delegates to
    the standard op.

    Both ``weight_config`` and ``activation_config`` (if set) must be
    :class:`~torchao.quantization.qat.fake_quantize_config.Float8FakeQuantizeConfig`.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        weight_config: FakeQuantizeConfigBase | None = None,
        activation_config: FakeQuantizeConfigBase | None = None,
    ):
        if weight_config is not None:
            if not isinstance(weight_config, Float8FakeQuantizeConfig):
                raise ValueError(
                    f"Only `Float8FakeQuantizeConfig` is supported for `weight_config` in {type(self).__name__}."
                )
            elif weight_config.granularity != PerRow(dim=-1):
                raise ValueError("Only the row-wise granularity is supported.")

        if activation_config is not None:
            if not isinstance(activation_config, Float8FakeQuantizeConfig):
                raise ValueError(
                    f"Only `Float8FakeQuantizeConfig` is supported for `activation_config` in {type(self).__name__}."
                )
            elif activation_config.granularity != PerRow(dim=-1):
                raise ValueError("Only the row-wise granularity is supported for `activation_config`.")

        super().__init__(tensor, weight_config=weight_config, activation_config=activation_config)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Because we defer fake-quantization in the case of slicing, indexing, transposing,
        and permuting the wrapper tensor, the following fake-quantization is fragile when
        complicated transpositions and permutations are applied before real computations.
        After a general sequence of transpositions and permutations, it is NOT guaranteed
        that the fake-quantization is still carried out along the desired dimension.

        1. During prepare, the default or user-defined params_filter_fn may also wrap bias
           parameters (1D or 2D) that appear in F.linear/linear/addmm. In either case, the
           fake-quantization of these wrapped biases is always bypassed here: we only unpack
           the weight position for fake-quantization. The bias passes through other positions
           and is unwrapped transparently by __torch_dispatch__ at add time.

        2. For torch.bmm, the wrapped weight may be at args[1] (is_transposed=True) or
           args[0] (is_transposed=False, as in HF's _batched_linear). We detect the
           position at runtime and correctly select the contracted dimension.
        """

        if kwargs is None:
            kwargs = {}

        if func in (torch._grouped_mm, torch.matmul, torch.mm):
            # weight at args[1], contracted dim=-2
            return cls._fake_quantize_then_compute(0, 1, PerRow(dim=-2), cls, func, types, args, kwargs)

        elif func is torch.nn.functional.linear:
            # weight at args[1], contracted dim=-1
            return cls._fake_quantize_then_compute(0, 1, PerRow(dim=-1), cls, func, types, args, kwargs)

        elif func is torch.bmm:
            # torch.bmm(input, mat2) — both args are 3D tensors. The wrapped weight
            # may be at args[1] (default) or args[0] (e.g. HF's _batched_linear).
            assert len(args) >= 2, f"torch.bmm expects 2 args, got {len(args)}"
            if isinstance(args[1], cls):
                # weight at args[1], shape [B, K, N], contracted dim=-2
                return cls._fake_quantize_then_compute(0, 1, PerRow(dim=-2), cls, func, types, args, kwargs)

            else:
                # weight at args[0], shape [B, N, K], contracted dim=-1
                return cls._fake_quantize_then_compute(1, 0, PerRow(dim=-1), cls, func, types, args, kwargs)

        elif func is torch.addmm:
            # weight at args[2], contracted dim=-2
            return cls._fake_quantize_then_compute(1, 2, PerRow(dim=-2), cls, func, types, args, kwargs)

        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    @staticmethod
    def _fake_quantize_then_compute(a_pos: int, b_pos: int, granularity: PerRow, cls, func, types, args, kwargs):
        A, B = args[a_pos], args[b_pos]

        assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
        assert isinstance(B, cls), f"B should be a {cls.__name__}"

        # Fake-quantize the activation if B.activation_config exists. With torch._grouped_mm, activation
        # is quantized once for the shared 3D weight. In a per-expert loop pattern, this repeats per expert.
        # Activation fake-quantization is skipped if the activation is empty. This is a possible case when
        # a loop over experts instead of grouped_mm is used and some experts don't receive any tokens.
        if B.activation_config is not None and A.numel() > 0:
            assert not isinstance(A, TorchAOBaseTensor), (
                f"When an activation config is specified, the activation must not be a quantized tensor, got {type(A)}"
            )
            fq_A = float8_rowwise_fake_quantize(
                A, B.activation_config, PerRow(dim=-1)
            )  # always quantize the last dimension of activations
        else:
            fq_A = A

        # Fake-quantize the weight
        B_data = unwrap_weight(B)
        if B.weight_config is not None:
            fq_B_data = float8_rowwise_fake_quantize(B_data, B.weight_config, granularity)
        else:
            fq_B_data = B_data

        args = list(args)
        args[a_pos] = fq_A
        args[b_pos] = fq_B_data

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


@register_parameter_swap_handler(Float8FakeQuantizeConfig)
def _(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: tuple[Any, ...] = (),
):
    from ..configs import ParamSwapConfig

    config: ParamSwapConfig = extra_args[0]

    if not isinstance(config, ParamSwapConfig):
        raise ValueError(f"extra_args[0] must be a ParamSwapConfig, got {type(config).__name__}.")

    return nn.Parameter(
        data=Float8TrainingWeightWrapperTensor(
            param.data,
            activation_config=config.activation_config,
            weight_config=config.weight_config,
        ),
        requires_grad=param.requires_grad,
    )
