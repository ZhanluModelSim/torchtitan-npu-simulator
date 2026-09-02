# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from collections.abc import Callable
from typing import Any

from torch import nn
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase

from .filters import ParameterFilterFn, _is_parameter

"""
Registry mapping a :class:`FakeQuantizeConfigBase` type to a handler that
wraps or unwraps a single ``nn.Parameter`` for ParamSwap. The handler receives
the parameter and the :class:`ParamSwapConfig` and returns the transformed parameter.

This is the parameter-level analogue of ``_QUANTIZE_CONFIG_HANDLER`` in
:mod:`torchao.quantization.transform_module`.
"""
_PARAM_SWAP_QUANTIZE_CONFIG_HANDLER: dict[
    type[FakeQuantizeConfigBase],
    Callable[[nn.Module, str, nn.Parameter, tuple[Any, ...]], nn.Parameter],
] = {}


def register_parameter_swap_handler(
    config_type: type[FakeQuantizeConfigBase],
):
    """
    Decorator to register a handler for a specific :class:`FakeQuantizeConfigBase`
    type. The handler receives ``(module, param_fqn, param, extra_args)`` where
    ``extra_args[0]`` is the :class:`ParamSwapConfig`, and returns the transformed
    ``nn.Parameter``.
    """

    @functools.wraps(config_type)
    def decorator(func):
        _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func

    return decorator


def unwrap_param(
    module: nn.Module,
    param_fqn: str,
    param: nn.Parameter,
    extra_args: tuple[Any, ...] = (),
):
    from ..wrapper_tensors.base_wrapper_tensor import BaseTrainingWeightWrapperTensor

    # ``unwrap_param`` is invoked with parameters already wrapped by
    # ``BaseTrainingWeightWrapperTensor`` (the convert step pairs with
    # ``_is_parameter_with_wrapped_data``). Narrow the type so static
    # checkers see the wrapper-specific ``to_tensor`` method.
    assert isinstance(param.data, BaseTrainingWeightWrapperTensor), (
        f"unwrap_param expects a parameter wrapped by BaseTrainingWeightWrapperTensor, got {type(param.data).__name__}"
    )
    return nn.Parameter(
        param.data.to_tensor(),
        requires_grad=param.requires_grad,
    )


def _replace_params_with_custom_fn_if_matches_filter(
    module: nn.Module,
    params_replacement_fn,
    params_filter_fn: ParameterFilterFn,
    cur_fqn: str = "",
    extra_args: tuple[Any, ...] = (),
):
    """Recursively replace matching parameters in a module and its submodules with custom replacements"""

    params_filter_fn = _is_parameter if params_filter_fn is None else params_filter_fn

    for child_name, child in module.named_children():
        child_fqn = f"{cur_fqn}.{child_name}" if cur_fqn else child_name
        new_child = _replace_params_with_custom_fn_if_matches_filter(
            child,
            params_replacement_fn,
            params_filter_fn,
            cur_fqn=child_fqn,
            extra_args=extra_args,
        )
        if new_child is not child and new_child is not None:
            # _replace_params_with_custom_fn_if_matches_filter mutates child in-place and returns it.
            # This branch is a safety net in case a future implementation returns a new module instead.
            setattr(module, child_name, new_child)

    for param_name, param in module.named_parameters(recurse=False):
        param_fqn = f"{cur_fqn}.{param_name}" if cur_fqn else param_name
        if not params_filter_fn(param, param_fqn):
            continue
        new_param = params_replacement_fn(module, param_fqn, param, extra_args)
        if new_param is not param and new_param is not None:
            setattr(module, param_name, new_param)

    return module
