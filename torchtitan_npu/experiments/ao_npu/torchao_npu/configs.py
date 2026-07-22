# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.qat import QATConfig, QATStep
from torchao.quantization.qat.fake_quantize_config import (
    FakeQuantizeConfigBase,
    Float8FakeQuantizeConfig,
    _infer_fake_quantize_configs,
)
from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.transform_module import register_quantize_module_handler

from .quantization.filters import ParameterFilterFn, _is_parameter, _is_parameter_with_wrapped_data
from .quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from .quantization.transform import (
    _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER,
    _replace_params_with_custom_fn_if_matches_filter,
    unwrap_param,
)


class ParamSwapConfig(QATConfig):
    """
    Config for low-precision training with parameter wrapping, to be used with
    :func:`~torchao.quantization.quant_api.quantize_`.

    Extends :class:`~torchao.quantization.qat.QATConfig` to reuse its
    prepare/convert machinery. Unlike module-level QAT (which replaces
    ``torch.nn.Linear`` modules with ``FakeQuantizedLinear``), ParamSwap wraps
    matched ``nn.Parameter`` data with a training-time tensor wrapper subclass
    (``BaseTrainingWeightWrapperTensor``). This intercepts computation ops
    (``torch.mm``, ``torch._grouped_mm``, etc.) via ``__torch_function__`` and
    applies the subclass-specific precision transform — fake-quantize with
    straight-through estimator, or real low-precision matmul — to the weights.

    The workflow follows the same two-step pattern:

    1. Prepare: wraps matched parameters with ``BaseTrainingWeightWrapperTensor``
    2. Convert: unwraps ``BaseTrainingWeightWrapperTensor`` back to regular tensors

    Supported configs: FP8 row-wise
    (:class:`~torchao.quantization.qat.fake_quantize_config.Float8FakeQuantizeConfig`),
    NPU MX block-wise
    (:class:`~torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs.MXQuantizeConfig`),
    and NPU Block FP8
    (:class:`~torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs.BlockQuantizeConfig`).
    """

    def __init__(
        self,
        base_config: AOBaseConfig | None = None,
        activation_config: FakeQuantizeConfigBase | None = None,
        weight_config: FakeQuantizeConfigBase | None = None,
        *,
        step: QATStep = QATStep.PREPARE,
        params_filter_fn: ParameterFilterFn = _is_parameter,
    ):
        self.params_filter_fn = params_filter_fn
        super().__init__(base_config, activation_config, weight_config, step=step)

    def __post_init__(self):
        torch._C._log_api_usage_once("torchao.prototype.param_swap.ParamSwapConfig")
        if self.activation_config is not None and not isinstance(
            self.activation_config, (Float8FakeQuantizeConfig, MXQuantizeConfig)
        ):
            raise ValueError(
                "Only `Float8FakeQuantizeConfig` or `MXQuantizeConfig` "
                "is supported for `activation_config` in ParamSwapConfig yet."
            )
        if self.weight_config is not None and not isinstance(
            self.weight_config, (Float8FakeQuantizeConfig, MXQuantizeConfig, BlockQuantizeConfig)
        ):
            raise ValueError(
                "Only `Float8FakeQuantizeConfig`, `MXQuantizeConfig`, or `BlockQuantizeConfig` "
                "is supported for `weight_config` in ParamSwapConfig yet."
            )

        super().__post_init__()

        # Upstream QATConfig.__post_init__ coerces `self.step = self.step.lower()`, which
        # stores the lowercase string ("prepare") instead of the QATStep enum member.
        # Coerce back to the enum so that downstream type checkers (e.g., tyro CLI parsing
        # of NpuQuantizeConverter.Config defaults) see QATStep.PREPARE rather than "prepare".
        self.step = QATStep(self.step)

        if self.step == QATStep.PREPARE:
            if self.base_config is not None:
                if not isinstance(self.base_config, Float8DynamicActivationFloat8WeightConfig):
                    raise ValueError(
                        "Only `Float8DynamicActivationFloat8WeightConfig` is supported for "
                        "`base_config` in ParamSwapConfig yet."
                    )
                self.activation_config, self.weight_config = _infer_fake_quantize_configs(self.base_config)
                self.base_config = None

            if self.weight_config is None:
                raise ValueError("`weight_config` is required for the prepare step of ParamSwapConfig.")

        elif self.step == QATStep.CONVERT:
            if self.base_config is not None:
                raise NotImplementedError("Applying PTQ in the convert step is not implemented yet.")


@register_quantize_module_handler(ParamSwapConfig)
def _param_swap_config_transform(
    module: nn.Module,
    config: ParamSwapConfig,
) -> nn.Module:
    """Prepare or convert parameter-level wrapping for ParamSwapConfig"""

    if config.step == QATStep.PREPARE:
        assert config.weight_config is not None
        params_handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[type(config.weight_config)]
        _replace_params_with_custom_fn_if_matches_filter(
            module, params_handler, config.params_filter_fn, extra_args=(config,)
        )

    elif config.step == QATStep.CONVERT:
        _replace_params_with_custom_fn_if_matches_filter(
            module, unwrap_param, _is_parameter_with_wrapped_data, extra_args=(config,)
        )

    else:
        raise ValueError(f"Invalid value of config.step: {config.step}")

    return module
