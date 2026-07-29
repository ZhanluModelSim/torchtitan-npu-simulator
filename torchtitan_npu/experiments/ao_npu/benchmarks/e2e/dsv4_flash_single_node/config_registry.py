# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2E benchmark config for DSV4 Flash single-node QAT.

The launch script (``debug_deepseek_v4_single_node_qat.sh`` in this folder)
cds into this directory so the flat module name ``config_registry`` is
importable as a top-level module by torchrun workers. Discovered by upstream
``ConfigManager`` via ``--module config_registry`` and
``--config debug_deepseek_v4_flash_single_node_qat``.

Note: ``torchao_npu`` is not a standalone package — it's a subpackage of
``torchtitan_npu.experiments.ao_npu``, so all imports must use the full path
``torchtitan_npu.experiments.ao_npu.torchao_npu.*``.
"""

from dataclasses import replace

import torch
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import TrainerConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.configs import ParamSwapConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.interfaces.torchtitan import (
    NpuQuantizeConverter,
    is_attention,
    is_routed_expert,
    is_shared_expert,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.filters import (
    any_filter,
    match_fqn_suffix,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import (
    BlockQuantizeConfig,
    MXQuantizeConfig,
)
from torchtitan_npu.models.deepseek_v4.config_registry import (
    debug_deepseek_v4_flash_single_node,
)


def debug_deepseek_v4_flash_single_node_qat() -> TrainerConfig:
    base = debug_deepseek_v4_flash_single_node()
    return replace(
        base,
        model_converters=ModelConvertersContainer.Config(
            converters=base.model_converters.converters  # noqa: RUF005
            + [
                NpuQuantizeConverter.Config(
                    base_config=ParamSwapConfig(
                        weight_config=MXQuantizeConfig(),
                        activation_config=MXQuantizeConfig(),
                    ),
                    filter_fn=any_filter(is_attention, is_shared_expert, match_fqn_suffix(".e_proj", ".h_proj")),
                ),
                NpuQuantizeConverter.Config(
                    base_config=ParamSwapConfig(
                        weight_config=BlockQuantizeConfig(
                            mxfp4_fake_quantize_config=MXQuantizeConfig(
                                elem_dtype=torch.float4_e2m1fn_x2,
                                dst_type_max=7.0,
                            ),
                        ),
                        activation_config=MXQuantizeConfig(),
                    ),
                    filter_fn=is_routed_expert,
                ),
            ]
        ),
    )
