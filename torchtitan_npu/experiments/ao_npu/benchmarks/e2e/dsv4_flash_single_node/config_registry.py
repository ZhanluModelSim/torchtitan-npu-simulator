# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""E2E benchmark config for DSV4 Flash single-node QAT.

The launch scripts in this folder (``debug_deepseek_v4_single_node_qat.sh``,
``debug_deepseek_v4_single_node_qat_hif8.sh``) cd into this directory so the
flat module name ``config_registry`` is importable as a top-level module by
torchrun workers. Discovered by upstream ``ConfigManager`` via
``--module config_registry`` and one of:
- ``--config debug_deepseek_v4_flash_single_node_qat`` (MX for
  attention/shared-expert, Block FP8 for routed experts)
- ``--config debug_deepseek_v4_flash_single_node_mxfp8_qat`` (MX
  everywhere)
- ``--config debug_deepseek_v4_flash_single_node_hif8_qat`` (HiF8
  everywhere)

Note: ``torchao_npu`` is not a standalone package — it's a subpackage of
``torchtitan_npu.experiments.ao_npu``, so all imports must use the full path
``torchtitan_npu.experiments.ao_npu.torchao_npu.*``.
"""

from dataclasses import replace

import torch
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase
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
    HiF8QuantizeConfig,
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


def _debug_deepseek_v4_flash_single_node_uniform_qat(
    quant_config_cls: type[FakeQuantizeConfigBase],
) -> TrainerConfig:
    """Shared body for the all-uniform-backend QAT configs (MXFP8, HiF8).

    Both apply a single ``quant_config_cls`` to every target FQN (attention,
    shared experts, ``e_proj``/``h_proj``, and routed experts alike) via one
    ``NpuQuantizeConverter.Config`` -- differing only in the quantize config
    class.
    """
    base = debug_deepseek_v4_flash_single_node()
    return replace(
        base,
        model_converters=ModelConvertersContainer.Config(
            converters=base.model_converters.converters  # noqa: RUF005
            + [
                NpuQuantizeConverter.Config(
                    base_config=ParamSwapConfig(
                        weight_config=quant_config_cls(),
                        activation_config=quant_config_cls(),
                    ),
                    filter_fn=any_filter(
                        is_attention,
                        is_shared_expert,
                        is_routed_expert,
                        match_fqn_suffix(".e_proj", ".h_proj"),
                    ),
                ),
            ]
        ),
    )


def debug_deepseek_v4_flash_single_node_mxfp8_qat() -> TrainerConfig:
    """All-MXFP8 counterpart of :func:`debug_deepseek_v4_flash_single_node_qat`.

    Unlike ``debug_deepseek_v4_flash_single_node_qat``, which splits MX
    (attention/shared-expert) from Block FP8 (routed experts), this uses a
    single ``MXQuantizeConfig`` for every target FQN -- structurally
    identical to :func:`debug_deepseek_v4_flash_single_node_hif8_qat`, just
    with ``MXQuantizeConfig`` in place of ``HiF8QuantizeConfig``, so the two
    are directly comparable (same ParamSwap plumbing, only the backend
    differs).
    """
    return _debug_deepseek_v4_flash_single_node_uniform_qat(MXQuantizeConfig)


def debug_deepseek_v4_flash_single_node_hif8_qat() -> TrainerConfig:
    """HiF8 counterpart of :func:`debug_deepseek_v4_flash_single_node_qat`.

    Unlike the MX/Block FP8 split above, HiF8 is a single per-tensor backend
    with no block-size/axis distinction, so one ``NpuQuantizeConverter.Config``
    covers attention, shared experts, ``e_proj``/``h_proj``, and routed
    experts alike -- mirroring the unified FQN target list the core
    ``HiF8Converter`` (``torchtitan_npu/converters/hif8.py`` on the
    ``dsv4-hif8`` branch) used for the same model.
    """
    return _debug_deepseek_v4_flash_single_node_uniform_qat(HiF8QuantizeConfig)
