# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration glue between ``torchao_npu`` and the torchtitan / torchtitan-npu stack.

Hosts functions and classes that exist specifically to plug ``torchao_npu``
into torchtitan's converter machinery (and the ``torchtitan-npu`` plugin layer
that wraps it).
"""

from dataclasses import dataclass
from typing import Annotated, ClassVar

import torch.nn as nn
import tyro
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import _is_linear, quantize_
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.quantization.module_utils import (
    capture_module_attrs,
    inject_module_protocol,
    verify_module_protocol,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

from ..quantization.filters import ModuleFilterFn, match_fqn_suffix


class NpuQuantizeConverter(QuantizationConverter):
    """Applies :func:`~torchao.quantization.quant_api.quantize_` with a config and a filter.

    See :class:`Config` for the accepted arguments.

    Must run **after** all PTQ-style converters (npu_rms_norm, npu_gmm, etc.)
    since the parameter-wrapping transform assumes the model structure is
    already final.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        _quantization_type: ClassVar[str] = "npu_quantize_converter"

        # Fully-formed quantize config (e.g., ``ParamSwapConfig``) forwarded to
        # ``quantize_``. The user is responsible for constructing the config with
        # whatever fields it requires (weight/activation configs,
        # ``params_filter_fn``, ``step``, etc.).
        #
        # Suppressed from tyro CLI parsing because ``AOBaseConfig`` is an
        # abstract base class — tyro can't pick a concrete subclass from CLI
        # args. The type is made Optional so ``None`` is a valid default at the
        # dataclass level; a non-None value must be supplied programmatically,
        # enforced by ``__post_init__``.
        base_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None

        # Module-level filter passed to ``quantize_``. Receives ``(module, fqn)``
        # and returns ``True`` if the quantize handler should be applied to that
        # module. Defaults to :func:`torchao.quantization.quant_api._is_linear`
        # (matches the default used by ``quantize_`` itself).
        #
        # Suppressed from tyro CLI parsing because ``ModuleFilterFn`` is a
        # Protocol; tyro can't auto-construct a callable from CLI args.
        filter_fn: Annotated[ModuleFilterFn, tyro.conf.Suppress] = _is_linear

        def __post_init__(self):
            if self.base_config is None:
                raise ValueError(
                    "NpuQuantizeConverter.Config.base_config must be provided "
                    "programmatically; it cannot be set via CLI."
                )

        def to_dict(self) -> dict:
            # Serialize base_config via repr to avoid JSON-serialization issues
            # with raw torch.dtype / Enum fields in the AO-side config dataclasses.
            d = {"_quantization_type": self._quantization_type}
            d["base_config"] = repr(self.base_config)
            d["filter_fn"] = repr(self.filter_fn)
            return d

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self.base_config = config.base_config
        self.filter_fn = config.filter_fn
        logger.info(f"Parameter quantize active with base_config={type(self.base_config).__name__}")

    def convert(self, model: nn.Module):
        # Capture Module attrs before conversion, matching the pattern used by
        # the upstream MXFP8Converter and Float8LinearConverter (even though
        # quantize_ with ParamSwapConfig only wraps parameters, not modules).
        verify_module_protocol(model, nn.Linear, Linear)
        saved_attrs = capture_module_attrs(model, ["_init_mean", "_init_std"], nn_module_cls=nn.Linear)

        # `base_config` is Optional at the type level (see Config), but
        # `__post_init__` guarantees it is non-None by the time we get here.
        assert self.base_config is not None, "base_config cannot be None"
        quantize_(model, self.base_config, filter_fn=self.filter_fn)

        # Re-inject Linear protocol and re-attach attrs lost during conversion
        inject_module_protocol(model, Linear, saved_attrs)
        verify_module_protocol(model, nn.Linear, Linear)

        logger.info("Applied parameter quantize wrapping (prepare step)")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        pass


# ---------------------------------------------------------------------------
# Module-level filters specific to torchtitan-npu's deepseek V4 family.
# These match module FQNs that exist in those model definitions; not generic
# — only used when the host framework is torchtitan-npu.
#
# For the combined "attention or shared expert" match, use
# ``any_filter(is_attention, is_shared_expert)``.
# ---------------------------------------------------------------------------

is_attention = match_fqn_suffix(
    "post_attention.wo_a",
    "post_attention.wo_b",
    "pre_attention.wq_a",
    "pre_attention.wq_b",
    "pre_attention.wkv",
    "pre_attention.indexer.wq_b",
)


is_shared_expert = match_fqn_suffix(
    "shared_experts.w1",
    "shared_experts.w2",
    "shared_experts.w3",
)


is_routed_expert = match_fqn_suffix(".experts")
