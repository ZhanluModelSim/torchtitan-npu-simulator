# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 quantization converter for mm_gc.

Covers every matmul in the model plus the MoE grouped matmul, controlled via
module-path (FQN) substrings — the same ``fqns`` mechanism as the upstream
``MXFP8Converter``:

- ``attention.to_``    attention QKV/output projections (to_q/to_k/to_v/to_out)
- ``feed_forward.``    dense SwiGLU MLP (gate/up/down projections)
- ``moe.proj_``        Multi-Head MoE split/merge projections (proj_in/proj_out)
- ``moe.shared_``      shared expert projections
- ``moe.experts``      routed-expert bank (3D ``[E, out, in]`` weights ->
                        MXFP8 grouped matmul via the NPU patch in
                        ``torchtitan_npu/patches/torchao_npu/mxfp8_grouped_mm.py``)

Excluded by design (routing runs in fp32 for numerical stability, matching
the reference implementation):

- ``attention.core.proj_q`` / ``attention.core.proj_k``  (SLA2 block router)
- ``moe.gate.router``                                    (MoE head router)

The converter fails fast when a configured FQN would match either router,
when an FQN matches no module, or when the underlying hardware/simulator
does not provide MXFP8 kernels. Linear modules touched by the conversion
keep the torchtitan ``Linear`` module protocol; the two routers are left
completely untouched.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.distributed import ParallelDims

DEFAULT_MM_GC_MXFP8_FQNS: tuple[str, ...] = (
    "attention.to_",
    "feed_forward.",
    "moe.proj_",
    "moe.shared_",
    "moe.experts",
)

_FORBIDDEN_ROUTER_FQN_MARKERS: tuple[str, ...] = (
    "attention.core.proj_",
    "moe.gate.router",
)


class MMGcMXFP8Converter(MXFP8Converter):
    """mm_gc-scoped MXFP8 dynamic-quantization converter.

    Subclasses the upstream ``MXFP8Converter`` (so ``find_pad_multiple`` and
    the simulator's meta patches keep working) with mm_gc defaults and
    router-safety checks.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MXFP8Converter.Config):
        fqns: list[str] = field(
            default_factory=lambda: list(DEFAULT_MM_GC_MXFP8_FQNS)
        )

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        self._validate_fqns(config.fqns)
        super().__init__(
            config,
            parallel_dims=parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )
        self.pad_token_groups_for_grouped_mm = not parallel_dims.ep_enabled

    @staticmethod
    def _validate_fqns(fqns: list[str]) -> None:
        if not fqns:
            raise ValueError(
                "mm_gc MXFP8 requires at least one module-path (FQN) filter; "
                f"defaults: {list(DEFAULT_MM_GC_MXFP8_FQNS)}"
            )
        for target in fqns:
            for marker in _FORBIDDEN_ROUTER_FQN_MARKERS:
                if marker in target:
                    raise ValueError(
                        f"mm_gc MXFP8 fqn filter {target!r} would match a "
                        f"router module ({marker}); routing must stay in "
                        "fp32, exclude it from the quantization scope"
                    )

    def convert(self, model: nn.Module):
        if not self.enabled:
            return

        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        fqn_hits = dict.fromkeys(self.config.fqns, 0)

        def module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            for target_fqn in self.config.fqns:
                if target_fqn in cur_fqn:
                    fqn_hits[target_fqn] += 1
                    return True
            return False

        recipe = MXFP8TrainingRecipe(self.config.recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        mxfp8_op_config.pad_token_groups_for_grouped_mm = (
            self.pad_token_groups_for_grouped_mm
        )

        quantize_(model, config=mxfp8_op_config, filter_fn=module_filter_fn)

        missed = [fqn for fqn, hits in fqn_hits.items() if hits == 0]
        if missed:
            raise RuntimeError(
                f"mm_gc MXFP8 fqn filters matched no module: {missed}; "
                "check the module paths against the model structure"
            )

        self._verify_routers_untouched(model)

    @staticmethod
    def _verify_routers_untouched(model: nn.Module) -> None:
        from torchtitan_npu.models.mm_gc.attention import SLA2Attention
        from torchtitan_npu.models.mm_gc.feed_forward import MultiHeadMoE

        def _is_plain_tensor(t: torch.Tensor) -> bool:
            return type(t) is torch.Tensor

        for module in model.modules():
            if isinstance(module, SLA2Attention):
                for name in ("proj_q", "proj_k"):
                    weight = getattr(module.core, name).weight.detach()
                    if not _is_plain_tensor(weight):
                        raise RuntimeError(
                            f"SLA2 router weight {name} was modified by "
                            "MXFP8 quantization; routing must stay fp32"
                        )
            if isinstance(module, MultiHeadMoE):
                router = module.gate.router.detach()
                if not _is_plain_tensor(router):
                    raise RuntimeError(
                        "MoE head router was modified by MXFP8 quantization; "
                        "routing must stay fp32"
                    )
