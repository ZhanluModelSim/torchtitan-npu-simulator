# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tracing contract for the MX helper consumed by quantized experts.

Only the NPU quantization boundary is stubbed. This checks symbolic layout
handling, not the numerical implementation of the CANN quantizer.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import _check
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@pytest.fixture
def mx_module(monkeypatch):
    # Import the real config and helper without loading CANN shared libraries.
    # Dtype tokens are opaque to this CPU boundary; no NPU numerics are emulated.
    pytest.importorskip("torchao")
    from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER

    original_handlers = _QUANTIZE_CONFIG_HANDLER.copy()
    original_modules = {k: v for k, v in sys.modules.items() if k == "torchao_npu" or k.startswith("torchao_npu.")}
    npu = ModuleType("torch_npu")
    for token in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn_x2", "hifloat8"):
        setattr(npu, token, object())
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "experiments" / "torchao-npu"))
    for name in original_modules:
        del sys.modules[name]
    try:
        yield importlib.import_module("torchao_npu.quantization.quant_primitives.mx")
    finally:
        for name in list(sys.modules):
            if name == "torchao_npu" or name.startswith("torchao_npu."):
                del sys.modules[name]
        sys.modules.update(original_modules)
        _QUANTIZE_CONFIG_HANDLER.clear()
        _QUANTIZE_CONFIG_HANDLER.update(original_handlers)


@pytest.mark.parametrize("half", [0, 1], ids=["gate", "up"])
def test_mx_quantize_preserves_unbacked_rows_in_cat_gradient(monkeypatch, mx_module, half):
    mx = mx_module
    config = mx.MXQuantizeConfig()
    calls = []

    def fake_npu_quant(tensor, *, axis, **kwargs):
        calls.append((tensor, axis))
        # The supported test input is [R, 64], with two 32-element MX blocks.
        return (
            torch.empty(tensor.shape, dtype=config.elem_dtype, device=tensor.device),
            torch.empty((tensor.shape[0], 1, 2), dtype=torch.uint8, device=tensor.device),
        )

    monkeypatch.setattr(mx.torch_npu, "npu_dynamic_mx_quant", fake_npu_quant, raising=False)
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        rows = shape_env.create_unbacked_symint()
        # Record a symbolic constraint without specializing the unbacked size.
        _check(rows >= 0)
        # CatBackward passes a half-width view of the packed activation gradient.
        # Neither R=0 nor R=1 is excluded; R has no runtime hint.
        packed_gradient = torch.empty((rows, 128), dtype=torch.bfloat16)
        start = half * 64
        end = start + 64
        gradient = packed_gradient[:, start:end]
        assert gradient.stride() == (128, 1)

        quantized, scale = mx.mx_quantize(gradient, -1, config)

        assert quantized.shape[0].node.expr == rows.node.expr
        assert quantized.shape[1:] == (64,)
        assert quantized.dtype == config.elem_dtype
        assert scale.shape[0].node.expr == rows.node.expr
        assert scale.shape[1:] == (1, 2)
        assert scale.dtype == torch.uint8
        assert len(calls) == 1
        operand, axis = calls[0]
        assert operand.stride() == (128, 1)
        assert operand.storage_offset() == half * 64
        assert axis == -1
        assert not rows.node.has_hint()
