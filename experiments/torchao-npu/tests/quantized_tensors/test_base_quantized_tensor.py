# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the behavior :class:`BaseQuantizedTensor` gives every subclass."""

import pytest
import torch
import torch_npu  # noqa: F401
from torchao_npu.quantization.quant_configs import MXQuantizeConfig
from torchao_npu.quantized_tensors.mx_tensor import MXTensor


def test_torch_function_layer_is_disabled():
    """Ops must bypass the function layer and reach ``__torch_dispatch__``."""
    assert MXTensor.__torch_function__ is torch._C._disabled_torch_function_impl


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize(
    "call",
    [
        lambda x: x.view(-1),
        lambda x: x.sum(),
        lambda x: x + 1,
        lambda x: x[0],
    ],
)
def test_unimplemented_ops_raise_not_implemented(shape, call):
    """Op support is a whitelist: anything outside it raises, naming the class."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(NotImplementedError, match=type(x).__name__):
        call(x)


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize("op", ["add_", "mul_", "div_"])
def test_in_place_ops_are_rejected(shape, op):
    """Values are frozen: every mutable-schema op is rejected."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)
    qdata_before = x.qdata.clone()

    with pytest.raises(RuntimeError, match="immutable"):
        getattr(x, op)(2.0)

    assert x.qdata.shape == qdata_before.shape, (
        f"qdata shape changed after {op}: {tuple(x.qdata.shape)} != {tuple(qdata_before.shape)}"
    )
    assert torch.equal(x.qdata.view(torch.uint8), qdata_before.view(torch.uint8)), f"qdata values changed after {op}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_clone_and_detach_preserve_class_and_metadata(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    for name, y in (("clone", x.clone()), ("detach", x.detach())):
        assert type(y) is type(x)

        assert y.qdata.shape == x.qdata.shape, (
            f"{name} changed the qdata shape: {tuple(y.qdata.shape)} != {tuple(x.qdata.shape)}"
        )
        assert torch.equal(y.qdata.view(torch.uint8), x.qdata.view(torch.uint8)), f"{name} changed the qdata values"

        assert y.scale.shape == x.scale.shape, (
            f"{name} changed the scale shape: {tuple(y.scale.shape)} != {tuple(x.scale.shape)}"
        )
        assert torch.equal(y.scale.view(torch.uint8), x.scale.view(torch.uint8)), f"{name} changed the scale values"

        assert y.quant_axis == x.quant_axis
        assert y.quant_config is x.quant_config
        assert y.orig_dtype is x.orig_dtype


def test_flatten_unflatten_round_trip():
    """The declared name lists are the serialization contract DCP relies on."""
    config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    act_config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randint(0, 256, (4, 32), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config, act_quant_config=act_config, pack_axis=-1)

    data_names, attributes = x.__tensor_flatten__()
    data = {name: getattr(x, name) for name in data_names}
    y = type(x).__tensor_unflatten__(data, attributes, None, None)

    assert data_names == ["qdata", "scale"]
    assert set(attributes) == {"orig_dtype", "quant_axis", "quant_config", "act_quant_config", "pack_axis"}
    assert y.qdata.shape == x.qdata.shape, (
        f"unflattened qdata shape {tuple(y.qdata.shape)} != the original {tuple(x.qdata.shape)}"
    )
    assert torch.equal(y.qdata.view(torch.uint8), x.qdata.view(torch.uint8)), (
        "unflattened qdata values differ from the original"
    )

    assert y.scale.shape == x.scale.shape, (
        f"unflattened scale shape {tuple(y.scale.shape)} != the original {tuple(x.scale.shape)}"
    )
    assert torch.equal(y.scale.view(torch.uint8), x.scale.view(torch.uint8)), (
        "unflattened scale values differ from the original"
    )

    assert y.quant_axis == x.quant_axis
    assert y.pack_axis == x.pack_axis
    assert y.act_quant_config is x.act_quant_config


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_to_dtype_changes_only_the_logical_dtype(shape, dtype):
    """``.to(dtype)`` retags the tensor; the stored quantization is shared, not cast."""
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(dtype)

    assert y.dtype is dtype, f".to({dtype}) left the logical dtype at {y.dtype}"
    assert y.orig_dtype is dtype, f".to({dtype}) left orig_dtype at {y.orig_dtype}"
    assert y.qdata is x.qdata, f".to({dtype}) cast the stored qdata"
    assert y.scale is x.scale, f".to({dtype}) cast the stored scale"
    assert x.dtype is torch.bfloat16, f".to({dtype}) changed the original's dtype to {x.dtype}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_dtype_chains(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(torch.float32).to(torch.float16)

    assert y.dtype is torch.float16, f"chained .to calls ended at {y.dtype}, expected torch.float16"


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.float8_e4m3fn])
def test_to_unsupported_dtype_raises(dtype):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn((4, 64)).to(torch.float8_e4m3fn)
    scale = torch.full((4, 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(ValueError, match="only supports the logical dtypes"):
        x.to(dtype)


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_same_dtype_is_a_no_op(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to(torch.bfloat16)

    assert y.dtype is torch.bfloat16, f".to(same dtype) ended at {y.dtype}"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_device_moves_the_components(shape):
    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = MXTensor(qdata, scale, torch.bfloat16, -1, config)

    y = x.to("npu")

    assert y.qdata.device.type == "npu", f".to('npu') left qdata on {y.qdata.device}"
    assert y.scale.device.type == "npu", f".to('npu') left scale on {y.scale.device}"
    assert y.dtype is x.dtype, f".to('npu') changed the logical dtype to {y.dtype}"
    assert x.qdata.device.type == "cpu", f".to('npu') moved the original's qdata to {x.qdata.device}"

    # the move must not alter the values: moving the plain components ourselves
    # (bypassing MXTensor's handler) has to reproduce y's components exactly
    moved_qdata = qdata.to(y.qdata.device)
    assert torch.equal(y.qdata.view(torch.uint8), moved_qdata.view(torch.uint8)), ".to('npu') changed the qdata values"

    moved_scale = scale.to(y.scale.device)
    assert torch.equal(y.scale.view(torch.uint8), moved_scale.view(torch.uint8)), ".to('npu') changed the scale values"


@pytest.mark.parametrize("shape", [(4, 64), (2, 4, 64)])
def test_to_dtype_requires_the_class_to_declare_orig_dtype(shape):
    """``.to(dtype)`` swaps ``orig_dtype``, so a subclass has to declare it."""

    class _NoOrigDtype(MXTensor):
        tensor_attribute_names = ["quant_axis", "quant_config", "act_quant_config"]

    config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    qdata = torch.randn(shape).to(torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1, 2), 2.0, dtype=torch.float8_e8m0fnu)
    x = _NoOrigDtype(qdata, scale, torch.bfloat16, -1, config)

    with pytest.raises(RuntimeError, match="must declare"):
        x.to(torch.float32)
