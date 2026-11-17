# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU protocol tests; NPU packing/conversion kernels are replaced by byte oracles."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def block_module(monkeypatch):
    pytest.importorskip("torchao")
    from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER

    # Register NPU-only custom-op schemas on the unrenamed PrivateUse1 backend.
    # No such kernel is executed by these CPU protocol tests.
    custom_op = torch.library.custom_op

    def register_custom_op(*args, **kwargs):
        if kwargs.get("device_types") == "npu":
            kwargs["device_types"] = "privateuseone"
        return custom_op(*args, **kwargs)

    monkeypatch.setattr(torch.library, "custom_op", register_custom_op)
    handlers = _QUANTIZE_CONFIG_HANDLER.copy()
    original = {k: v for k, v in sys.modules.items() if k == "torchao_npu" or k.startswith("torchao_npu.")}
    npu = ModuleType("torch_npu")
    for token in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu", "float4_e2m1fn_x2", "hifloat8"):
        setattr(npu, token, object())
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "cann_ops_nn", ModuleType("cann_ops_nn"))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[5] / "experiments" / "torchao-npu"))
    for name in original:
        del sys.modules[name]
    try:
        yield importlib.import_module("torchao_npu.wrapper_tensors.block_mx_wrapper_tensor")
    finally:
        for name in list(sys.modules):
            if name == "torchao_npu" or name.startswith("torchao_npu."):
                del sys.modules[name]
        sys.modules.update(original)
        _QUANTIZE_CONFIG_HANDLER.clear()
        _QUANTIZE_CONFIG_HANDLER.update(handlers)


@pytest.mark.parametrize("shape", [(128, 64), (4, 64, 64)], ids=["matrix_rows", "experts"])
def test_mxfp4_hooks_restore_rank_order_and_reuse_storage(block_module, monkeypatch, shape):
    mod = block_module
    primitives = importlib.import_module("torchao_npu.quantization.quant_primitives.block_mx")
    cfg = mod.BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=mod.MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )

    # Byte oracle: pack two small integers in each byte; independent decode below.
    def pack(x, **kwargs):
        assert kwargs["axis"] == -1
        values = x.to(torch.uint8)
        packed = values[..., 0::2] | (values[..., 1::2] << 4)
        scales = values[..., ::32].reshape(*x.shape[:-1], x.shape[-1] // 64, 2).contiguous()
        return packed, scales

    decoded = []
    received_scales = []

    def convert(fp4, scale, config):
        values = torch.stack((fp4 & 15, fp4 >> 4), dim=-1).flatten(-2).float()
        decoded.append(values.clone())
        received_scales.append(scale.clone())
        rows, cols = values.shape[-2:]
        s1 = torch.ones((*values.shape[:-2], rows, cols // 64, 2), dtype=torch.uint8)
        s2 = torch.ones((*values.shape[:-2], rows // 64, cols, 2), dtype=torch.uint8)
        return values.to(torch.float8_e4m3fn), s1, s2

    monkeypatch.setattr(primitives.torch_npu, "npu_dynamic_mx_quant", pack, raising=False)
    monkeypatch.setattr(mod, "mxfp4_to_block_mx", convert)
    full = (torch.arange(torch.tensor(shape).prod().item()).view(shape) // 17 % 16).to(torch.bfloat16)
    mesh = SimpleNamespace(size=lambda: 2)
    wrappers = [
        mod.BlockMXTrainingWeightWrapperTensor(
            shard.contiguous(), weight_config=cfg, activation_config=mod.MXQuantizeConfig()
        )
        for shard in full.chunk(2)
    ]
    results = [
        w.fsdp_pre_all_gather(mesh, full.shape, full.stride(), None, SimpleNamespace(param_dtype=torch.bfloat16))
        for w in wrappers
    ]
    payload = tuple(torch.cat([result[0][i] for result in results]) for i in range(2))
    assert payload[0].numel() == full.numel() // 2
    output, inner = wrappers[0].fsdp_post_all_gather(payload, results[0][1], torch.bfloat16)
    torch.testing.assert_close(output._data.float(), full.float(), rtol=0, atol=0)
    torch.testing.assert_close(decoded[0], full.float(), rtol=0, atol=0)
    expected_scale = full[..., ::32].to(torch.uint8)
    torch.testing.assert_close(received_scales[0].flatten(-2), expected_scale, rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    # Match FSDP's contiguous as_strided + Parameter construction, then release/reallocate
    # the registered storage as reshard/unshard does before the out= callback.
    output = torch.nn.Parameter(torch.as_strided(output, full.shape, full.stride()), requires_grad=True)
    saved_view = output.detach()
    sizes = [t.untyped_storage().nbytes() for t in inner]
    for tensor, size in zip(inner, sizes, strict=True):
        tensor.untyped_storage().resize_(0)
        tensor.untyped_storage().resize_(size)
    pointers = [t.data_ptr() for t in inner]
    versions = [t._version for t in inner]
    changed = (torch.zeros_like(payload[0]), payload[1])
    wrappers[0].fsdp_post_all_gather(changed, results[0][1], torch.bfloat16, out=output)
    assert [t.data_ptr() for t in inner] == pointers
    assert [t._version for t in inner] == versions
    assert output._data.untyped_storage().data_ptr() == inner[0].untyped_storage().data_ptr()
    assert torch.count_nonzero(saved_view._data.float()) == 0
    assert torch.count_nonzero(output._data.float()) == 0


@pytest.mark.parametrize(
    "shape,outer", [((2, 64, 64), (3, 64, 64)), ((4, 64, 64), (4, 128, 64))], ids=["uneven", "shard1"]
)
def test_mxfp4_hooks_reject_unsupported_sharding(block_module, shape, outer):
    mod = block_module
    cfg = mod.BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=mod.MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    weight = torch.zeros(shape, dtype=torch.bfloat16)
    wrapper = mod.BlockMXTrainingWeightWrapperTensor(
        weight, weight_config=cfg, activation_config=mod.MXQuantizeConfig()
    )
    with pytest.raises(ValueError, match="even Shard"):
        wrapper.fsdp_pre_all_gather(
            SimpleNamespace(size=lambda: 2),
            torch.Size(outer),
            weight.stride(),
            None,
            SimpleNamespace(param_dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("qat", [False, True], ids=["fp8", "qat_fp8"])
def test_default_fp8_communication_keeps_three_payloads(block_module, monkeypatch, qat):
    mod = block_module
    config = mod.BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        mxfp4_fake_quantize_config=mod.MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2) if qat else None,
    )
    weight = torch.ones((64, 64), dtype=torch.bfloat16)
    q = (torch.arange(4096).reshape(64, 64) % 16).to(torch.float8_e4m3fn)
    s1 = torch.arange(128, dtype=torch.uint8).reshape(64, 1, 2)
    s2 = torch.arange(128, dtype=torch.uint8).reshape(1, 64, 2)
    expected = (q.t(), s2.transpose(0, 1), s1.transpose(0, 1)) if qat else (q, s1, s2)
    monkeypatch.setattr(mod, "block_mx_quantize", lambda x, config, axis: expected)
    wrapper = mod.BlockMXTrainingWeightWrapperTensor(
        weight, weight_config=config, activation_config=mod.MXQuantizeConfig()
    )
    payload, metadata = wrapper.fsdp_pre_all_gather(
        SimpleNamespace(size=lambda: 2),
        torch.Size((128, 64)),
        (64, 1),
        None,
        SimpleNamespace(param_dtype=torch.bfloat16),
    )
    assert config.fsdp_prequantize_dtype == "fp8"
    for actual, reference in zip(payload, expected, strict=True):
        assert actual.is_contiguous()
        # Exercise FSDP's actual payload flattening contract, preserving logical order.
        torch.testing.assert_close(actual.view(-1).float(), reference.reshape(-1).float())
    assert metadata == ()
    output, inner = wrapper.fsdp_post_all_gather(payload, metadata, torch.bfloat16)
    assert output._data is payload[0]
    assert all(a is b for a, b in zip(inner, payload, strict=True))


def test_fp8_prequantized_write_to_out_reuses_payload_tensors(block_module):
    """FSDP reuse writes new payloads without rebinding or bumping versions."""
    mod = block_module
    config = mod.BlockMXQuantizeConfig(fsdp_prequantize=True)
    out_config = mod.BlockMXQuantizeConfig(fsdp_prequantize=True)
    activation_config = mod.MXQuantizeConfig()
    out_activation_config = mod.MXQuantizeConfig()

    q = (torch.arange(4096).reshape(64, 64) % 16).to(torch.float8_e4m3fn)
    s1 = torch.arange(128, dtype=torch.uint8).reshape(64, 1, 2)
    s2 = torch.arange(128, dtype=torch.uint8).reshape(1, 64, 2)
    wrapper = mod.BlockMXTrainingWeightWrapperTensor._from_prequantized(
        q, s1, s2, config, activation_config
    )
    out = mod.BlockMXTrainingWeightWrapperTensor._from_prequantized(
        q.clone(), s1.clone(), s2.clone(), out_config, out_activation_config
    )

    inner = (out._data, out._scale_s1, out._scale_s2)
    versions = tuple(tensor._version for tensor in inner)
    payload = (q.flip(0).contiguous(), s1.flip(0).contiguous(), s2.flip(1).contiguous())

    result = wrapper.fsdp_post_all_gather(payload, (), torch.bfloat16, out=out)

    assert result is None
    assert all(
        actual is expected
        for actual, expected in zip((out._data, out._scale_s1, out._scale_s2), inner, strict=True)
    )
    assert tuple(tensor._version for tensor in inner) == versions
    assert out.weight_config is config
    assert out.activation_config is activation_config
    for actual, expected in zip(inner, payload, strict=True):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
