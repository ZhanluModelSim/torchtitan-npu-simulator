# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import fields

import pytest
import torch
import torch_npu
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig


@pytest.mark.parametrize(
    "elem_dtype, expected_scale_alg",
    [
        (torch.float8_e4m3fn, 1),
        (torch.float8_e5m2, 1),
        (torch.float4_e2m1fn_x2, 2),
    ],
)
def test_mx_config_infers_scale_algorithm(elem_dtype, expected_scale_alg):
    config = MXQuantizeConfig(elem_dtype=elem_dtype)

    assert config.block_size == 32
    assert config.scale_alg == expected_scale_alg


def test_mx_config_fp4_scale_alg_constraints():
    """FP4 supports scale_alg 0 or 2 only; other values are rejected."""
    assert MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, scale_alg=0).scale_alg == 0
    with pytest.raises(AssertionError, match="scale_alg=0 or 2 for FP4"):
        MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, scale_alg=1)


def test_mx_config_accepts_positive_multiples_of_32():
    """Any positive multiple of 32 is a valid MX block size."""
    assert MXQuantizeConfig(block_size=32).block_size == 32
    assert MXQuantizeConfig(block_size=64).block_size == 64


def test_mx_config_rejects_unsupported_block_size_and_dtype():
    with pytest.raises(AssertionError, match="positive multiple of 32"):
        MXQuantizeConfig(block_size=16)
    with pytest.raises(AssertionError, match="positive multiple of 32"):
        MXQuantizeConfig(block_size=0)
    with pytest.raises(AssertionError, match="positive multiple of 32"):
        MXQuantizeConfig(block_size=-32)
    with pytest.raises(AssertionError, match="elem_dtype must be one of"):
        MXQuantizeConfig(elem_dtype=torch.float32)


def test_mx_config_scale_dtype_is_read_only_e8m0():
    """scale_dtype is a constant read-only property, not a configurable field."""
    config = MXQuantizeConfig()
    assert config.scale_dtype is torch.float8_e8m0fnu
    with pytest.raises(AttributeError):
        config.scale_dtype = torch.float16
    with pytest.raises(TypeError):
        MXQuantizeConfig(scale_dtype=torch.float8_e8m0fnu)


@pytest.mark.parametrize(
    "elem_dtype, expected_elem_token, expected_matmul_token",
    [
        (torch.float8_e4m3fn, torch_npu.float8_e4m3fn, None),
        (torch.float8_e5m2, torch_npu.float8_e5m2, None),
        (torch.float4_e2m1fn_x2, torch_npu.float4_e2m1fn_x2, torch_npu.float4_e2m1fn_x2),
    ],
)
def test_mx_config_npu_dtype_properties(elem_dtype, expected_elem_token, expected_matmul_token):
    """npu_* properties return the torch_npu tokens the NPU kernels expect.

    FP8 must yield ``npu_matmul_dtype is None``: passing ``x1_dtype``/``x2_dtype``
    for FP8 raises under ``torch.compile``'s fake-tensor tracing.
    """
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    assert config.npu_elem_dtype == expected_elem_token
    assert config.npu_matmul_dtype == expected_matmul_token
    assert config.npu_scale_dtype == torch_npu.float8_e8m0fnu


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"block_size": 16}, "positive multiple of 32"),
        ({"elem_dtype": torch.float32}, "elem_dtype must be"),
        (
            {
                "elem_dtype": torch.float4_e2m1fn_x2,
                "mxfp4_fake_quantize_config": MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
            },
            "When fake quantization is enabled, elem_dtype must be",
        ),
        (
            # scale_alg keeps the FP8-only-0 constraint even in mxfp4 mode,
            # where it is not consumed: rejected, not silently ignored.
            {
                "scale_alg": 1,
                "mxfp4_fake_quantize_config": MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
            },
            "scale_alg=0 for FP8",
        ),
    ],
)
def test_block_mx_config_rejects_unsupported_kernel_options(kwargs, message):
    with pytest.raises(AssertionError, match=message):
        BlockMXQuantizeConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"elem_dtype": torch.float8_e5m2},
        {"elem_dtype": torch.float4_e2m1fn_x2, "round_mode": "floor", "dst_type_max": 7.0},
    ],
)
def test_block_mx_config_without_mxfp4_follows_mx_semantics(kwargs):
    """Without mxfp4 fake-quant, BlockMXQuantizeConfig behaves like MXQuantizeConfig.

    Differential check: for identical kwargs, every inherited field and every
    ``npu_*`` property must agree with a reference MXQuantizeConfig, and
    construction must not emit any warning. ``scale_alg`` is excluded -- its
    block-kernel constraints diverge from MX and are covered by the dedicated
    test below.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = BlockMXQuantizeConfig(**kwargs)
    ref = MXQuantizeConfig(**kwargs)

    for f in fields(MXQuantizeConfig):
        if f.name == "scale_alg":
            continue
        assert getattr(config, f.name) == getattr(ref, f.name), f"field {f.name} diverges from MXQuantizeConfig"
    assert config.npu_elem_dtype == ref.npu_elem_dtype
    assert config.npu_matmul_dtype == ref.npu_matmul_dtype
    assert config.npu_scale_dtype == ref.npu_scale_dtype
    assert config.mxfp4_fake_quantize_config is None


def test_block_mx_config_direct_path_scale_alg_constraints():
    """Direct path: FP8 supports scale_alg=0 only; FP4 inherits the MX constraint (0 or 2)."""
    # FP8: None infers 0 (not the parent's 1); only 0 is accepted.
    assert BlockMXQuantizeConfig().scale_alg == 0
    assert BlockMXQuantizeConfig(scale_alg=0).scale_alg == 0
    with pytest.raises(AssertionError, match="scale_alg=0 for FP8"):
        BlockMXQuantizeConfig(scale_alg=1)

    # FP4: None infers 2 (as in MX); 0 and 2 are accepted, others rejected.
    assert BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2).scale_alg == 2
    assert BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, scale_alg=0).scale_alg == 0
    with pytest.raises(AssertionError, match="scale_alg=0 or 2 for FP4"):
        BlockMXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2, scale_alg=1)


def test_block_mx_config_mxfp4_warns_on_ignored_fields():
    """With mxfp4 enabled, non-default dst_type_max/round_mode are ignored with a warning."""
    fp4 = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    with pytest.warns(UserWarning, match="are ignored"):
        BlockMXQuantizeConfig(round_mode="floor", mxfp4_fake_quantize_config=fp4)


def test_block_mx_config_mxfp4_defaults_do_not_warn():
    """Default construction (scale_alg left unset or the old block value 0) must stay silent."""
    fp4 = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BlockMXQuantizeConfig(mxfp4_fake_quantize_config=fp4)
        BlockMXQuantizeConfig(scale_alg=0, mxfp4_fake_quantize_config=fp4)


def test_block_mx_config_accepts_only_fp4_nested_fake_quantization():
    fp4 = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    assert BlockMXQuantizeConfig(mxfp4_fake_quantize_config=fp4).mxfp4_fake_quantize_config is fp4

    with pytest.raises(AssertionError, match="must be FP4"):
        BlockMXQuantizeConfig(mxfp4_fake_quantize_config=MXQuantizeConfig())


def test_block_mx_config_fsdp_prequantize_defaults_to_false():
    """fsdp_prequantize defaults to False (on-the-fly quantization baseline)."""
    config = BlockMXQuantizeConfig()
    assert config.fsdp_prequantize is False


def test_block_mx_config_fsdp_prequantize_can_be_enabled():
    """fsdp_prequantize can be enabled and is excluded from equality comparison."""
    config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    assert config.fsdp_prequantize is True
    # compare=False: two configs differing only in fsdp_prequantize are equal.
    assert BlockMXQuantizeConfig(fsdp_prequantize=True) == BlockMXQuantizeConfig(fsdp_prequantize=False)
