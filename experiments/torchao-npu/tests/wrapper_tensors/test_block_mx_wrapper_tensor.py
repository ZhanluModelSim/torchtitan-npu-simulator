# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchao_npu import ParamSwapConfig
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantization.transform import _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER
from torchao_npu.quantized_tensors.mx_tensor import MXTensor
from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

from ..testing_utils import target_devices


def test_block_mx_handler_wraps_parameter_and_preserves_requires_grad():
    parameter = nn.Parameter(torch.randn(2, 4), requires_grad=True)
    config = ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(),
        activation_config=MXQuantizeConfig(),
    )
    handler = _PARAM_SWAP_QUANTIZE_CONFIG_HANDLER[BlockMXQuantizeConfig]

    result = handler(nn.Linear(4, 2), "weight", parameter, (config,))

    assert isinstance(result.data, BlockMXTrainingWeightWrapperTensor)
    assert result.data.weight_config is config.weight_config
    assert result.data.activation_config is config.activation_config
    assert result.requires_grad


def _make_prequantized_wrapper(shape=(64, 128), fsdp_prequantize=True):
    """Build a wrapper carrying pre-quantized FP8 data + scales."""
    weight_config = BlockMXQuantizeConfig(fsdp_prequantize=fsdp_prequantize)
    activation_config = MXQuantizeConfig()
    # torch.randn does not implement a CPU kernel for FP8 dtypes, so build the
    # FP8 tensors by casting from a float source instead.
    b_q = torch.randn(*shape).to(torch.float8_e4m3fn)
    b_s1 = torch.randn(shape[0], 2, 2).to(torch.float8_e8m0fnu)
    b_s2 = torch.randn(2, shape[1], 2).to(torch.float8_e8m0fnu)
    return BlockMXTrainingWeightWrapperTensor._from_prequantized(b_q, b_s1, b_s2, weight_config, activation_config)


def test_prequantized_wrapper_has_prequantized_data():
    wrapper = _make_prequantized_wrapper()
    assert wrapper._has_prequantized_data()
    # Logical dtype stays BF16 while _data holds FP8.
    assert wrapper.dtype == torch.bfloat16
    assert wrapper._data.dtype == torch.float8_e4m3fn


def test_prequantized_wrapper_from_prequantized_preserves_configs():
    weight_config = BlockMXQuantizeConfig(fsdp_prequantize=True)
    activation_config = MXQuantizeConfig()
    b_q = torch.randn(64, 128).to(torch.float8_e4m3fn)
    b_s1 = torch.randn(64, 2, 2).to(torch.float8_e8m0fnu)
    b_s2 = torch.randn(2, 128, 2).to(torch.float8_e8m0fnu)
    wrapper = BlockMXTrainingWeightWrapperTensor._from_prequantized(
        b_q, b_s1, b_s2, weight_config, activation_config, requires_grad=True
    )
    assert wrapper.weight_config is weight_config
    assert wrapper.activation_config is activation_config
    assert wrapper.requires_grad


def test_prequantized_wrapper_transpose_scales():
    """Transposing the last two dims swaps the roles of the two scales."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128))
    b_q, b_s1, b_s2 = (
        wrapper._data,
        wrapper._scale_s1,
        wrapper._scale_s2,
    )
    new_b_q, new_s1, new_s2 = BlockMXTrainingWeightWrapperTensor._transpose_prequantized_scales(b_q, b_s1, b_s2)
    assert new_b_q.shape == (128, 64)
    # new s1 (N-dim) = old K-dim scale B_s2 transposed: (2, 128, 2) -> (128, 2, 2)
    assert new_s1.shape == (128, 2, 2)
    # new s2 (K-dim) = old N-dim scale B_s1 transposed: (64, 2, 2) -> (2, 64, 2)
    assert new_s2.shape == (2, 64, 2)


def test_prequantized_wrapper_reshape_scales_2d_to_3d():
    """A 2D->3D view (adding a leading batch dim) reshapes the scales."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128))
    b_s1, b_s2 = wrapper._scale_s1, wrapper._scale_s2
    new_s1, new_s2 = BlockMXTrainingWeightWrapperTensor._reshape_prequantized_scales(
        b_s1, b_s2, torch.Size([64, 128]), torch.Size([2, 32, 128])
    )
    assert new_s1.shape == (2, 32, 2, 2)
    assert new_s2.shape == (2, 1, 128, 2)


def test_prequantized_wrapper_can_prequantize_falls_back_when_disabled():
    """_can_prequantize returns False when fsdp_prequantize is disabled."""
    wrapper = _make_prequantized_wrapper(fsdp_prequantize=False)
    assert wrapper._can_prequantize(None) is False


def test_prequantized_wrapper_can_prequantize_requires_block_alignment():
    """_can_prequantize requires a 32-aligned last dim."""
    wrapper = _make_prequantized_wrapper(shape=(64, 128), fsdp_prequantize=True)
    # Non-32-aligned last dim -> the configuration cannot be honored.
    wrapper._data = torch.randn(64, 100).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="last dim"):
        wrapper._can_prequantize(None)


def test_prequantized_wrapper_can_prequantize_requires_scale_pack_alignment():
    """The FSDP-sharded quantized dim (axis=-2) must be 64-aligned.

    Block MX packs two adjacent 32-block scales along the quantized dim, so a
    shard aligned to 32 but not 64 rounds its own K-dim scale count up. After
    all-gather the K-dim scale has more rows than the globally quantized one
    and the dgrad ``npu_quant_matmul`` fails the "k dimension of scale and
    pertoken_scale must be equal" meta check (16P wkv: N=512, shard 32 rows).
    An enabled-but-unsatisfiable configuration raises instead of silently
    falling back, so training behavior never diverges from the configuration.
    """
    wrapper = _make_prequantized_wrapper(shape=(64, 128), fsdp_prequantize=True)
    # 32-aligned but not 64-aligned axis=-2 -> fail fast.
    wrapper._data = torch.randn(32, 128).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="fsdp_prequantize_fqns"):
        wrapper._can_prequantize(None)
    # 64-aligned axis=-2 -> pre-quantize path is taken.
    wrapper._data = torch.randn(64, 128).to(torch.float8_e4m3fn)
    assert wrapper._can_prequantize(None) is True


def test_prequantized_wrapper_shard_dim_decides_scale_pack_alignment():
    """The 64-alignment applies whenever a sharded dim IS the quantized dim.

    3D expert weights use Shard(0) (expert axis) unless the EFSDP degree
    exceeds the expert count (extensions/distributed/fsdp.py). Under Shard(0)
    each local shard keeps the quantized dim whole, so a 32- but not 64-aligned
    axis=-2 is legitimate and must not fail; under Shard(1) the same shard
    breaks the scale packing contract and must still fail. A 2D mesh that
    shards both the expert axis and the quantized dim still hits the guard on
    the split quantized dim.
    """
    wrapper = _make_prequantized_wrapper(shape=(4, 32, 128), fsdp_prequantize=True)
    wrapper._data = torch.randn(4, 32, 128).to(torch.float8_e4m3fn)
    # Shard(0): local (4,32,128) vs global (8,32,128) -> expert axis sharded.
    assert wrapper._can_prequantize(None, torch.Size((8, 32, 128))) is True
    # Shard(1): local (4,32,128) vs global (4,64,128) -> quantized dim sharded.
    with pytest.raises(ValueError, match="fsdp_prequantize_fqns"):
        wrapper._can_prequantize(None, torch.Size((4, 64, 128)))
    # Shard(0)+Shard(1): local (4,32,128) vs global (8,64,128) -> both the
    # expert axis and the quantized dim are sharded; the 32-row quantized-dim
    # shard still breaks the 64 scale packing and must fail even though the
    # expert axis is sharded too.
    with pytest.raises(ValueError, match="fsdp_prequantize_fqns"):
        wrapper._can_prequantize(None, torch.Size((8, 64, 128)))
    # Local shape == global shape: nothing is sharded here, structural no-op.
    assert wrapper._can_prequantize(None, torch.Size((4, 32, 128))) is False


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_matches_the_training_forward(device):
    """The converted single-axis weight must multiply the way the block-MX wrapped one does."""
    weight_config = BlockMXQuantizeConfig()
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert isinstance(inference_weight, MXTensor), f"got {type(inference_weight).__name__}"
    assert torch.equal(y_training, y_inference), "the inference weight does not reproduce training's forward"


@pytest.mark.parametrize("device", target_devices)
def test_to_inference_weight_with_mxfp4_matches_the_training_forward(device):
    """With mxfp4-QAT the inference weight is FP4, as training's QAT pre-pass produces."""
    weight_config = BlockMXQuantizeConfig(
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    )
    activation_config = MXQuantizeConfig()
    A = torch.randn(64, 128, dtype=torch.bfloat16, device=device)  # [M, K]
    w = torch.randn(256, 128, dtype=torch.bfloat16, device=device)  # [N, K]
    wrapped = BlockMXTrainingWeightWrapperTensor(w, weight_config=weight_config, activation_config=activation_config)

    inference_weight = wrapped.to_inference_weight()
    y_training = F.linear(A, wrapped)
    y_inference = F.linear(A, inference_weight)

    assert inference_weight.quant_config.elem_dtype is torch.float4_e2m1fn_x2, (
        f"expected an FP4 inference weight, got {inference_weight.quant_config.elem_dtype}"
    )
    assert torch.equal(y_training, y_inference), "the FP4 inference weight does not reproduce training's forward"


@pytest.mark.parametrize("device", target_devices)
@pytest.mark.parametrize(
    "shape",
    [(256, 128), (4, 128, 128)],
    ids=["matrix_rows", "experts"],
)
def test_mxfp4_communication_matches_fp8_payload(device, shape):
    """Real NPU kernels; simulate rank-order all-gather to isolate the format conversion."""
    from dataclasses import replace
    from types import SimpleNamespace

    config = BlockMXQuantizeConfig(
        fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        mxfp4_fake_quantize_config=MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2),
    )
    weight = torch.randn(shape, device=device, dtype=torch.bfloat16)
    mesh = SimpleNamespace(size=lambda: 2)
    wrappers = [
        BlockMXTrainingWeightWrapperTensor(
            shard.contiguous(), weight_config=config, activation_config=MXQuantizeConfig()
        )
        for shard in weight.chunk(2)
    ]
    results = [
        w.fsdp_pre_all_gather(mesh, weight.shape, weight.stride(), None, SimpleNamespace(param_dtype=torch.bfloat16))
        for w in wrappers
    ]
    gathered = tuple(torch.cat([r[0][i] for r in results]) for i in range(2))
    output, inner = wrappers[0].fsdp_post_all_gather(gathered, results[0][1], torch.bfloat16)
    reference = []
    for w in wrappers:
        old = BlockMXTrainingWeightWrapperTensor(
            w._data,
            weight_config=replace(config, fsdp_prequantize_dtype="fp8"),
            activation_config=MXQuantizeConfig(),
        )
        reference.append(
            old.fsdp_pre_all_gather(
                mesh, weight.shape, weight.stride(), None, SimpleNamespace(param_dtype=torch.bfloat16)
            )[0]
        )
    for actual, expected in zip(inner, (torch.cat([r[i] for r in reference]) for i in range(3)), strict=True):
        torch.testing.assert_close(actual.view(torch.uint8), expected.contiguous().view(torch.uint8), rtol=0, atol=0)
    assert output.is_contiguous()
