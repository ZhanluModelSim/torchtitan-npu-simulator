# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchtitan_npu.distributed.fsdp_parameter_precision import (
    FSDP_PARAMETER_PRESERVE_DTYPE_ATTR,
    apply_fsdp_parameter_precision,
)
from torchtitan_npu.patches.torch.fsdp import parameter_precision


def test_apply_precision_pattern_marks_matching_parameter_and_is_idempotent():
    model = nn.Module()
    model.norm = nn.Linear(4, 4, bias=False)
    model.other = nn.Linear(4, 4, bias=False)

    assert apply_fsdp_parameter_precision(model, ["norm.*"]) == ("norm.weight",)
    assert apply_fsdp_parameter_precision(model, ["norm.*"]) == ("norm.weight",)
    assert getattr(model.norm.weight, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR) is True
    assert not hasattr(model.other.weight, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR)


def test_precision_patterns_support_single_and_recursive_fqn_wildcards():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    pre_attention = nn.Module()
    model.layers[0].attention = nn.Module()
    model.layers[0].attention.pre_attention = pre_attention
    pre_attention.q_norm = nn.Linear(4, 4, bias=False)
    pre_attention.compressor = nn.Module()
    pre_attention.compressor.norm = nn.Linear(4, 4, bias=False)
    pre_attention.indexer = nn.Module()
    pre_attention.indexer.compressor = nn.Module()
    pre_attention.indexer.compressor.norm = nn.Linear(4, 4, bias=False)

    marked = apply_fsdp_parameter_precision(
        model,
        [
            "layers.*.attention.pre_attention.*_norm.weight",
            "layers.*.attention.pre_attention.**.norm.weight",
        ],
    )

    assert set(marked) == {
        "layers.0.attention.pre_attention.q_norm.weight",
        "layers.0.attention.pre_attention.compressor.norm.weight",
        "layers.0.attention.pre_attention.indexer.compressor.norm.weight",
    }


def test_recursive_wildcard_can_match_zero_fqn_segments():
    model = nn.Module()
    model.norm = nn.Linear(4, 4, bias=False)

    assert apply_fsdp_parameter_precision(model, ["**.norm.weight"]) == ("norm.weight",)


def test_compressor_pattern_excludes_top_level_torchao_wrapper():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    pre_attention = nn.Module()
    model.layers[0].attention = nn.Module()
    model.layers[0].attention.pre_attention = pre_attention
    pre_attention.wkv = nn.Linear(4, 4, bias=False)
    pre_attention.wkv.weight.fsdp_pre_all_gather = lambda *args: None
    pre_attention.compressor = nn.Module()
    pre_attention.compressor.wkv = nn.Linear(4, 4, bias=False)
    pre_attention.compressor_128 = nn.Module()
    pre_attention.compressor_128.wkv = nn.Linear(4, 4, bias=False)
    pre_attention.indexer = nn.Module()
    pre_attention.indexer.compressor = nn.Module()
    pre_attention.indexer.compressor.wkv = nn.Linear(4, 4, bias=False)

    marked = apply_fsdp_parameter_precision(
        model,
        ["layers.*.attention.pre_attention.**.compressor*.wkv.weight"],
    )

    assert set(marked) == {
        "layers.0.attention.pre_attention.compressor.wkv.weight",
        "layers.0.attention.pre_attention.compressor_128.wkv.weight",
        "layers.0.attention.pre_attention.indexer.compressor.wkv.weight",
    }
    assert not hasattr(pre_attention.wkv.weight, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR)


def test_precision_patterns_deduplicate_shared_parameter_aliases():
    model = nn.Module()
    shared = nn.Linear(4, 4, bias=False)
    model.first = shared
    model.second = shared

    marked = apply_fsdp_parameter_precision(model, ["*.weight"])

    assert marked == ("first.weight", "second.weight")
    assert getattr(shared.weight, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR) is True


def test_precision_pattern_rejects_wrapper_parameter():
    model = nn.Linear(4, 4, bias=False)
    model.weight.fsdp_pre_all_gather = lambda *args: None

    with pytest.raises(ValueError, match="TorchAO wrapper"):
        apply_fsdp_parameter_precision(model, ["weight"])


def test_precision_patterns_must_match_at_least_one_parameter():
    with pytest.raises(ValueError, match="matched no parameters"):
        apply_fsdp_parameter_precision(nn.Linear(4, 4, bias=False), ["missing.weight"])


def test_unmatched_precision_pattern_warns_when_another_pattern_matches(caplog):
    model = nn.Linear(4, 4, bias=False)

    with caplog.at_level("WARNING"):
        marked = apply_fsdp_parameter_precision(model, ["weight", "missing.*"])

    assert marked == ("weight",)
    assert "pattern 'missing.*' matched 0 parameters" in caplog.text


@pytest.mark.parametrize("pattern", ["", "layers..weight", "layers.**suffix.weight"])
def test_precision_pattern_must_be_valid(pattern):
    with pytest.raises(ValueError, match="FSDP parameter precision pattern"):
        apply_fsdp_parameter_precision(nn.Linear(4, 4, bias=False), [pattern])


class _FakeFSDPParam:
    def __init__(self, preserve_dtype: bool):
        setattr(
            self,
            parameter_precision._FSDP_PARAM_PRESERVE_DTYPE_ATTR,
            preserve_dtype,
        )


def _call_patched_foreach_reduce(fsdp_params, grads, reduce_dtype):
    return parameter_precision._patched_foreach_reduce(
        fsdp_params,
        grads,
        None,
        None,
        None,
        torch.float32,
        reduce_dtype,
        torch.device("cpu"),
        None,
        None,
        None,
        True,
        None,
        None,
    )


def test_mixed_gradients_for_preserved_parameter_are_cast_to_original_dtype(monkeypatch):
    seen = {}

    def fake_foreach_reduce(*args):
        seen["grads"] = args[1]
        return "reduced"

    monkeypatch.setattr(parameter_precision, "_ORIGINAL_FOREACH_REDUCE", fake_foreach_reduce)
    grads = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.bfloat16)]

    result = _call_patched_foreach_reduce(
        [_FakeFSDPParam(preserve_dtype=True)],
        grads,
        torch.float32,
    )

    assert result == "reduced"
    assert {grad.dtype for grad in seen["grads"]} == {torch.float32}
    assert {grad.dtype for grad in grads} == {torch.float32}


def test_mixed_gradients_without_preserved_parameter_follow_native_path(monkeypatch):
    seen = {}

    def fake_foreach_reduce(*args):
        seen["grads"] = args[1]
        return "native"

    monkeypatch.setattr(parameter_precision, "_ORIGINAL_FOREACH_REDUCE", fake_foreach_reduce)
    grads = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.bfloat16)]

    result = _call_patched_foreach_reduce(
        [_FakeFSDPParam(preserve_dtype=False)],
        grads,
        torch.float32,
    )

    assert result == "native"
    assert {grad.dtype for grad in seen["grads"]} == {
        torch.float32,
        torch.bfloat16,
    }


@pytest.mark.parametrize("reduce_dtype", [None, torch.bfloat16])
def test_mixed_gradients_for_preserved_parameter_require_original_reduce_dtype(
    monkeypatch,
    reduce_dtype,
):
    monkeypatch.setattr(
        parameter_precision,
        "_ORIGINAL_FOREACH_REDUCE",
        lambda *args: None,
    )

    with pytest.raises(RuntimeError, match="to match orig_dtype"):
        _call_patched_foreach_reduce(
            [_FakeFSDPParam(preserve_dtype=True)],
            [
                torch.ones(2, dtype=torch.float32),
                torch.ones(2, dtype=torch.bfloat16),
            ],
            reduce_dtype,
        )


def test_single_fsdp_unit_supports_bf16_and_preserved_fp32_parameters(
    single_rank_process_group,
):
    class MixedPrecisionUnit(nn.Module):
        def __init__(self):
            super().__init__()
            self.low = nn.Linear(4, 4, bias=False)
            self.high = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            low = self.low(x)
            high = self.high(x.float()).to(low.dtype)
            return low + high

    model = MixedPrecisionUnit()
    apply_fsdp_parameter_precision(
        model,
        ["high.weight"],
    )
    fully_shard(
        model,
        mesh=init_device_mesh("cpu", (1,)),
        reshard_after_forward=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )

    output = model(torch.randn(2, 4))

    assert model.low.weight.dtype is torch.bfloat16
    assert model.high.weight.dtype is torch.float32
    output.float().sum().backward()
    assert model.low.weight.grad is not None
    assert model.high.weight.grad is not None
    assert model.low.weight.grad.dtype is torch.float32
    assert model.high.weight.grad.dtype is torch.float32


def test_preserved_trainable_parameter_requires_original_reduce_dtype(
    single_rank_process_group,
):
    model = nn.Linear(4, 4, bias=False)
    apply_fsdp_parameter_precision(
        model,
        ["weight"],
    )
    fully_shard(
        model,
        mesh=init_device_mesh("cpu", (1,)),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
    )

    with pytest.raises(ValueError, match="reduce_dtype to match that original dtype"):
        model(torch.randn(2, 4))
