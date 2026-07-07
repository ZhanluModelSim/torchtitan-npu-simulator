# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import pytest
import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan_npu.models.common import moe_parallelize


class _DummyExperts(nn.Module):
    pass


class _DummySharedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(1, 1)
        self.w2 = nn.Linear(1, 1)
        self.w3 = nn.Linear(1, 1)


class _DummyMoE(nn.Module):
    def __init__(self, *, shared_experts: nn.Module | None = None):
        super().__init__()
        self.experts = _DummyExperts()
        self.shared_experts = shared_experts
        self.score_before_experts = True


class _DummyBlock(nn.Module):
    def __init__(self, *, moe_enabled: bool = True, shared_experts: nn.Module | None = None):
        super().__init__()
        self.moe_enabled = moe_enabled
        self.moe = _DummyMoE(shared_experts=shared_experts)


class _DummyModel(nn.Module):
    def __init__(self, *blocks: _DummyBlock):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): block for i, block in enumerate(blocks)})


class _FakeDeepEPExpertParallel:
    def __init__(
        self,
        *,
        score_before_experts: bool = True,
        comm_backend: str = "deepep",
        hybridep_non_blocking_expert_capacity_factor: float | None = None,
        pad_multiple: int | None = None,
    ):
        self.score_before_experts = score_before_experts
        self.comm_backend = comm_backend
        self.hybridep_non_blocking_expert_capacity_factor = hybridep_non_blocking_expert_capacity_factor
        self.pad_multiple = pad_multiple


class _FakeTorchAOExpertParallel:
    def __init__(self, pad_multiple: int):
        self.pad_multiple = pad_multiple


def _record_parallelize_calls(monkeypatch):
    calls = []

    def fake_parallelize_module(*, module, device_mesh, parallelize_plan):
        calls.append((module, device_mesh, parallelize_plan))
        return module

    monkeypatch.setattr(moe_parallelize, "parallelize_module", fake_parallelize_module)
    return calls


def _mesh(_name: str):
    return cast("Any", object())


def test_sequence_sharded_moe_tp_plan_includes_router_and_shared_experts(monkeypatch):
    calls = _record_parallelize_calls(monkeypatch)
    tp_mesh = _mesh("tp")
    block = _DummyBlock(shared_experts=_DummySharedExperts())
    model = _DummyModel(_DummyBlock(moe_enabled=False), block)

    moe_parallelize.apply_sequence_sharded_moe_ep_tp(
        model,
        tp_mesh=tp_mesh,
        ep_mesh=None,
        etp_mesh=None,
        ep_etp_mesh=None,
    )

    assert len(calls) == 2
    assert calls[0][0] is block
    assert calls[0][1] is tp_mesh
    moe_layer_plan = calls[0][2]
    assert isinstance(moe_layer_plan["moe"], PrepareModuleInputOutput)
    assert isinstance(moe_layer_plan["moe.router.gate"], SequenceParallel)
    assert isinstance(moe_layer_plan["moe.shared_experts"], PrepareModuleInput)
    assert isinstance(moe_layer_plan["moe.shared_experts.w1"], ColwiseParallel)
    assert isinstance(moe_layer_plan["moe.shared_experts.w2"], RowwiseParallel)
    assert isinstance(moe_layer_plan["moe.shared_experts.w3"], ColwiseParallel)
    assert calls[1][0] is block.moe.experts
    assert calls[1][1] is tp_mesh
    assert isinstance(calls[1][2], moe_parallelize.TensorParallel)


@pytest.mark.parametrize(
    ("kwargs", "expected_plan_type"),
    [
        ({"comm_backend": "standard"}, moe_parallelize.ExpertParallel),
        ({"comm_backend": "deepep"}, _FakeDeepEPExpertParallel),
        ({"comm_backend": "hybridep", "pad_multiple": 8}, _FakeDeepEPExpertParallel),
        ({"comm_backend": "standard", "pad_multiple": 8}, _FakeTorchAOExpertParallel),
    ],
)
def test_sequence_sharded_moe_ep_selects_expected_expert_parallel_plan(
    monkeypatch,
    kwargs,
    expected_plan_type,
):
    monkeypatch.setattr(moe_parallelize, "DeepEPExpertParallel", _FakeDeepEPExpertParallel)
    monkeypatch.setattr(moe_parallelize, "TorchAOExpertParallel", _FakeTorchAOExpertParallel)
    calls = _record_parallelize_calls(monkeypatch)
    ep_mesh = _mesh("ep")
    block = _DummyBlock()

    moe_parallelize.apply_sequence_sharded_moe_ep_tp(
        _DummyModel(block),
        tp_mesh=None,
        ep_mesh=ep_mesh,
        etp_mesh=None,
        ep_etp_mesh=None,
        **kwargs,
    )

    assert len(calls) == 1
    assert calls[0][0] is block.moe.experts
    assert calls[0][1] is ep_mesh
    assert isinstance(calls[0][2], expected_plan_type)


def test_sequence_sharded_moe_rejects_invalid_parallel_options(monkeypatch):
    _record_parallelize_calls(monkeypatch)
    model = _DummyModel(_DummyBlock())
    mesh = _mesh("mesh")

    with pytest.raises(AssertionError, match="At least one of Tensor Parallel mesh"):
        moe_parallelize.apply_sequence_sharded_moe_ep_tp(
            model,
            tp_mesh=None,
            ep_mesh=None,
            etp_mesh=None,
            ep_etp_mesh=None,
        )

    with pytest.raises(ValueError, match="DeepEP does not support pad_multiple"):
        moe_parallelize.apply_sequence_sharded_moe_ep_tp(
            model,
            tp_mesh=None,
            ep_mesh=mesh,
            etp_mesh=None,
            ep_etp_mesh=None,
            comm_backend="deepep",
            pad_multiple=8,
        )

    with pytest.raises(ValueError, match="Unsupported MoE communication backend"):
        moe_parallelize.apply_sequence_sharded_moe_ep_tp(
            model,
            tp_mesh=None,
            ep_mesh=mesh,
            etp_mesh=None,
            ep_etp_mesh=None,
            comm_backend="unknown",
        )

    with pytest.raises(NotImplementedError, match="ETP is not supported"):
        moe_parallelize.apply_sequence_sharded_moe_ep_tp(
            model,
            tp_mesh=mesh,
            ep_mesh=mesh,
            etp_mesh=mesh,
            ep_etp_mesh=mesh,
        )
