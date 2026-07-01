# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.cost.op_cost_model import CostEstimate, OpCostModel
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


def test_unknown_op_type_returns_unknown_cost():
    model = OpCostModel()
    result = model.compute("some_op_nobody_registered", [], [], {})
    assert result == CostEstimate.unknown_cost()
    assert result.unknown is True
    assert result.flops == 0


def test_matmul_cost_matches_formula():
    model = OpCostModel()
    a = TensorMeta(name="a", shape=(8, 16), dtype="float32", device="meta")
    w = TensorMeta(name="w", shape=(16, 32), dtype="float32", device="meta", is_parameter=True)
    out = TensorMeta(name="out", shape=(8, 32), dtype="float32", device="meta")
    result = model.compute("matmul", [a, w], [out], {})
    assert result.flops == 2 * 8 * 32 * 16
    assert result.peak_mem == 8 * 32 * 4
    assert result.param_mem == 16 * 32 * 4
    assert result.unknown is False


def test_matmul_missing_inputs_returns_unknown():
    model = OpCostModel()
    out = TensorMeta(name="out", shape=(8, 32), dtype="float32", device="meta")
    result = model.compute("matmul", [], [out], {})
    assert result.unknown is True


def test_rms_norm_cost():
    model = OpCostModel()
    x = TensorMeta(name="x", shape=(2, 8, 16), dtype="float32", device="meta")
    out = TensorMeta(name="out", shape=(2, 8, 16), dtype="float32", device="meta")
    result = model.compute("rms_norm", [x], [out], {})
    assert result.flops == 5 * 2 * 8 * 16
    assert result.peak_mem == 2 * 8 * 16 * 4


def test_allreduce_cost_doubles_bytes():
    model = OpCostModel()
    t = TensorMeta(name="t", shape=(1024,), dtype="bfloat16", device="meta")
    result = model.compute("allreduce", [t], [t], {})
    assert result.comm_bytes == 1024 * 2 * 2


def test_allgather_cost_single_multiple():
    model = OpCostModel()
    t = TensorMeta(name="t", shape=(512,), dtype="float16", device="meta")
    result = model.compute("allgather", [t], [t], {})
    assert result.comm_bytes == 512 * 2


def test_moe_token_permute_is_data_move_not_flops():
    model = OpCostModel()
    tokens = TensorMeta(name="tok", shape=(64, 128), dtype="bfloat16", device="meta")
    out = TensorMeta(name="out", shape=(64, 128), dtype="bfloat16", device="meta")
    result = model.compute("moe_token_permute", [tokens], [out], {})
    assert result.flops == 0
    assert result.peak_mem == 64 * 128 * 2
