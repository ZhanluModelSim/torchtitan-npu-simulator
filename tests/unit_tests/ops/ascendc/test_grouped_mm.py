# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[4] / "torchtitan_npu" / "ops" / "ascendc" / "grouped_mm.py"


def _load_grouped_mm(monkeypatch):
    calls = []

    def grouped_matmul(x, weight, **kwargs):
        # CPU boundary substitute for the wrapper's 2D x 3D and split-K
        # 2D x 2D calls. It does not emulate ACLNN kernels or registration.
        calls.append((x, weight, kwargs))
        outputs = []
        start = 0
        for expert, end in enumerate(kwargs["group_list"].tolist()):
            if kwargs["group_type"] == 2:
                outputs.append(x[0][:, start:end] @ weight[0][start:end])
            else:
                outputs.append(x[0][start:end] @ weight[0][expert])
            start = end
        output = torch.stack(outputs) if kwargs["group_type"] == 2 else torch.cat(outputs)
        if kwargs["output_dtype"] is not None:
            output = output.to(kwargs["output_dtype"])
        return [output]

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_grouped_matmul=grouped_matmul),
    )
    monkeypatch.setattr(
        torch.library,
        "impl",
        lambda *args, **kwargs: lambda fn: fn,
    )

    spec = importlib.util.spec_from_file_location("grouped_mm_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize("layout", ["compact", "expanded", "padded"])
def test_grouped_mm_split_k_preserves_weight_gradient(monkeypatch, layout):
    module, calls = _load_grouped_mm(monkeypatch)
    offs = torch.tensor([3, 8], dtype=torch.int32)
    if layout == "compact":
        x = (torch.arange(32, dtype=torch.float64).reshape(8, 4) / 8).T
    elif layout == "expanded":
        x = (torch.arange(4, dtype=torch.float64).reshape(4, 1) / 8).expand(4, 8)
    else:
        x = (torch.arange(48, dtype=torch.float64).reshape(8, 6) / 8).T[:4]
    weight = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 16

    actual = module._(x, weight, offs)
    expected = torch.stack((x[:, :3] @ weight[:3], x[:, 3:] @ weight[3:]))

    fixed_x = calls[-1][0][0]
    assert fixed_x.shape == x.shape
    assert fixed_x.stride() == (1, 4)
    torch.testing.assert_close(fixed_x, x)
    if layout == "compact":
        assert fixed_x is x
    assert calls[-1][1][0] is weight
    torch.testing.assert_close(calls[-1][2]["group_list"], offs.long())
    assert calls[-1][2]["group_type"] == 2
    torch.testing.assert_close(actual, expected)


def test_grouped_mm_forward_preserves_layout_and_output(monkeypatch):
    module, calls = _load_grouped_mm(monkeypatch)
    offs = torch.tensor([3, 8], dtype=torch.int32)
    x = (torch.arange(64, dtype=torch.float64).reshape(8, 8) / 8)[:, ::2]
    weight = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3) / 16

    actual = module._(x, weight, offs, out_dtype=torch.float32)
    expert_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    expected = torch.bmm(x.unsqueeze(1), weight[expert_ids]).squeeze(1).float()

    assert calls[-1][0][0] is x
    assert calls[-1][1][0] is weight
    torch.testing.assert_close(calls[-1][2]["group_list"], offs.long())
    assert calls[-1][2]["group_type"] == 0
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rows", [0, 1])
def test_grouped_mm_split_k_normalizes_degenerate_swiglu_cat_rows(monkeypatch, rows):
    """The fused SwiGLU cat returns its gate/up backward slices with the
    packed stride; for R=0/1 PyTorch treats the transposed slice as already
    contiguous, so the wrapper must still materialize the split-K layout."""
    module, calls = _load_grouped_mm(monkeypatch)
    width = 4
    gate = torch.randn(rows, width, dtype=torch.float64, requires_grad=True)
    up = torch.randn(rows, width, dtype=torch.float64, requires_grad=True)
    packed = torch.cat((gate, up), dim=-1)
    grad_gate, _ = torch.autograd.grad(packed, (gate, up), torch.randn_like(packed))
    x = grad_gate.T
    n_out = 3
    weight = torch.arange(rows * n_out, dtype=torch.float64).reshape(rows, n_out) / 16
    offs = torch.tensor([rows], dtype=torch.int32)

    actual = module._(x, weight, offs)

    fixed_x = calls[-1][0][0]
    assert fixed_x.shape == x.shape
    assert fixed_x.stride() == (1, width)
    torch.testing.assert_close(fixed_x, x)
    torch.testing.assert_close(actual, (x @ weight).unsqueeze(0))
