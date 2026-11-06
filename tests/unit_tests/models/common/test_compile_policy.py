# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whether a run may compile: the arch x degree table, and what happens when it may not.

The rule is a conjunction of two facts -- arch22's fused metadata carries a device row count, so a
block cannot be one graph there, and at one degree nothing is rebuilt per rank.  The architecture
is passed in rather than queried, so both halves are pinned without an NPU, and the refusal is
pinned to be loud: a skipped compile must never be silent.
"""

import logging
import threading
from types import SimpleNamespace

import pytest
import torch
import torch._dynamo.config
import torchtitan.distributed.spmd_types as spmd_runtime
from torch._dynamo.testing import CompileCounterWithBackend
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.testing import assert_close

from torchtitan_npu.config import CompileConfig, ModelCompileConfig
from torchtitan_npu.models.common.compile_policy import (
    compiles_under_cp,
    compile_transformer_blocks,
    skip_compile_under_cp,
)


def _dims(cp: int) -> SimpleNamespace:
    # The policy reads one attribute of ``ParallelDims``; the rest is the caller's business.
    return SimpleNamespace(cp=cp)


@pytest.mark.parametrize(
    ("cp", "a5", "expected"),
    [
        (1, True, True),
        (1, False, True),
        (2, True, True),
        (2, False, False),
        (8, False, False),
    ],
)
def test_compiles_table(cp, a5, expected):
    assert compiles_under_cp(_dims(cp), a5=a5) is expected


def test_skip_compile_turns_it_off_and_says_why(caplog):
    config = CompileConfig(model=ModelCompileConfig())

    with caplog.at_level(logging.WARNING):
        returned = skip_compile_under_cp(object(), config, _dims(2), a5=False)

    assert returned is not config
    assert returned.model is None
    assert "Skipping torch.compile" in caplog.text
    # The reason names the architecture fact, not just the degree.
    assert "arch22" in caplog.text


def test_skip_compile_keeps_the_object_when_it_may_compile():
    config = CompileConfig(model=ModelCompileConfig())
    # Same object back: the caller's config is shared with the recipe that built it.
    assert skip_compile_under_cp(object(), config, _dims(2), a5=True) is config
    assert skip_compile_under_cp(object(), config, _dims(1), a5=False) is config


def test_skip_compile_is_silent_when_not_asked():
    config = CompileConfig(model=None)
    assert skip_compile_under_cp(object(), config, _dims(2), a5=False) is config


def test_compiled_full_ac_preserves_mesh_scope_and_gradients(monkeypatch):
    # Reproduce the balanced upstream mesh-stack mutation inside the actual
    # checkpoint HOP. The oracle is x**2*w and its analytic derivatives.
    monkeypatch.setattr(spmd_runtime, "_MESH_TLS", threading.local())
    for setting in ("capture_scalar_outputs", "skip_fwd_side_effects_in_bwd_under_checkpoint"):
        monkeypatch.setattr(torch._dynamo.config, setting, False)

    class MeshScopedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            with spmd_runtime.set_current_spmd_mesh(None):
                return x.square() * self.weight

    block = MeshScopedBlock()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict({"0": checkpoint_wrapper(block)})
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    backend = CompileCounterWithBackend("aot_eager")
    try:
        with spmd_runtime.set_current_spmd_mesh(None):
            compile_transformer_blocks(model, SimpleNamespace(backend=backend, fullgraph=True))
            result = model.layers["0"](x)
            result.sum().backward()

        assert_close(result, torch.tensor([2.0, 8.0]), rtol=0, atol=0)
        assert_close(x.grad, torch.tensor([4.0, 8.0]), rtol=0, atol=0)
        assert_close(block.weight.grad, torch.tensor(5.0), rtol=0, atol=0)
        assert backend.frame_count == 1
        assert spmd_runtime._spmd_mesh_stack() == []
    finally:
        torch._dynamo.reset()
