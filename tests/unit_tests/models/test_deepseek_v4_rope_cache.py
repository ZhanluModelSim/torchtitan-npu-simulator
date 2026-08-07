# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU behavior tests for DeepSeek-V4 YaRN and single-tensor RoPE."""

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
NPU_ROOT = REPO_ROOT / "torchtitan_npu"


class _ComplexRoPE(torch.nn.Module):
    @dataclass(kw_only=True, slots=True)
    class Config:
        dim: int
        max_seq_len: int
        theta: float = 10000.0
        scaling: str = "none"
        rope_factor: float = 1.0
        beta_fast: float = 32.0
        beta_slow: float = 1.0
        original_seq_len: int = 4096

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _precompute_cache(self):
        inv_freq = 1.0 / (
            self.config.theta
            ** (
                torch.arange(0, self.config.dim, 2, dtype=torch.float32)
                / self.config.dim
            )
        )
        phase = torch.outer(
            torch.arange(self.config.max_seq_len), inv_freq
        ).float()
        return torch.polar(torch.ones_like(phase), phase)

    def _reshape_cache(self, x, positions=None):
        if positions is None:
            return self.cache[: x.shape[1]].view(1, x.shape[1], 1, -1)
        return self.cache[positions].unsqueeze(2)


def _install_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, package)


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _install_torchtitan_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "torchtitan",
        "torchtitan.models",
        "torchtitan.models.common",
    ):
        _install_package(monkeypatch, name, REPO_ROOT)

    config = types.ModuleType("torchtitan.config")
    config.derive = lambda source, target: target()
    config.override = lambda **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "torchtitan.config", config)

    rope = types.ModuleType("torchtitan.models.common.rope")
    rope.ComplexRoPE = _ComplexRoPE
    monkeypatch.setitem(sys.modules, "torchtitan.models.common.rope", rope)


@pytest.fixture(scope="module")
def rope_modules():
    monkeypatch = pytest.MonkeyPatch()
    _install_torchtitan_stubs(monkeypatch)
    package_paths = {
        "torchtitan_npu": NPU_ROOT,
        "torchtitan_npu.patches": NPU_ROOT / "patches",
        "torchtitan_npu.patches.torchtitan": NPU_ROOT / "patches" / "torchtitan",
        "torchtitan_npu.patches.torchtitan.models": NPU_ROOT / "patches" / "torchtitan" / "models",
        "torchtitan_npu.patches.torchtitan.models.common": (
            NPU_ROOT / "patches" / "torchtitan" / "models" / "common"
        ),
        "torchtitan_npu.override": NPU_ROOT / "override",
        "torchtitan_npu.override.common": NPU_ROOT / "override" / "common",
        "torchtitan_npu.override.deepseek_v4": NPU_ROOT / "override" / "deepseek_v4",
    }
    for name, path in package_paths.items():
        _install_package(monkeypatch, name, path)

    single_rope = _load_module(
        monkeypatch,
        "torchtitan_npu.patches.torchtitan.models.common.rope",
        NPU_ROOT / "patches" / "torchtitan" / "models" / "common" / "rope.py",
    )

    common_rope = types.ModuleType("torchtitan_npu.override.common.rope")
    common_rope.NPUComplexRoPE = _ComplexRoPE
    common_rope.NPUFusedRoPE = _ComplexRoPE
    common_rope.NPUSingleComplexRoPE = single_rope.SingleComplexRoPE
    common_rope.NPUFusedSingleRoPE = single_rope.SingleComplexRoPE
    monkeypatch.setitem(sys.modules, "torchtitan_npu.override.common.rope", common_rope)

    dsv4_rope = _load_module(
        monkeypatch,
        "_test_deepseek_v4_rope_runtime",
        NPU_ROOT / "override" / "deepseek_v4" / "rope.py",
    )
    yield dsv4_rope, single_rope
    monkeypatch.undo()


def _rope_config(*, scaling: str) -> SimpleNamespace:
    return SimpleNamespace(
        dim=8,
        max_seq_len=32,
        theta=10000.0,
        scaling=scaling,
        original_seq_len=8,
        rope_factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
    )


@pytest.mark.cpu
def test_precompute_complex_cache_matches_standard_rope(rope_modules):
    dsv4_rope, _ = rope_modules
    config = _rope_config(scaling="none")

    cache = dsv4_rope.precompute_complex_cache_dsv4_yarn(config)

    inv_freq = 1.0 / (
        config.theta
        ** (torch.arange(0, config.dim, 2, dtype=torch.float32) / config.dim)
    )
    phase = torch.outer(torch.arange(config.max_seq_len), inv_freq).float()
    expected = torch.polar(torch.ones_like(phase), phase)
    assert cache.shape == (32, 4)
    assert cache.dtype == torch.complex64
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_yarn_scales_only_the_interpolated_frequency_bands(rope_modules):
    dsv4_rope, _ = rope_modules

    cache = dsv4_rope.precompute_complex_cache_dsv4_yarn(
        _rope_config(scaling="yarn")
    )

    # For this small deterministic configuration the YaRN correction range is
    # [0, 1]: the first band remains unchanged and all later bands are divided
    # by rope_factor.
    expected_inv_freq = torch.tensor([1.0, 0.025, 0.0025, 0.00025])
    torch.testing.assert_close(
        torch.angle(cache[1]), expected_inv_freq, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(cache.abs(), torch.ones_like(cache.abs()))
    assert torch.equal(cache[0], torch.ones(4, dtype=torch.complex64))


@pytest.mark.cpu
def test_single_complex_rope_honors_positions_and_inverse_round_trips(rope_modules):
    _, single_rope = rope_modules
    config = single_rope.SingleComplexRoPE.Config(
        dim=4,
        max_seq_len=8,
        theta=10000.0,
        scaling="none",
    )
    rope = single_rope.SingleComplexRoPE(config)
    x = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0]], [[2.0, 1.0, 4.0, 3.0]], [[1.0, -1.0, 2.0, -2.0]]],
            [[[0.5, 1.5, 2.5, 3.5]], [[-1.0, 2.0, -3.0, 4.0]], [[4.0, 3.0, 2.0, 1.0]]],
        ]
    )
    positions = torch.tensor([[0, 1, 3], [2, 4, 6]])

    rotated = rope(x, positions)

    selected_cache = rope.cache[positions].unsqueeze(2)
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    expected = torch.view_as_real(x_complex * selected_cache).flatten(-2)
    torch.testing.assert_close(rotated, expected)
    torch.testing.assert_close(rotated[0, 0], x[0, 0])

    restored = rope(rotated, positions, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)
