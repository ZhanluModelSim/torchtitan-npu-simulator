# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan_npu.converters.kernels.rope import (
    _ROPE_REPLACEMENTS,
    NpuRoPEConverter,
    apply_reshape_for_broadcast_complex_patch,
    npu_apply_rotary_emb_complex,
    npu_apply_rotary_emb_cos_sin,
    npu_apply_rotary_emb_single_complex,
    reshape_for_broadcast_complex,
)
from torchtitan_npu.converters.kernels.inplace_partial_rope import (
    NpuInplacePartialRoPEConverter,
    npu_apply_rotary_emb_partial_complex_,
)

# Upstream helper name. Referenced via getattr/monkeypatch with this (non-literal)
# name so the tests neither touch the module-private attribute directly
# (CodeCheck G.CLS.11) nor pass getattr a constant literal (flake8 B009).
UPSTREAM_HELPER = "_reshape_for_broadcast_complex"


def _complex_freqs(shape):
    real = torch.randn(*shape, dtype=torch.float32)
    imag = torch.randn(*shape, dtype=torch.float32)
    return torch.complex(real, imag)


def _convert_with_single_replacement(func_name, impl, model):
    converter = NpuRoPEConverter(model_spec=MagicMock())
    with patch("torchtitan_npu.converters.kernels.rope._ROPE_REPLACEMENTS", {func_name: impl}):
        converter.convert(model)


def _convert_with_recorded_replacements(model):
    replacement_calls = []

    def record_replacement(func_name, impl, target_model):
        replacement_calls.append((func_name, impl, target_model))
        return 0

    converter = NpuRoPEConverter(model_spec=MagicMock())
    with patch.object(converter, "_replace_one", side_effect=record_replacement):
        converter.convert(model)
    return replacement_calls


def _apply_partial_rope_with_test_inputs(model_mod):
    x = torch.zeros(2, 4, 3, 10, dtype=torch.float16)
    freqs_cis = torch.stack((torch.ones(4, 4), torch.zeros(4, 4)))
    positions = torch.tensor([[0, 1, 2, 3]])
    actual = model_mod.apply_partial_rotary_emb_(
        x,
        freqs_cis,
        partial_slice=[6, 10],
        inverse=True,
        positions=positions,
    )
    return x, freqs_cis, positions, actual


def _patch_cann_ops_transformer(monkeypatch, inplace_partial_rotary_mul):
    cann_ops_transformer_ops = SimpleNamespace(
        inplace_partial_rotary_mul=inplace_partial_rotary_mul,
    )
    cann_ops_transformer = SimpleNamespace(ops=cann_ops_transformer_ops)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", cann_ops_transformer)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", cann_ops_transformer_ops)


def _patch_recording_inplace_partial_rotary_mul(monkeypatch):
    calls = []

    def fake_inplace_partial_rotary_mul(x, cos, sin, *, rotary_mode, partial_slice):
        calls.append((x, cos, sin, rotary_mode, partial_slice))
        start, end = partial_slice
        x[..., start:end].add_(1.0)

    _patch_cann_ops_transformer(monkeypatch, fake_inplace_partial_rotary_mul)
    return calls


def _patch_autograd_inplace_partial_rotary_mul(monkeypatch):
    class FakeInplacePartialRotaryMulFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, partial_slice):
            ctx.mark_dirty(x)
            start, end = partial_slice
            x[..., start:end].add_(1.0)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def fake_inplace_partial_rotary_mul(x, cos, sin, *, rotary_mode, partial_slice):
        FakeInplacePartialRotaryMulFn.apply(x, partial_slice)

    _patch_cann_ops_transformer(monkeypatch, fake_inplace_partial_rotary_mul)


def _patch_inplace_partial_rope_environment(monkeypatch):
    _patch_cann_ops_transformer(monkeypatch, MagicMock())


def _patch_partial_rope_model_module(monkeypatch):
    model_module = SimpleNamespace(
        __name__="torchtitan_npu.models.deepseek_v4.model",
        apply_partial_rotary_emb_=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    return model_module


def test_rope_replacement_mapping_tracks_current_upstream_api():
    assert {
        "apply_rotary_emb_complex": npu_apply_rotary_emb_complex,
        "apply_rotary_emb_single_complex": npu_apply_rotary_emb_single_complex,
        "apply_rotary_emb_cos_sin": npu_apply_rotary_emb_cos_sin,
    } == _ROPE_REPLACEMENTS


def test_standard_rope_implementations_call_npu_rotary_mul_directly():
    assert "_apply_torch_npu_rotary_interleave" not in inspect.getsource(npu_apply_rotary_emb_complex)
    assert "_apply_torch_npu_rotary_interleave" not in inspect.getsource(npu_apply_rotary_emb_single_complex)


def test_inplace_partial_rope_converter_is_independent_from_standard_rope():
    assert not issubclass(NpuInplacePartialRoPEConverter, NpuRoPEConverter)


@pytest.mark.parametrize("func_name,impl", list(_ROPE_REPLACEMENTS.items()), ids=list(_ROPE_REPLACEMENTS))
def test_convert_invokes_replace_functions_for_each_entry(func_name, impl):
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        _convert_with_single_replacement(func_name, impl, fake_model)

    assert mock_replace.call_count >= 1
    for call in mock_replace.call_args_list:
        assert call.args[0] == func_name
        assert call.args[1] is impl


def test_convert_iterates_all_replacements():
    fake_model = MagicMock()

    replacement_calls = _convert_with_recorded_replacements(fake_model)

    assert len(replacement_calls) == len(_ROPE_REPLACEMENTS)
    called_pairs = {(func_name, impl.__name__) for func_name, impl, _ in replacement_calls}
    expected_pairs = {(name, impl.__name__) for name, impl in _ROPE_REPLACEMENTS.items()}
    assert called_pairs == expected_pairs
    assert all(target_model is fake_model for _, _, target_model in replacement_calls)


def test_inplace_partial_converter_returns_without_replacements_for_unsupported_model(monkeypatch):
    converter = NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "example.model"
    monkeypatch.setitem(sys.modules, "example.model", SimpleNamespace(__name__="example.model"))

    monkeypatch.setattr(
        "torchtitan_npu.tools.device.get_npu_device_type",
        lambda: pytest.fail("unsupported models must return before platform validation"),
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", None)

    assert converter.convert(fake_model) is None


def test_inplace_partial_converter_patches_a_matching_model_binding(monkeypatch):
    import torchtitan_npu.converters.kernels.inplace_partial_rope as partial_rope_mod

    _patch_inplace_partial_rope_environment(monkeypatch)
    model_module = SimpleNamespace(
        __name__="example.partial_model",
        apply_partial_rotary_emb_=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

    converter = partial_rope_mod.NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = model_module.__name__

    converter.convert(fake_model)

    assert model_module.apply_partial_rotary_emb_ is partial_rope_mod.npu_apply_rotary_emb_partial_complex_


def test_standard_rope_converter_does_not_modify_deepseek_v4_partial_rope(monkeypatch):
    import torchtitan_npu.models.deepseek_v4.model as deepseek_v4_model

    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"
    monkeypatch.setattr(
        deepseek_v4_model,
        "apply_partial_rotary_emb_",
        npu_apply_rotary_emb_partial_complex_,
    )

    _convert_with_recorded_replacements(fake_model)

    assert deepseek_v4_model.apply_partial_rotary_emb_ is npu_apply_rotary_emb_partial_complex_


def test_inplace_partial_converter_uses_available_op_without_platform_gating(monkeypatch):
    import torchtitan_npu.converters.kernels.inplace_partial_rope as partial_rope_mod

    monkeypatch.setattr("torchtitan_npu.tools.device.get_npu_device_type", lambda: "A3")
    _patch_inplace_partial_rope_environment(monkeypatch)
    model_module = _patch_partial_rope_model_module(monkeypatch)
    converter = partial_rope_mod.NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"

    converter.convert(fake_model)

    assert (
        model_module.apply_partial_rotary_emb_
        is partial_rope_mod.npu_apply_rotary_emb_partial_complex_
    )


def test_inplace_partial_converter_only_patches_partial_rope_binding(monkeypatch):
    import torchtitan_npu.converters.kernels.inplace_partial_rope as partial_rope_mod

    _patch_inplace_partial_rope_environment(monkeypatch)
    standard_rope = MagicMock()
    model_module = SimpleNamespace(
        __name__="example.partial_model",
        apply_partial_rotary_emb_=MagicMock(),
        apply_rotary_emb_single_complex=standard_rope,
    )
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    converter = partial_rope_mod.NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = model_module.__name__

    converter.convert(fake_model)

    assert model_module.apply_partial_rotary_emb_ is partial_rope_mod.npu_apply_rotary_emb_partial_complex_
    assert model_module.apply_rotary_emb_single_complex is standard_rope


def test_inplace_partial_converter_fails_fast_when_cann_op_is_missing(monkeypatch):
    _patch_inplace_partial_rope_environment(monkeypatch)
    _patch_partial_rope_model_module(monkeypatch)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", None)
    converter = NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"

    with pytest.raises(
        RuntimeError,
        match="npu_rope_inplace_partial requires a compatible cann_ops_transformer",
    ):
        converter.convert(fake_model)


def test_inplace_partial_converter_patches_deepseek_v4_partial_rope(monkeypatch):
    import torchtitan_npu.converters.kernels.inplace_partial_rope as partial_rope_mod
    import torchtitan_npu.models.deepseek_v4.model as deepseek_v4_model

    _patch_inplace_partial_rope_environment(monkeypatch)
    converter = partial_rope_mod.NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"
    monkeypatch.setattr(
        deepseek_v4_model,
        "apply_partial_rotary_emb_",
        deepseek_v4_model.apply_partial_rotary_emb_fallback,
    )

    converter.convert(fake_model)

    assert deepseek_v4_model.apply_partial_rotary_emb_ is partial_rope_mod.npu_apply_rotary_emb_partial_complex_


def test_deepseek_v4_partial_rope_uses_functional_fallback(monkeypatch):
    import torchtitan_npu.models.deepseek_v4.model as model_mod

    calls = []

    def fake_apply_rotary_emb(x, freqs_cis, inverse=False, positions=None):
        calls.append((x, freqs_cis, inverse, positions))
        return x + 1

    monkeypatch.setattr(model_mod, "apply_partial_rotary_emb_", model_mod.apply_partial_rotary_emb_fallback)
    monkeypatch.setattr(model_mod, "apply_rotary_emb", fake_apply_rotary_emb)

    x, freqs_cis, positions, actual = _apply_partial_rope_with_test_inputs(model_mod)

    assert len(calls) == 1
    rotary_x, rotary_freqs, inverse, rotary_positions = calls[0]
    assert rotary_x.shape == (2, 4, 3, 4)
    assert rotary_freqs is freqs_cis
    assert inverse is True
    assert rotary_positions is positions
    assert actual is not x
    torch.testing.assert_close(actual[..., :6], torch.zeros_like(actual[..., :6]))
    torch.testing.assert_close(actual[..., 6:10], torch.ones_like(actual[..., 6:10]))
    torch.testing.assert_close(x, torch.zeros_like(x))


def test_deepseek_v4_partial_rope_uses_patched_inplace_partial(monkeypatch):
    import torchtitan_npu.converters.kernels.inplace_partial_rope as partial_rope_mod
    import torchtitan_npu.models.deepseek_v4.model as model_mod

    calls = []

    def fake_partial_rotary_emb(x, freqs_cis, partial_slice, inverse=False, positions=None):
        calls.append((x, freqs_cis, partial_slice, inverse, positions))
        start, end = partial_slice
        x[..., start:end].add_(1)
        return x

    monkeypatch.setattr(partial_rope_mod, "npu_apply_rotary_emb_partial_complex_", fake_partial_rotary_emb)
    monkeypatch.setattr(model_mod, "apply_partial_rotary_emb_", model_mod.apply_partial_rotary_emb_fallback)

    _patch_inplace_partial_rope_environment(monkeypatch)
    converter = partial_rope_mod.NpuInplacePartialRoPEConverter(model_spec=MagicMock())
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.deepseek_v4.model"
    converter.convert(fake_model)

    x, freqs_cis, positions, actual = _apply_partial_rope_with_test_inputs(model_mod)

    assert actual is x
    assert len(calls) == 1
    call_x, call_freqs, partial_slice, inverse, call_positions = calls[0]
    assert call_x is x
    assert call_freqs is freqs_cis
    assert partial_slice == [6, 10]
    assert inverse is True
    assert call_positions is positions
    torch.testing.assert_close(x[..., :6], torch.zeros_like(x[..., :6]))
    torch.testing.assert_close(x[..., 6:10], torch.ones_like(x[..., 6:10]))
    torch.testing.assert_close(actual[..., :6], torch.zeros_like(actual[..., :6]))
    torch.testing.assert_close(actual[..., 6:10], torch.ones_like(actual[..., 6:10]))


def test_convert_walks_three_packages_for_npu_model():
    """
    torchtitan_npu.* model → walk three locations:
    (1) the model's own module tree,
    (2) the upstream-rewritten package (torchtitan_npu→torchtitan),
    (3) the shared torchtitan.models.common package.
    """
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan_npu.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        _convert_with_single_replacement("apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model)

    assert mock_replace.call_count == 3
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {"package": "torchtitan.models.llama3.model"}
    assert mock_replace.call_args_list[2].kwargs == {"package": "torchtitan.models.common"}


def test_convert_walks_two_packages_when_model_is_already_upstream():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.llama3.model"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        _convert_with_single_replacement("apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model)

    assert mock_replace.call_count == 2
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}
    assert mock_replace.call_args_list[1].kwargs == {"package": "torchtitan.models.common"}


def test_convert_walks_only_model_when_already_in_common_pkg():
    fake_model = MagicMock()
    fake_model.__class__.__module__ = "torchtitan.models.common.rope"

    with patch(
        "torchtitan_npu.converters.kernels.rope.replace_functions",
        return_value=0,
    ) as mock_replace:
        _convert_with_single_replacement("apply_rotary_emb_complex", npu_apply_rotary_emb_complex, fake_model)

    assert mock_replace.call_count == 1
    assert mock_replace.call_args_list[0].kwargs == {"model": fake_model}


def test_none_positions_uses_contiguous_slice():
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))

    actual = reshape_for_broadcast_complex(freqs_cis, x, None)
    expected = freqs_cis[:4].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_shared_positions_index_real_view():
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))
    positions = torch.tensor([[0, 2, 4, 6]])

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    expected = freqs_cis[positions.squeeze(0)].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_batched_positions_use_shared_first_row():
    """(bsz, seqlen) positions are treated as batch-shared (fused-op constraint):
    row 0 is selected and broadcast over the batch.
    """
    x = _complex_freqs((2, 4, 3, 8))
    freqs_cis = _complex_freqs((8, 8))
    positions = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    expected = freqs_cis[positions[0]].view(1, 4, 1, 8)

    assert actual.shape == (1, 4, 1, 8)
    torch.testing.assert_close(actual, expected)


def test_partial_rotary_inplace_helper_rejects_complex_cache(monkeypatch):
    _patch_recording_inplace_partial_rotary_mul(monkeypatch)
    x = torch.zeros(2, 4, 3, 10, dtype=torch.float16)
    invalid_cache = torch.complex(torch.ones(4, 2), torch.zeros(4, 2))

    with pytest.raises(AssertionError, match="interleaved RoPE cache"):
        npu_apply_rotary_emb_partial_complex_(x, invalid_cache, partial_slice=[6, 10])


@pytest.mark.parametrize(
    ("positions", "inverse"),
    [
        pytest.param(None, False, id="contiguous"),
        pytest.param(torch.tensor([[6, 4, 2, 0]]), True, id="positions_inverse"),
    ],
)
def test_partial_rotary_inplace_helper_uses_precomputed_rope_cache(monkeypatch, positions, inverse):
    calls = _patch_recording_inplace_partial_rotary_mul(monkeypatch)
    x = torch.zeros(2, 4, 3, 10, dtype=torch.float16)
    cos = torch.arange(32, dtype=torch.float32).view(8, 4)
    sin = cos + 100
    rope_cache = torch.stack((cos, sin))

    actual = npu_apply_rotary_emb_partial_complex_(
        x,
        rope_cache,
        partial_slice=[6, 10],
        inverse=inverse,
        positions=positions,
    )

    assert actual is x
    assert len(calls) == 1
    op_x, op_cos, op_sin, rotary_mode, partial_slice = calls[0]
    assert op_x is x
    position_index = slice(0, x.shape[1]) if positions is None else positions[0]
    expected_cos = cos[position_index].unsqueeze(0).unsqueeze(2)
    expected_sin = sin[position_index].unsqueeze(0).unsqueeze(2)
    if inverse:
        expected_sin = -expected_sin
    assert torch.equal(op_cos, expected_cos)
    assert torch.equal(op_sin, expected_sin)
    assert rotary_mode == "interleave"
    assert partial_slice == [6, 10]
    torch.testing.assert_close(x[..., :6], torch.zeros_like(x[..., :6]))
    torch.testing.assert_close(x[..., 6:10], torch.ones_like(x[..., 6:10]))


def test_partial_rotary_inplace_helper_rejects_leaf_autograd_tensor():
    x = torch.zeros(2, 4, 3, 10, dtype=torch.float32, requires_grad=True)
    freqs_cis = torch.stack((torch.ones(4, 4), torch.zeros(4, 4)))

    with pytest.raises(RuntimeError, match="requires a non-leaf tensor"):
        npu_apply_rotary_emb_partial_complex_(x, freqs_cis, partial_slice=[6, 10])


def test_partial_rotary_inplace_helper_materializes_tnd_autograd_view(monkeypatch):
    _patch_autograd_inplace_partial_rotary_mul(monkeypatch)
    source = torch.zeros(4, 3, 10, dtype=torch.float32, requires_grad=True)
    x = source * 1.0
    freqs_cis = torch.stack((torch.ones(4, 4), torch.zeros(4, 4)))
    positions = torch.tensor([[0, 1, 2, 3]])

    actual = npu_apply_rotary_emb_partial_complex_(
        x,
        freqs_cis,
        partial_slice=[6, 10],
        positions=positions,
    )

    assert actual is not x
    assert actual.shape == x.shape
    torch.testing.assert_close(actual[..., :6], torch.zeros_like(actual[..., :6]))
    torch.testing.assert_close(actual[..., 6:10], torch.ones_like(actual[..., 6:10]))
    torch.testing.assert_close(x, torch.zeros_like(x))

    actual.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize(
    ("input_shape", "expected_op_shape"),
    [
        pytest.param((4, 3, 10), (1, 4, 3, 10), id="attention"),
        pytest.param((4, 10), (1, 4, 1, 10), id="compressor"),
    ],
)
def test_partial_rotary_inplace_helper_supports_tnd_layout(monkeypatch, input_shape, expected_op_shape):
    calls = _patch_recording_inplace_partial_rotary_mul(monkeypatch)

    x = torch.zeros(input_shape, dtype=torch.float16)
    freqs_cis = torch.stack((torch.ones(4, 4), torch.zeros(4, 4)))
    positions = torch.tensor([[0, 1, 2, 3]])

    actual = npu_apply_rotary_emb_partial_complex_(x, freqs_cis, partial_slice=[6, 10], positions=positions)

    assert actual is x
    assert len(calls) == 1
    op_x, _op_cos, _op_sin, rotary_mode, partial_slice = calls[0]
    assert op_x.shape == expected_op_shape
    assert rotary_mode == "interleave"
    assert partial_slice == [6, 10]
    torch.testing.assert_close(x[..., :6], torch.zeros_like(x[..., :6]))
    torch.testing.assert_close(x[..., 6:10], torch.ones_like(x[..., 6:10]))


def test_npu_smoke_complex_index_workaround(npu_device):
    """CPU can't catch this: NPU rejects complex64 index, so naive indexing must
    fail on-device while the real-view path succeeds.
    """
    x = _complex_freqs((2, 4, 3, 8)).to(npu_device)
    freqs_cis = _complex_freqs((8, 8)).to(npu_device)
    positions = torch.tensor([[0, 2, 4, 6]], device=npu_device)

    with pytest.raises(RuntimeError):
        _ = freqs_cis[positions]

    actual = reshape_for_broadcast_complex(freqs_cis, x, positions)
    assert actual.shape == (1, 4, 1, 8)
    expected = reshape_for_broadcast_complex(freqs_cis.cpu(), x.cpu(), positions.cpu())
    torch.testing.assert_close(actual.cpu(), expected)


def test_rope_patch_installs_npu_impl_on_real_upstream_helper():
    """Sentinel against upstream rename/move: the patch must install the NPU
    implementation onto the real upstream helper. Re-apply explicitly so the
    assertion is independent of cross-test module import/reload ordering
    (another test in the suite may reset the shared common.rope module).

    NOTE: test_registry.py reloads converters.kernels.rope, which replaces
    the module-level function objects. The top-level ``from ... import``
    bindings in this file become stale. Re-fetch from the live module to
    compare against the current (post-reload) function object.
    """
    import torchtitan.models.common.rope as upstream_rope

    import torchtitan_npu.converters.kernels.rope as rope_mod

    rope_mod.apply_reshape_for_broadcast_complex_patch()
    assert getattr(upstream_rope, UPSTREAM_HELPER) is rope_mod.reshape_for_broadcast_complex


def test_apply_patch_fails_loud_when_target_missing(monkeypatch):
    """A missing target must fail loud, not silently skip."""
    import torchtitan.models.common.rope as upstream_rope

    monkeypatch.delattr(upstream_rope, UPSTREAM_HELPER, raising=False)
    with pytest.raises(RuntimeError, match=UPSTREAM_HELPER):
        apply_reshape_for_broadcast_complex_patch()
