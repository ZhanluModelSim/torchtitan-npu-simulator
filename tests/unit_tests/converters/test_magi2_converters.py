# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MAGI-2-preview NPU converters (MHC + multi-modality RMSNorm).

The fused MHC triton kernels and the real ``torch_npu`` ops only execute on
NPU hardware, so the CPU-runnable math-equivalence parts run against
pure-torch transcriptions:

- ``_RefMHCPreTriton`` / ``_RefMHCPostTriton`` mirror the composition of
  ``torchtitan_npu.ops.triton.mhc_triton`` (norm -> phi matmul -> sigmoid /
  sinkhorn coefficients -> pre bmm), including the kernel-specific details
  (``+eps`` on the pre sigmoid, the baked-in 2.0 post scale, and the triton
  sinkhorn schedule);
- ``_ref_npu_rms_norm`` mirrors ``torch_npu.npu_rms_norm``.

This validates the converters' argument mapping (phi transpose, alpha/bias
packing, gain = ``mhc_norm.weight + weight_bias``, per-modality segments,
the ``h_res`` transpose handed to ``MHCPostTriton``) against the phase-1
pure-torch path. The last test class is skip-guarded and only runs when the
real triton ops are importable and an NPU device is available.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torch_npu

import torchtitan_npu.converters.kernels.magi2_mhc as magi2_mhc
from torchtitan_npu.converters import registry
from torchtitan_npu.converters.kernels.magi2_mhc import (
    Magi2MHCModelConfig,
    NpuMagi2MHCConverter,
    NpuMagi2TransformerLayer,
)
from torchtitan_npu.converters.kernels.rms_norm import (
    NPURMSNorm,
    NpuMultiModalityRMSNorm,
    NpuRMSNormConverter,
)
from torchtitan_npu.models.magi2_preview.model import (
    Magi2PreviewModel,
    Modality,
    TransformerLayer,
)
from torchtitan_npu.models.magi2_preview.norms import MultiModalityRMSNorm

# Tolerances for the fused-path vs pure-torch comparison. Measured fp32
# differences sit at ~1e-6 (forward) and ~5e-5 relative (grads) near-init;
# the residual comes from the triton sinkhorn schedule / eps smoothing (see
# the magi2_mhc module docstring).
FWD_ATOL, FWD_RTOL = 1e-4, 1e-4
GRAD_ATOL, GRAD_RTOL = 1e-4, 1e-3


# ---------------------------------------------------------------------------
# Pure-torch transcriptions of the fused triton ops (CPU stand-ins)
# ---------------------------------------------------------------------------


def _ref_npu_rms_norm(x, gamma, epsilon):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + epsilon)
    return (xf * rstd * gamma.float()).to(x.dtype), rstd


class _RefMHCPreTriton:
    """Torch transcription of ``MHCPreTriton.forward`` (kernel semantics)."""

    @staticmethod
    def apply(
        x,
        weight,
        branch_alpha,
        branch_beta,
        norm_gamma,
        mhc_use_gamma=True,
        num_stream=4,
        sinkhorn_iters=20,
        eps=1e-6,
    ):
        B, S, nD = x.shape
        dtype = x.dtype
        x = x.float()
        weight = weight.float().t()
        branch_alpha = branch_alpha.float()
        branch_beta = branch_beta.float()

        x_flat = x.reshape(-1, nD)
        gamma = (
            torch.ones(nD, dtype=torch.float32)
            if not mhc_use_gamma
            else norm_gamma.float()
        )
        x_norm_flat = (
            x_flat * torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + eps) * gamma
        )
        x_proj = x_norm_flat.view(B, S, nD) @ weight

        n = num_stream
        pre = (
            torch.sigmoid(x_proj[..., :n] * branch_alpha[0] + branch_beta[:n]) + eps
        )
        post = 2.0 * torch.sigmoid(
            x_proj[..., n : 2 * n] * branch_alpha[1] + branch_beta[n : 2 * n]
        )
        base_res = branch_beta[2 * n :].view(n, n)
        comb_logits = (
            x_proj[..., 2 * n :].view(B, S, n, n) * branch_alpha[2] + base_res
        )
        # Triton sinkhorn schedule: row softmax, additive eps, column norm,
        # then (row norm, column norm) x (iters - 1).
        comb = torch.exp(comb_logits - comb_logits.amax(dim=-1, keepdim=True))
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)

        x_unflatten = x.unflatten(dim=-1, sizes=(num_stream, -1))
        y = (pre.unsqueeze(-1) * x_unflatten).sum(dim=2)
        return y.to(dtype), post, comb


class _RefMHCPostTriton:
    """Torch transcription of ``MHCPostTriton.forward`` (kernel semantics)."""

    @staticmethod
    def apply(x, residual, h_post, h_res):
        h_post = h_post.float()
        h_res = h_res.permute(0, 1, 3, 2).float()
        B, S, D = x.shape
        dtype = x.dtype
        N = h_post.shape[-1]
        x = x.float()
        residual = residual.float()

        bmm1 = h_post.unsqueeze(-1) * x.unsqueeze(-2)
        residual_unflat = residual.view(B, S, N, D)
        bmm2 = torch.matmul(h_res, residual_unflat)
        return (bmm1 + bmm2).reshape(B, S, N * D).to(dtype)


@pytest.fixture
def ref_npu_ops(monkeypatch):
    """Route the converter wrappers through the pure-torch transcriptions."""
    monkeypatch.setattr(torch_npu, "npu_rms_norm", _ref_npu_rms_norm)
    monkeypatch.setattr(magi2_mhc, "MHCPreTriton", _RefMHCPreTriton)
    monkeypatch.setattr(magi2_mhc, "MHCPostTriton", _RefMHCPostTriton)


# ---------------------------------------------------------------------------
# Small-model builders
# ---------------------------------------------------------------------------


def _tiny_config(num_stream=4, moe_layers=()):
    return Magi2PreviewModel.Config(
        num_layers=2,
        hidden_size=64,
        head_dim=16,
        num_stream=num_stream,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=64,
        time_channel_dim=64,
        dense_intermediate_size=64,
        mm_layers=[0],
        moe_layers=list(moe_layers),
        moe_num_heads=2,
        num_experts=4,
        moe_top_k=2,
        expert_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )


def _make_converter(converter_cls):
    return converter_cls(SimpleNamespace(name="magi2_preview"))


def _randomize_mhc(model, gen):
    """Move the MHC params off their degenerate init values."""
    with torch.no_grad():
        for layer in model.block.layers.values():
            for name, param in layer.named_parameters(recurse=False):
                if name.startswith("mhc_phi_fused_"):
                    param.normal_(0.0, 0.3, generator=gen)
                elif name.startswith("mhc_alpha_"):
                    param.copy_(torch.rand(1, generator=gen) * 2.0 + 0.5)
                elif name.startswith("mhc_bias_res_"):
                    param.normal_(0.0, 0.5, generator=gen)
                elif name.startswith("mhc_bias_"):
                    param.normal_(0.0, 0.3, generator=gen)
            layer.mhc_norm.weight.normal_(0.0, 0.2, generator=gen)


def _block_inputs(model, gen, cu_seqlens=None, modalities=(0, 1, 2), num_tokens=12):
    """Sorted-order block inputs mirroring ``Magi2PreviewModel.forward``."""
    layer0 = model.block.layers["0"]
    n, c = layer0.num_stream, layer0.hidden_size
    t = num_tokens
    # Random modality ids drawn from the requested set only.
    choice = torch.tensor(list(modalities), dtype=torch.long)
    modality_mapping = choice[torch.randint(0, len(modalities), (t,), generator=gen)]
    # Guarantee every requested modality is present.
    for i, modality in enumerate(modalities):
        modality_mapping[i] = modality
    sort_idx = torch.argsort(modality_mapping, stable=True)
    inv_sort_idx = torch.argsort(sort_idx, stable=True)
    m_splits = [int(v) for v in torch.bincount(modality_mapping, minlength=3)]
    x_emb = torch.randn(t, n * c, generator=gen)
    rope = torch.randn(t, layer0.attention.head_dim, generator=gen)
    return (
        x_emb.index_select(0, sort_idx),
        rope,
        sort_idx,
        inv_sort_idx,
        m_splits,
        cu_seqlens,
    )


def _converted_pair(seed=321):
    """Identical (torch, npu-converted) model pair with randomized MHC params."""
    gen = torch.Generator().manual_seed(seed)
    model_torch = Magi2PreviewModel(_tiny_config())
    model_torch.init_weights()
    _randomize_mhc(model_torch, gen)
    model_npu = copy.deepcopy(model_torch)
    _make_converter(NpuMagi2MHCConverter).convert(model_npu)
    return model_torch, model_npu, gen


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestMagi2MHCConverterRegistration:
    def test_registered_under_npu_magi2_mhc(self):
        config = registry().get("npu_magi2_mhc")
        assert config is Magi2MHCModelConfig
        assert config.name == "npu_magi2_mhc"
        assert config.model_converter is NpuMagi2MHCConverter


# ---------------------------------------------------------------------------
# Converter structure: module swap, keys, idempotency, skipping
# ---------------------------------------------------------------------------


class TestMagi2MHCConverterStructure:
    def test_convert_swaps_layers_and_keeps_state_dict(self, ref_npu_ops):
        model = Magi2PreviewModel(_tiny_config())
        model.init_weights()
        keys_before = sorted(model.state_dict().keys())
        params_before = dict(model.named_parameters())
        attention_before = model.block.layers["0"].attention

        _make_converter(NpuMagi2MHCConverter).convert(model)

        for layer in model.block.layers.values():
            assert type(layer) is NpuMagi2TransformerLayer
        # Children are shared, not rebuilt: state-dict keys and parameter
        # objects must be unchanged by the conversion.
        assert sorted(model.state_dict().keys()) == keys_before
        for name, param in model.named_parameters():
            assert param is params_before[name], f"param object changed: {name}"
        assert model.block.layers["0"].attention is attention_before

    def test_convert_is_idempotent(self, ref_npu_ops):
        model = Magi2PreviewModel(_tiny_config())
        model.init_weights()
        converter = _make_converter(NpuMagi2MHCConverter)
        converter.convert(model)
        layers_once = [id(layer) for layer in model.block.layers.values()]
        keys_once = sorted(model.state_dict().keys())

        converter.convert(model)

        assert [id(layer) for layer in model.block.layers.values()] == layers_once
        assert sorted(model.state_dict().keys()) == keys_once

    def test_convert_skips_unsupported_num_stream(self, ref_npu_ops):
        model = Magi2PreviewModel(_tiny_config(num_stream=2))
        model.init_weights()
        _make_converter(NpuMagi2MHCConverter).convert(model)
        assert all(type(layer) is TransformerLayer for layer in model.block.layers.values())

    def test_convert_skips_foreign_models(self, ref_npu_ops):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        _make_converter(NpuMagi2MHCConverter).convert(model)
        assert isinstance(model[0], nn.Linear)

    def test_convert_raises_without_triton_ops(self, monkeypatch):
        monkeypatch.setattr(magi2_mhc, "MHCPreTriton", None)
        monkeypatch.setattr(magi2_mhc, "MHCPostTriton", None)
        model = Magi2PreviewModel(_tiny_config())
        model.init_weights()
        with pytest.raises(RuntimeError, match="triton"):
            _make_converter(NpuMagi2MHCConverter).convert(model)


# ---------------------------------------------------------------------------
# MHC math equivalence vs the pure-torch path (fp32)
# ---------------------------------------------------------------------------


class TestMagi2MHCEquivalence:
    @pytest.mark.parametrize(
        "cu_seqlens",
        (None, torch.tensor([0, 5, 12]), torch.tensor([0, 3, 7, 12])),
        ids=("single-segment", "two-segments", "three-segments"),
    )
    def test_block_forward_equivalence(self, ref_npu_ops, cu_seqlens):
        model_torch, model_npu, gen = _converted_pair()
        args = _block_inputs(model_torch, gen, cu_seqlens)
        with torch.no_grad():
            ref = model_torch.block(*args)
            got = model_npu.block(*args)
        torch.testing.assert_close(got, ref, atol=FWD_ATOL, rtol=FWD_RTOL)

    def test_block_grad_equivalence(self, ref_npu_ops):
        model_torch, model_npu, gen = _converted_pair()
        args = _block_inputs(model_torch, gen, torch.tensor([0, 4, 12]))
        target = torch.randn(12, 4 * 64, generator=gen)
        torch.nn.functional.mse_loss(model_torch.block(*args), target).backward()
        torch.nn.functional.mse_loss(model_npu.block(*args), target).backward()

        for (name_t, param_t), (name_n, param_n) in zip(
            model_torch.named_parameters(),
            model_npu.named_parameters(),
            strict=True,
        ):
            assert name_t == name_n
            if param_t.grad is None and param_n.grad is None:
                continue
            assert param_t.grad is not None and param_n.grad is not None, name_t
            torch.testing.assert_close(
                param_n.grad, param_t.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL
            )

    def test_mm_layer_with_empty_modality_segment(self, ref_npu_ops):
        """mm layers must tolerate a zero-row modality segment (no audio)."""
        model_torch, model_npu, gen = _converted_pair()
        args = _block_inputs(
            model_torch, gen, None, modalities=(Modality.VIDEO, Modality.TEXT)
        )
        m_splits = args[4]
        assert 0 in m_splits
        with torch.no_grad():
            ref = model_torch.block(*args)
            got = model_npu.block(*args)
        torch.testing.assert_close(got, ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        assert torch.isfinite(got).all()

    def test_full_model_forward_with_all_converters(self, ref_npu_ops):
        """MHC + RMSNorm converters together, end-to-end on the full model."""
        from torchtitan_npu.models.magi2_preview.dataset import (
            Magi2SyntheticDataLoader,
        )

        gen = torch.Generator().manual_seed(99)
        model_torch = Magi2PreviewModel(_tiny_config(moe_layers=[1]))
        model_torch.init_weights()
        _randomize_mhc(model_torch, gen)
        model_npu = copy.deepcopy(model_torch)
        _make_converter(NpuMagi2MHCConverter).convert(model_npu)
        _make_converter(NpuRMSNormConverter).convert(model_npu)
        assert sorted(model_npu.state_dict().keys()) == sorted(
            model_torch.state_dict().keys()
        )

        loader = Magi2SyntheticDataLoader.Config().build()
        inputs, labels = next(iter(loader))
        x = inputs.pop("input")
        with torch.no_grad():
            ref = model_torch(x, **inputs)
            got = model_npu(x, **inputs)
        torch.testing.assert_close(got, ref, atol=FWD_ATOL, rtol=FWD_RTOL)
        assert got.shape == labels.shape


# ---------------------------------------------------------------------------
# MultiModalityRMSNorm converter
# ---------------------------------------------------------------------------


def _build_titan_rms_norm():
    from torchtitan.models.common.rmsnorm import RMSNorm

    return RMSNorm.Config(normalized_shape=32, eps=1e-6).build()


class _NormHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_norm = MultiModalityRMSNorm(32, num_modality=3)
        self.plain_norm = MultiModalityRMSNorm(32)
        self.titan_norm = _build_titan_rms_norm()


class TestMagi2NormConverterStructure:
    def test_convert_replaces_norms_and_keeps_state_dict(self):
        host = _NormHost()
        keys_before = sorted(host.state_dict().keys())
        weights_before = {
            name: param for name, param in host.named_parameters()
        }

        _make_converter(NpuRMSNormConverter).convert(host)

        assert type(host.mm_norm) is NpuMultiModalityRMSNorm
        assert type(host.plain_norm) is NpuMultiModalityRMSNorm
        assert type(host.titan_norm) is NPURMSNorm
        assert sorted(host.state_dict().keys()) == keys_before
        for name, param in host.named_parameters():
            assert param is weights_before[name], f"param object changed: {name}"

        # Idempotent: a second pass must not re-wrap.
        _make_converter(NpuRMSNormConverter).convert(host)
        assert type(host.mm_norm) is NpuMultiModalityRMSNorm
        assert type(host.titan_norm) is NPURMSNorm
        assert sorted(host.state_dict().keys()) == keys_before


class TestMagi2NormEquivalence:
    @pytest.mark.parametrize(
        ("num_modality", "num_patterns", "out_dtype", "shape", "m_splits"),
        (
            (1, 1, None, (10, 32), None),
            (3, 1, torch.float32, (10, 32), [2, 5, 3]),
            (3, 1, torch.float32, (10, 4, 16), [4, 0, 6]),
            (1, 4, torch.float32, (5, 4, 16), None),
            (2, 4, None, (5, 4, 16), [3, 2]),
        ),
        ids=(
            "single-modality",
            "three-modalities",
            "three-modalities-empty-segment-3d-input",
            "single-modality-multi-pattern",
            "two-modalities-multi-pattern",
        ),
    )
    def test_forward_and_grad_equivalence(
        self, ref_npu_ops, num_modality, num_patterns, out_dtype, shape, m_splits
    ):
        dim = shape[-1]
        gen = torch.Generator().manual_seed(11)
        parent = MultiModalityRMSNorm(
            dim,
            eps=1e-6,
            num_modality=num_modality,
            num_patterns=num_patterns,
            out_dtype=out_dtype,
        )
        with torch.no_grad():
            parent.weight.normal_(0.0, 0.5, generator=gen)
        npu_norm = NpuMultiModalityRMSNorm(parent)

        x_ref = torch.randn(*shape, generator=gen, requires_grad=True)
        x_npu = x_ref.detach().clone().requires_grad_(True)
        y_ref = parent(x_ref, m_splits)
        y_npu = npu_norm(x_npu, m_splits)
        torch.testing.assert_close(y_npu, y_ref, atol=1e-6, rtol=1e-6)

        # The wrapper shares the parent's weight parameter, so reset the
        # accumulated grad between the two backwards to compare them.
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)
        weight_grad_ref = parent.weight.grad.detach().clone()
        parent.weight.grad = None
        y_npu.backward(grad_out)
        weight_grad_npu = parent.weight.grad.detach().clone()

        torch.testing.assert_close(x_npu.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            weight_grad_npu, weight_grad_ref, atol=1e-6, rtol=1e-6
        )

    def test_missing_m_splits_raises(self, ref_npu_ops):
        parent = MultiModalityRMSNorm(16, num_modality=2)
        npu_norm = NpuMultiModalityRMSNorm(parent)
        x = torch.randn(4, 16)
        with pytest.raises(ValueError, match="m_splits"):
            npu_norm(x)
        with pytest.raises(ValueError, match="m_splits entries"):
            npu_norm(x, [4])

    def test_converted_attention_qk_norm_equivalence(self, ref_npu_ops):
        """q/k norms keep fp32 outputs and sorted-row segments after convert."""
        model_torch, model_npu, gen = _converted_pair()
        _make_converter(NpuRMSNormConverter).convert(model_npu)
        attn_t = model_torch.block.layers["0"].attention
        attn_n = model_npu.block.layers["0"].attention
        assert type(attn_n.q_norm) is NpuMultiModalityRMSNorm

        q = torch.randn(12, attn_t.num_heads, attn_t.head_dim, generator=gen)
        m_splits = [5, 3, 4]
        ref = attn_t.q_norm(q, m_splits)
        got = attn_n.q_norm(q, m_splits)
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Real-kernel path: only runs with importable triton ops and an NPU device
# ---------------------------------------------------------------------------


def _real_npu_ops_available() -> bool:
    if magi2_mhc.MHCPreTriton is None or magi2_mhc.MHCPostTriton is None:
        return False
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(
    not _real_npu_ops_available(),
    reason="requires importable triton MHC ops and an NPU device",
)
class TestMagi2MHCRealKernels:
    def test_block_forward_equivalence_on_npu(self):
        device = torch.device("npu")
        model_torch, model_npu, gen = _converted_pair()
        model_torch.to(device)
        model_npu.to(device)
        args = tuple(
            a.to(device) if isinstance(a, torch.Tensor) else a
            for a in _block_inputs(model_torch, gen, None)
        )
        with torch.no_grad():
            ref = model_torch.block(*args)
            got = model_npu.block(*args)
        # Real-kernel tolerance: triton schedule/eps residual (fp32).
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
