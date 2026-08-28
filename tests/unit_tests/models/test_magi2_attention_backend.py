# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MAGI-2-preview attention backends ("sdpa" vs "flex").

The "flex" backend must be numerically equivalent to the reference
per-segment softmax path: forward and backward (inputs and every parameter,
including the learned sinks) must match for single- and multi-segment
cu_seqlens, on CPU. On CPU the flex backend runs as the documented
single-call masked-SDPA mechanism (eager flex_attention has no CPU backward
kernel in torch 2.12); the flex_attention kernel mechanism covers
accelerator devices (see torchtitan_npu/models/magi2_preview/attention.py).
"""

import dataclasses

import pytest
import torch
from torch import nn


def _init_attention(attn):
    """Fill a standalone Magi2Attention like Magi2PreviewModel.init_weights.

    Sinks get small random values (instead of init zeros) so sink-gradient
    checks do not depend on the zero-init coincidence.
    """
    from torchtitan_npu.models.magi2_preview.grouped_linear import GroupedLinear
    from torchtitan_npu.models.magi2_preview.norms import MultiModalityRMSNorm

    with torch.no_grad():
        for module in attn.modules():
            if isinstance(module, MultiModalityRMSNorm):
                nn.init.zeros_(module.weight)
            elif isinstance(module, GroupedLinear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        attn.sinks.normal_(mean=0.0, std=0.02)
    return attn


def _attention_inputs(seq_len, hidden_size):
    """Build valid sorted-order inputs for Magi2Attention.forward."""
    generator = torch.Generator().manual_seed(11)
    # Interleaved modalities keep every modality non-empty and make the
    # sort a real permutation.
    modality = torch.arange(seq_len) % 3
    sort_idx = torch.argsort(modality)
    inv_sort_idx = torch.argsort(sort_idx)
    m_splits = [int(v) for v in torch.bincount(modality, minlength=3).tolist()]
    x = torch.randn(seq_len, hidden_size, generator=generator)
    # rotary_dim 24 (< head_dim 32) exercises the RoPE pass-through dims.
    rope = torch.randn(seq_len, 24, generator=generator)
    return x, rope, m_splits, sort_idx, inv_sort_idx


def _make_attention(backend, hidden_size=128, head_dim=32, sink_token_num=1):
    from torchtitan_npu.models.magi2_preview.attention import Magi2Attention

    return Magi2Attention(
        Magi2Attention.Config(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_modality=3,
            sink_token_num=sink_token_num,
            attn_backend=backend,
        )
    )


def _run_attention(attn, inputs, cu_seqlens):
    x, rope, m_splits, sort_idx, inv_sort_idx = inputs
    x = x.clone().requires_grad_(True)
    out = attn(x, rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens)
    out.sum().backward()
    param_grads = {name: p.grad for name, p in attn.named_parameters()}
    return out.detach(), x.grad, param_grads


# ---------------------------------------------------------------------------
# Config and plumbing
# ---------------------------------------------------------------------------


class TestBackendConfig:
    @pytest.mark.parametrize("backend", ("sdpa", "flex"))
    def test_attention_config_accepts_supported_backends(self, backend):
        attn = _make_attention(backend)
        assert attn.attn_backend == backend

    def test_attention_config_rejects_unknown_backend(self):
        from torchtitan_npu.models.magi2_preview.attention import Magi2Attention

        with pytest.raises(ValueError, match="attn_backend"):
            Magi2Attention(Magi2Attention.Config(attn_backend="flash"))

    def test_model_config_plumbs_backend_to_every_layer(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        config = dataclasses.replace(
            magi2_preview_configs["debug"](), attn_backend="flex"
        )
        model = Magi2PreviewModel(config)
        backends = {
            layer.attention.attn_backend for layer in model.block.layers.values()
        }
        assert backends == {"flex"}

    def test_model_config_defaults_to_sdpa(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        assert magi2_preview_configs["debug"]().attn_backend == "sdpa"
        model = Magi2PreviewModel(magi2_preview_configs["debug"]())
        for layer in model.block.layers.values():
            assert layer.attention.attn_backend == "sdpa"


# ---------------------------------------------------------------------------
# Attention-level equivalence: "flex" vs "sdpa" (fwd + bwd)
# ---------------------------------------------------------------------------


class TestAttentionBackendEquivalence:
    @pytest.mark.parametrize(
        "cu_seqlens",
        [
            pytest.param(None, id="single-segment"),
            pytest.param(torch.tensor([0, 5, 12], dtype=torch.int32), id="multi-segment"),
            pytest.param(
                torch.tensor([0, 5, 5, 12], dtype=torch.int32),
                id="empty-middle-segment",
            ),
        ],
    )
    @pytest.mark.parametrize("sink_token_num", (1, 2))
    def test_flex_matches_sdpa_fwd_bwd(self, cu_seqlens, sink_token_num):
        torch.manual_seed(3)
        sdpa = _init_attention(_make_attention("sdpa", sink_token_num=sink_token_num))
        flex = _make_attention("flex", sink_token_num=sink_token_num)
        flex.load_state_dict(sdpa.state_dict())

        inputs = _attention_inputs(seq_len=12, hidden_size=128)
        out_sdpa, x_grad_sdpa, grads_sdpa = _run_attention(sdpa, inputs, cu_seqlens)
        out_flex, x_grad_flex, grads_flex = _run_attention(flex, inputs, cu_seqlens)

        assert torch.allclose(out_flex, out_sdpa, atol=1e-4, rtol=1e-4)
        assert torch.allclose(x_grad_flex, x_grad_sdpa, atol=1e-4, rtol=1e-4)
        assert grads_sdpa.keys() == grads_flex.keys()
        for name in grads_sdpa:
            assert grads_flex[name] is not None, f"missing grad for {name}"
            assert torch.allclose(
                grads_flex[name], grads_sdpa[name], atol=1e-4, rtol=1e-4
            ), f"grad mismatch for {name}"

    @pytest.mark.parametrize(
        "cu_seqlens",
        [None, torch.tensor([0, 5, 12], dtype=torch.int32)],
        ids=["single-segment", "multi-segment"],
    )
    def test_flex_sinks_receive_nonzero_grad(self, cu_seqlens):
        torch.manual_seed(3)
        flex = _init_attention(_make_attention("flex"))
        inputs = _attention_inputs(seq_len=12, hidden_size=128)
        _, _, grads = _run_attention(flex, inputs, cu_seqlens)
        assert grads["sinks"] is not None
        assert grads["sinks"].abs().sum() > 0

    @pytest.mark.parametrize(
        "cu_seqlens",
        [None, torch.tensor([0, 5, 12], dtype=torch.int32)],
        ids=["single-segment", "multi-segment"],
    )
    def test_flex_kernel_mechanism_matches_sdpa_forward(self, cu_seqlens):
        """Forward-only check of the flex_attention kernel mechanism.

        Eager flex_attention has no CPU backward kernel (torch 2.12), so the
        kernel mechanism is exercised through the public forward under
        no_grad by routing the CPU dispatch to it.
        """
        pytest.importorskip("torch.nn.attention.flex_attention")
        torch.manual_seed(3)
        sdpa = _init_attention(_make_attention("sdpa"))
        flex = _make_attention("flex")
        flex.load_state_dict(sdpa.state_dict())
        # Force the accelerator mechanism on CPU for this forward-only check.
        flex._flex_attention_masked_sdpa = flex._flex_attention_kernel

        inputs = _attention_inputs(seq_len=12, hidden_size=128)
        x, rope, m_splits, sort_idx, inv_sort_idx = inputs
        with torch.no_grad():
            out_sdpa = sdpa(x, rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens)
            out_flex = flex(x, rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens)
        assert torch.allclose(out_flex, out_sdpa, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Full-model equivalence
# ---------------------------------------------------------------------------


class TestModelBackendEquivalence:
    def _debug_model_pair(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.attention import Magi2Attention
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        sdpa = Magi2PreviewModel(magi2_preview_configs["debug"]())
        sdpa.init_weights()
        flex = Magi2PreviewModel(
            dataclasses.replace(
                magi2_preview_configs["debug"](), attn_backend="flex"
            )
        )
        flex.load_state_dict(sdpa.state_dict())
        # Random (identical) sinks so the sink gradient signal is unambiguous;
        # reseed per model so both get the same draws in module order.
        for model in (sdpa, flex):
            generator = torch.Generator().manual_seed(5)
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, Magi2Attention):
                        module.sinks.normal_(generator=generator)
        return sdpa, flex

    def _loader_inputs(self):
        from torchtitan_npu.models.magi2_preview.dataset import (
            Magi2SyntheticDataLoader,
        )

        loader = Magi2SyntheticDataLoader.Config().build()
        inputs, labels = next(iter(loader))
        x = inputs.pop("input")
        return x, inputs, labels

    def test_full_model_flex_matches_sdpa_fwd_bwd(self):
        sdpa, flex = self._debug_model_pair()
        x, inputs, labels = self._loader_inputs()

        pred_sdpa = sdpa(x.clone(), **inputs)
        torch.nn.functional.mse_loss(pred_sdpa, labels).backward()
        pred_flex = flex(x.clone(), **inputs)
        torch.nn.functional.mse_loss(pred_flex, labels).backward()

        assert torch.allclose(pred_flex, pred_sdpa, atol=1e-4, rtol=1e-4)
        sdpa_params = dict(sdpa.named_parameters())
        for name, param in flex.named_parameters():
            assert param.grad is not None, f"missing grad for {name}"
            assert torch.allclose(
                param.grad, sdpa_params[name].grad, atol=1e-4, rtol=1e-4
            ), f"grad mismatch for {name}"

    def test_full_model_flex_sinks_receive_nonzero_grad(self):
        _, flex = self._debug_model_pair()
        x, inputs, labels = self._loader_inputs()
        pred = flex(x, **inputs)
        torch.nn.functional.mse_loss(pred, labels).backward()
        for layer in flex.block.layers.values():
            sinks = layer.attention.sinks
            assert sinks.grad is not None
            assert sinks.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# CLI overrides and validation
# ---------------------------------------------------------------------------


class TestOverridesAndCLI:
    def test_overrides_round_trip_includes_attn_backend(self):
        from torchtitan_npu.models.magi2_preview import model_registry
        from torchtitan_npu.models.magi2_preview.config_overrides import (
            Magi2PreviewModelOverrides,
        )

        original = model_registry("debug").model
        assert original.attn_backend == "sdpa"
        overrides = Magi2PreviewModelOverrides.from_model_config(original)
        overrides.attn_backend = "flex"
        assert overrides.to_model_config().attn_backend == "flex"

    def test_validate_model_overrides_rejects_unknown_backend(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.config_overrides import (
            Magi2PreviewModelOverrides,
            validate_model_overrides,
        )

        overrides = Magi2PreviewModelOverrides.from_model_config(
            magi2_preview_configs["debug"]()
        )
        overrides.attn_backend = "bogus"
        with pytest.raises(ValueError, match="attn_backend"):
            validate_model_overrides(overrides)

    @pytest.mark.parametrize(
        "module",
        ("torchtitan_npu.models.magi2_preview", "torchtitan_npu.simulator"),
    )
    def test_cli_override_attn_backend_rebuilds_model(self, module):
        from torchtitan.config import ConfigManager

        config = ConfigManager().parse_args(
            [
                "--module",
                module,
                "--config",
                "magi2_preview_smoketest",
                "--model-overrides.attn-backend",
                "flex",
            ]
        )

        model_config = config.model_spec.model
        assert model_config.attn_backend == "flex"
