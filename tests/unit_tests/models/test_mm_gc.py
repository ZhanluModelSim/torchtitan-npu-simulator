# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for mm_gc: SLA2 core, Multi-Head MoE, model skeleton, and
state-dict round trips. All tests run on CPU with the debug flavor and are
independent from the optional Triton SLA kernel."""

import pytest
import torch

from torchtitan_npu.models.mm_gc import mm_gc_configs, model_registry
from torchtitan_npu.models.mm_gc.core import (
    SparseLinearAttention,
    get_block_map,
    soft_top_k,
)
from torchtitan_npu.models.mm_gc.feed_forward import MultiHeadMoE

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


class TestConfigRegistry:
    @pytest.mark.parametrize("flavor", ("debug", "reduced", "full"))
    def test_flavors_build_configs(self, flavor):
        cfg = mm_gc_configs[flavor]()
        assert cfg.layers
        assert cfg.layers[0].attention.seq_len == cfg.seq_len

    def test_dense_moe_layout(self):
        cfg = mm_gc_configs["debug"]()
        n_layers = len(cfg.layers)
        for layer in cfg.layers:
            is_dense = layer.feed_forward is not None
            expected = layer.layer_id < 2 or layer.layer_id >= n_layers - 2
            assert is_dense == expected

    def test_unknown_flavor_fails_fast(self):
        with pytest.raises(KeyError):
            model_registry("unknown")

    def test_top_k_must_not_exceed_experts(self):
        cfg = mm_gc_configs["debug"]()
        cfg.layers[2].moe.top_k = cfg.layers[2].moe.experts_per_head + 1
        with pytest.raises(ValueError):
            cfg.build()


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def expected_params(cfg) -> int:
    first = cfg.layers[0].attention
    V, D = cfg.vocab_size, cfg.dim
    Dh = first.head_dim
    L = first.seq_len
    nb = (L + first.sla2_blkq - 1) // first.sla2_blkq
    total = V * D + D * V + D
    for layer in cfg.layers:
        attn = 4 * D * D + 2 * Dh + 2 * (Dh * Dh + Dh) + nb
        total += attn + 2 * D
        if layer.feed_forward is not None:
            total += 3 * D * layer.feed_forward.intermediate_size
        else:
            moe = layer.moe
            S, hs, E = moe.moe_num_heads, moe.moe_head_hidden_size, moe.experts_per_head
            total += 2 * D * (S * hs)
            total += S * hs * E
            total += 3 * (S * E) * hs * (hs * moe.moe_expert_inter_mult)
            total += 3 * D * moe.moe_shared_inter_dim
    return total


class TestParameterCount:
    @pytest.mark.parametrize("flavor", ("debug", "reduced"))
    def test_matches_independent_formula(self, flavor):
        cfg = mm_gc_configs[flavor]()
        with torch.device("meta"):
            model = cfg.build()
        actual = sum(p.numel() for p in model.parameters())
        assert actual == expected_params(cfg)

    def test_expert_bias_is_not_a_parameter(self):
        cfg = mm_gc_configs["debug"]()
        with torch.device("meta"):
            model = cfg.build()
        names = {n for n, _ in model.named_parameters()}
        assert not any("expert_bias" in n for n in names)


# ---------------------------------------------------------------------------
# SLA2 core
# ---------------------------------------------------------------------------


class TestSla2Core:
    def _core(self, **overrides):
        kwargs = dict(
            head_dim=32,
            topk=0.5,
            L=128,
            feature_map="softmax",
            BLKQ=64,
            BLKK=64,
            mode="train",
            stage=1,
        )
        kwargs.update(overrides)
        return SparseLinearAttention(**kwargs)

    def test_stage1_forward_backward_shape(self):
        core = self._core()
        q = torch.randn(2, 4, 128, 32, requires_grad=True)
        k = torch.randn(2, 4, 128, 32, requires_grad=True)
        v = torch.randn(2, 4, 128, 32, requires_grad=True)
        out = core(q, k, v)
        assert out.shape == q.shape
        out.sum().backward()
        assert q.grad is not None and q.grad.shape == q.shape
        assert core.proj_q.weight.grad is not None
        assert core.alpha.grad is not None

    def test_stage2_eager_fallback_shape(self):
        core = self._core()
        core.stage = 2
        q = torch.randn(1, 2, 128, 32, requires_grad=True)
        k = torch.randn(1, 2, 128, 32, requires_grad=True)
        v = torch.randn(1, 2, 128, 32, requires_grad=True)
        out = core(q, k, v)
        assert out.shape == q.shape
        assert out.dtype == core.dtype
        out.sum().backward()
        assert q.grad is not None

    def test_stage2_requires_router_path(self):
        with pytest.raises(ValueError):
            self._core(stage=2)

    def test_invalid_stage_fails_fast(self):
        with pytest.raises(ValueError):
            self._core(stage=3)

    def test_runtime_length_mismatch_fails_fast(self):
        core = self._core(L=128)
        q = torch.randn(1, 2, 64, 32)
        with pytest.raises(ValueError):
            core(q, q, q)

    def test_soft_top_k_rows_sum_near_k(self):
        scores = torch.randn(3, 5, 16, 16)
        mask = soft_top_k(scores, k=4, max_iter=200, tol=1e-4)
        sums = mask.sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, 4.0), atol=0.05)

    def test_get_block_map_stage1_soft_mask_shape(self):
        q = torch.randn(2, 3, 128, 32)
        k = torch.randn(2, 3, 128, 32)
        proj_q = torch.nn.Linear(32, 32)
        proj_k = torch.nn.Linear(32, 32)
        sparse_map, lut, topk = get_block_map(
            q, k, topk_ratio=0.5, BLKQ=64, BLKK=64,
            proj_q=proj_q, proj_k=proj_k, stage=1,
        )
        assert sparse_map.shape == (2, 3, 2, 2)
        assert lut is None
        assert topk == 1

    def test_get_block_map_stage2_hard_lut(self):
        q = torch.randn(2, 3, 128, 32)
        k = torch.randn(2, 3, 128, 32)
        proj_q = torch.nn.Linear(32, 32)
        proj_k = torch.nn.Linear(32, 32)
        sparse_map, lut, topk = get_block_map(
            q, k, topk_ratio=0.5, BLKQ=64, BLKK=64,
            proj_q=proj_q, proj_k=proj_k, stage=2,
        )
        assert sparse_map.dtype == torch.int8
        assert lut.shape == (2, 3, 2, 1)
        assert sparse_map.sum() == lut.numel()

    def test_odd_head_dim_fails_fast(self):
        with pytest.raises(ValueError):
            SparseLinearAttention(
                head_dim=31, topk=0.5, L=64, mode="train", stage=1
            )


# ---------------------------------------------------------------------------
# Multi-Head MoE
# ---------------------------------------------------------------------------


class TestMultiHeadMoE:
    def _moe(self):
        cfg = mm_gc_configs["debug"]()
        moe = MultiHeadMoE(cfg.layers[2].moe)
        with torch.no_grad():
            for param in moe.parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        return moe

    def test_forward_shape_and_routing_semantics(self):
        moe = self._moe()
        x = torch.randn(2, 8, moe.hidden_size)
        out = moe(x)
        assert out.shape == x.shape

    def test_route_scale_multiplies_probs(self):
        moe = self._moe()
        probs, indices = moe.gate(torch.randn(6, moe.moe_num_heads, moe.head_hidden_size))
        assert probs.shape == (6, moe.moe_num_heads, moe.top_k)
        assert indices.shape == (6, moe.moe_num_heads, moe.top_k)
        if moe.gate.route_norm:
            sums = probs.sum(dim=-1) / moe.gate.route_scale
            assert torch.allclose(
                sums, torch.ones_like(sums), atol=1e-5
            )

    def test_expert_bias_shifts_selection_only(self):
        moe = self._moe()
        x = torch.randn(4, moe.moe_num_heads, moe.head_hidden_size)
        _, idx_before = moe.gate(x)
        with torch.no_grad():
            moe.gate.expert_bias.zero_()
            moe.gate.expert_bias[0] = 100.0
        probs_after, idx_after = moe.gate(x)
        assert not torch.equal(idx_before, idx_after)
        assert (idx_after == 0).any()
        if moe.gate.route_norm:
            sums = probs_after.sum(dim=-1) / moe.gate.route_scale
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_invalid_head_divisibility_fails_fast(self):
        cfg = mm_gc_configs["debug"]()
        cfg.layers[2].moe.moe_num_heads = 3
        with pytest.raises(ValueError):
            cfg.build()


# ---------------------------------------------------------------------------
# Model skeleton
# ---------------------------------------------------------------------------


class TestModelSkeleton:
    def test_debug_forward_backward_loss(self):
        cfg = mm_gc_configs["debug"]()
        model = cfg.build()
        model.init_weights()
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        logits = model(tokens)
        assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, cfg.vocab_size), tokens.view(-1)
        )
        loss.backward()
        for name, param in (
            ("embedding", model.tok_embeddings.weight),
            ("sla router", model.layers["2"].attention.core.proj_q.weight),
            ("sla alpha", model.layers["2"].attention.core.alpha),
            ("moe router", model.layers["2"].moe.gate.router),
            ("experts", model.layers["2"].moe.experts.w1),
            ("output", model.output.weight),
        ):
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    def test_meta_forward_all_flavors(self):
        for flavor in ("debug", "reduced", "full"):
            cfg = mm_gc_configs[flavor]()
            with torch.device("meta"):
                model = cfg.build()
                model.init_weights()
            tokens = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device="meta")
            logits = model(tokens)
            assert logits.shape == (1, cfg.seq_len, cfg.vocab_size)

    def test_stage2_missing_router_path_fails_at_init(self):
        cfg = mm_gc_configs["debug"]()
        for layer in cfg.layers:
            layer.attention.sla2_stage = 2
            layer.attention.sla2_router_data_path = "/nonexistent"
        with pytest.raises(FileNotFoundError):
            cfg.build()


# ---------------------------------------------------------------------------
# MXFP8 quantization converter
# ---------------------------------------------------------------------------


class TestMXFP8ConverterConfig:
    def test_default_fqns_cover_all_mm_and_grouped_experts(self):
        from torchtitan_npu.converters.kernels.mm_gc_mxfp8 import (
            DEFAULT_MM_GC_MXFP8_FQNS,
            MMGcMXFP8Converter,
        )

        cfg = mm_gc_configs["debug"]()
        with torch.device("meta"):
            model = cfg.build()
        covered = set()
        for fqn, module in model.named_modules():
            is_linear = isinstance(module, torch.nn.Linear)
            is_grouped = (
                hasattr(module, "w1")
                and hasattr(module, "w3")
                and hasattr(module, "w2")
            )
            if not (is_linear or is_grouped):
                continue
            if any(target in fqn for target in DEFAULT_MM_GC_MXFP8_FQNS):
                covered.add(fqn)
        assert any("to_q" in f for f in covered)
        assert any("gate_proj" in f for f in covered)
        assert any("proj_in" in f for f in covered)
        assert any("shared_gate_proj" in f for f in covered)
        assert any("experts" in f for f in covered)

    def test_router_fqn_filters_fail_fast(self):
        from torchtitan_npu.converters.kernels.mm_gc_mxfp8 import MMGcMXFP8Converter

        with pytest.raises(ValueError):
            MMGcMXFP8Converter._validate_fqns(["moe.gate.router"])
        with pytest.raises(ValueError):
            MMGcMXFP8Converter._validate_fqns(["attention.core.proj_"])
        MMGcMXFP8Converter._validate_fqns(["moe.experts"])

    def test_empty_fqns_fail_fast(self):
        from torchtitan_npu.converters.kernels.mm_gc_mxfp8 import MMGcMXFP8Converter

        with pytest.raises(ValueError):
            MMGcMXFP8Converter._validate_fqns([])

    def test_mxfp8_trainer_configs_registered(self):
        from torchtitan_npu.models.mm_gc import config_registry

        cfg = config_registry.mm_gc_smoketest_mxfp8()
        owners = [
            getattr(c, "_owner", None).__name__
            for c in cfg.model_converters.converters
        ]
        assert any("MMGcMXFP8Converter" in (o or "") for o in owners)
        plain = config_registry.mm_gc_smoketest()
        assert list(plain.model_converters.converters) == []


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------


class TestStateDict:
    def test_round_trip(self):
        cfg = mm_gc_configs["debug"]()
        model = cfg.build()
        model.init_weights()
        sd = model.state_dict()
        model2 = cfg.build()
        model2.init_weights()
        model2.load_state_dict(sd)
        for key in (
            "tok_embeddings.weight",
            "layers.2.moe.experts.w1",
            "layers.2.attention.core.alpha",
            "layers.2.attention.core.proj_q.weight",
        ):
            assert torch.equal(sd[key], model2.state_dict()[key])

    def test_expert_weight_layout(self):
        cfg = mm_gc_configs["debug"]()
        with torch.device("meta"):
            model = cfg.build()
        moe = model.layers["2"].moe
        assert moe.experts.w1.shape[0] == moe.flatten_num_experts
        assert moe.experts.w1.shape[2] == moe.head_hidden_size
        assert moe.experts.w2.shape[1] == moe.head_hidden_size
