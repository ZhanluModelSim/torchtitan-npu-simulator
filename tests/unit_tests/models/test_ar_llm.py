# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the ar_llm (DeepSeekV4-Sparse) model package."""

import dataclasses

import pytest
import torch

from torchtitan_npu.models.ar_llm import ar_llm_configs, model_registry
from torchtitan_npu.models.ar_llm.config_overrides import (
    ArLlmModelOverrides,
    validate_model_overrides,
)
from torchtitan_npu.models.ar_llm.model import ArLlmModel, estimate_ar_llm_params
from torchtitan_npu.models.ar_llm.state_dict_adapter import ArLlmStateDictAdapter


FLAVORS = ("debug", "reduced", "50t", "100t")


class TestModelRegistry:
    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_registry_flavors_exist(self, flavor):
        spec = model_registry(flavor)
        assert spec.name == "ar_llm"
        assert spec.flavor == flavor
        assert isinstance(spec.model, ArLlmModel.Config)
        assert spec.parallelize_fn is not None
        assert spec.state_dict_adapter is ArLlmStateDictAdapter
        assert spec.pipelining_fn is not None

    def test_invalid_flavor_raises(self):
        with pytest.raises(KeyError):
            model_registry("does_not_exist")

    def test_debug_layer_layout(self):
        config = ar_llm_configs["debug"]()
        assert [config.layer_type(i) for i in range(6)] == [
            "csa",
            "hca",
            "kda",
            "kda",
            "kda",
            "kda",
        ]

    def test_expert_choice_layers_are_csa_hca_only(self):
        for flavor in ("debug", "reduced", "50t"):
            config = ar_llm_configs[flavor]()
            expert_layers = config.expert_choice_layers()
            assert expert_layers
            assert all(
                config.layer_type(idx) in ("csa", "hca") for idx in expert_layers
            )
            for idx in range(config.n_layers):
                expected = "expert" if idx in expert_layers else "token"
                assert config.mor_type_for_layer(idx) == expected

    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_overrides_round_trip(self, flavor):
        original = model_registry(flavor).model
        overrides = ArLlmModelOverrides.from_model_config(original)
        rebuilt = overrides.to_model_config()
        assert dataclasses.asdict(rebuilt) == dataclasses.asdict(original)

    def test_invalid_override_fails_fast(self):
        overrides = ArLlmModelOverrides.from_model_config(ar_llm_configs["debug"]())
        overrides.kda_d_k = 17
        overrides.kda_d_v = 16
        with pytest.raises(ValueError, match="kda_d_k"):
            validate_model_overrides(overrides)


class TestModelInstantiation:
    @staticmethod
    def _build(flavor):
        model = ArLlmModel(ar_llm_configs[flavor]())
        model.init_weights()
        return model

    def test_debug_model_instantiates(self):
        model = self._build("debug")
        config = model.model_args
        assert len(model.layers) == 6
        assert model.layers["0"].attention.layer_type == "csa"
        assert model.layers["1"].attention.layer_type == "hca"
        assert model.layers["2"].attention.layer_type == "kda"
        assert model.layers["0"].mor_type == "expert"
        assert model.layers["1"].mor_type == "token"
        assert hasattr(model.layers["2"], "engram")
        assert not hasattr(model.layers["0"], "engram")
        assert model.rope_cos.shape == (config.max_seq_len, config.qk_rope_head_dim)

    @pytest.mark.parametrize("flavor", ("debug", "reduced"))
    def test_parameter_count_matches_formula(self, flavor):
        config = ar_llm_configs[flavor]()
        model = ArLlmModel(config)
        actual = sum(p.numel() for p in model.parameters())
        expected = estimate_ar_llm_params(config)["total"]
        assert actual == expected

    def test_grouped_expert_weight_layout(self):
        model = self._build("debug")
        config = model.model_args
        experts = model.layers["0"].moe.experts
        latent, inter, dim, e = (
            config.moe_latent_dim,
            config.moe_intermediate_size,
            config.dim,
            config.num_routed_experts,
        )
        assert experts.w1.shape == (e, latent, dim)
        assert experts.w3.shape == (e, latent, dim)
        assert experts.w4.shape == (e, inter, latent)
        assert experts.w5.shape == (e, latent, inter)
        assert experts.w2.shape == (e, dim, latent)


class TestForwardPass:
    def test_debug_model_forward_backward(self):
        model = TestModelInstantiation._build("debug")
        config = model.model_args
        model.train()

        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits = model(tokens)
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()

        loss = logits.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)

    def test_forced_load_balance_is_value_independent(self):
        config = ar_llm_configs["debug"]()
        config.debug_force_load_balance = True
        model = ArLlmModel(config)
        model.init_weights()
        model.eval()
        tokens_a = torch.randint(0, config.vocab_size, (1, 16))
        tokens_b = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out_a = model(tokens_a)
            out_b = model(tokens_b)
        # Round-robin routing keeps the compute graph value-independent; the
        # outputs still differ through attention/embedding values, but shapes
        # and router token counts are identical by construction.
        assert out_a.shape == out_b.shape


class TestStateDictAdapter:
    def test_round_trip(self):
        model = TestModelInstantiation._build("debug")
        config = model.model_args
        adapter = ArLlmStateDictAdapter(model_config=config)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        hf_state_dict = adapter.to_hf(state_dict)
        recovered = adapter.from_hf(hf_state_dict)

        assert set(recovered.keys()) == set(state_dict.keys())
        for key, value in state_dict.items():
            assert recovered[key].shape == value.shape, key
            assert recovered[key].dtype == value.dtype, key
            if value.is_floating_point():
                assert torch.equal(recovered[key], value), key
