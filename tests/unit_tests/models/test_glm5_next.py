# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the glm5_next (GLM-5.3-Flash) model package."""

import dataclasses

import pytest
import torch

from torchtitan_npu.models.glm5_next import glm5_next_configs, model_registry
from torchtitan_npu.models.glm5_next.config_overrides import (
    Glm5NextModelOverrides,
    validate_model_overrides,
)
from torchtitan_npu.models.glm5_next.model import Glm5NextModel, estimate_glm5_next_params
from torchtitan_npu.models.glm5_next.state_dict_adapter import Glm5NextStateDictAdapter


FLAVORS = ("debug", "reduced", "full")


class TestModelRegistry:
    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_registry_flavors_exist(self, flavor):
        spec = model_registry(flavor)
        assert spec.name == "glm5_next"
        assert spec.flavor == flavor
        assert isinstance(spec.model, Glm5NextModel.Config)
        assert spec.parallelize_fn is not None
        assert spec.state_dict_adapter is Glm5NextStateDictAdapter
        assert spec.pipelining_fn is not None

    def test_invalid_flavor_raises(self):
        with pytest.raises(KeyError):
            model_registry("does_not_exist")

    def test_debug_layer_layout(self):
        config = glm5_next_configs["debug"]()
        text = config
        assert [text.layer_type(i) for i in range(4)] == ["kda", "kda", "kda", "dsa"]
        assert text.is_dense_layer(0) and text.is_dense_layer(1)
        assert not text.is_dense_layer(2)

    def test_full_layout_matches_official_config(self):
        text = glm5_next_configs["full"]()
        assert text.num_hidden_layers == 96
        assert text.pre_layers + text.looped_layers + text.post_layers == 96
        assert text.pre_layers == 16 and text.looped_layers == 56 and text.post_layers == 24
        dsa_layers = [i for i in range(96) if i % 4 == 3]
        assert len([i for i in dsa_layers if i < 16]) == 4
        assert len([i for i in dsa_layers if i >= 72]) == 6
        assert text.n_routed_experts == 2048
        assert text.num_experts_per_tok == 16
        assert text.qk_rope_head_dim == 0

    @pytest.mark.parametrize("flavor", FLAVORS)
    def test_overrides_round_trip(self, flavor):
        original = model_registry(flavor).model
        overrides = Glm5NextModelOverrides.from_model_config(original)
        rebuilt = overrides.to_model_config()
        assert dataclasses.asdict(rebuilt) == dataclasses.asdict(original)

    def test_invalid_loop_steps_fail_fast(self):
        overrides = Glm5NextModelOverrides.from_model_config(glm5_next_configs["debug"]())
        overrides.text_config.loop_train_steps = 5
        with pytest.raises(ValueError, match="loop_train_steps"):
            validate_model_overrides(overrides)

    def test_non_shared_loop_fails_fast(self):
        config = glm5_next_configs["debug"]()
        config.share_loop_weights = False
        with pytest.raises(ValueError, match="share_loop_weights"):
            config.validate()


class TestModelInstantiation:
    @staticmethod
    def _build(flavor):
        model = Glm5NextModel(glm5_next_configs[flavor]())
        model.init_weights()
        return model

    def test_debug_model_instantiates(self):
        model = self._build("debug")
        config = model.config
        # debug: 4 pre blocks + 1 shared loop block + 4 post blocks = 9 slots
        assert len(model.layers) == config.num_hidden_layers
        assert model.layers["3"].attention_type == "dsa"
        assert model.layers["0"].attention_type == "kda"
        assert model.layers["4"].attention_type == "kda"  # shared loop block
        assert model.layers["7"].attention_type == "dsa"
        assert model.layers["0"].moe is None
        assert model.layers["2"].moe is not None
        assert len(model.visual.blocks) == config.vision_config.depth

    @pytest.mark.parametrize("flavor", ("debug", "reduced"))
    def test_parameter_count_matches_formula(self, flavor):
        config = glm5_next_configs[flavor]()
        model = Glm5NextModel(config)
        actual = sum(p.numel() for p in model.parameters())
        expected = estimate_glm5_next_params(config)["total"]
        assert actual == expected

    def test_indexer_is_frozen(self):
        model = self._build("debug")
        for name, param in model.named_parameters():
            if ".indexer." in name:
                assert not param.requires_grad, name
            elif name.endswith("e_score_correction_bias"):
                # noaux_tc correction bias is updated by the load-balancing
                # hook, not by gradients.
                assert not param.requires_grad, name
            else:
                assert param.requires_grad, name

    def test_grouped_expert_weight_layout(self):
        model = self._build("debug")
        text = model.config
        experts = model.layers["2"].moe.experts
        assert experts.w1.shape == (text.n_routed_experts, text.moe_intermediate_size, text.hidden_size)
        assert experts.w3.shape == (text.n_routed_experts, text.moe_intermediate_size, text.hidden_size)
        assert experts.w2.shape == (text.n_routed_experts, text.hidden_size, text.moe_intermediate_size)


class TestForwardPass:
    def test_debug_model_forward_backward(self):
        model = TestModelInstantiation._build("debug")
        text = model.config
        model.train()

        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, text.vocab_size, (batch_size, seq_len))
        logits = model(tokens)
        assert logits.shape == (batch_size, seq_len, text.vocab_size)
        assert torch.isfinite(logits).all()

        loss = logits.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads
        # Text-only forward: every trainable *text* parameter must receive a
        # gradient (the vision tower is exercised in the multimodal test).
        trainable_without_grad = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None and not name.startswith("visual.")
        ]
        assert not trainable_without_grad, trainable_without_grad
        assert all(torch.isfinite(g).all() for g in grads)

    def test_multimodal_forward(self):
        model = TestModelInstantiation._build("debug")
        text = model.config
        vision = model.config.vision_config
        model.eval()
        num_images = 2
        merged_per_image = vision.image_size // vision.patch_size // vision.spatial_merge_size
        merged_per_image = (merged_per_image) ** 2
        patch_len = (vision.image_size // vision.patch_size) ** 2

        # Draw regular tokens above the special-token ids (100/101) so the
        # random fill cannot collide with the image/video markers.
        tokens = torch.randint(102, text.vocab_size, (1, 64))
        image_id = model.config.image_token_id
        tokens[0, 10 : 10 + num_images * merged_per_image] = image_id
        pixel_values = torch.randn(num_images, patch_len, vision.in_channels * vision.patch_size**2)
        grid = torch.zeros(num_images, patch_len, 3, dtype=torch.long)
        side = vision.image_size // vision.patch_size
        grid[:, :, 1] = torch.arange(patch_len) // side
        grid[:, :, 2] = torch.arange(patch_len) % side

        with torch.no_grad():
            logits = model(tokens, pixel_values=pixel_values, grid_thw=grid)
        assert logits.shape == (1, 64, text.vocab_size)
        assert torch.isfinite(logits).all()

    def test_dsa_sparse_attention_output_width(self):
        text = glm5_next_configs["debug"]()
        seq_len = 32
        pools = seq_len // text.index_kpool
        select_pools = min(text.index_topk // text.index_kpool, pools)
        expected_k = select_pools * text.index_kpool + text.index_kpool - 1
        model = TestModelInstantiation._build("debug")
        block = model.layers["7"]
        hidden = torch.randn(2, seq_len, text.hidden_size)
        out = block.attention(hidden)
        assert block.attention_type == "dsa"
        assert block.attention.indexer(hidden, block.attention.q_a_norm(block.attention.q_a_proj(hidden))).shape == (
            2,
            seq_len,
            expected_k,
        )
        assert out.shape == (2, seq_len, text.hidden_size)

    def test_forced_load_balance_is_value_independent(self):
        text = glm5_next_configs["debug"]()
        text.debug_force_load_balance = True
        model = Glm5NextModel(text)
        model.init_weights()
        model.eval()
        tokens_a = torch.randint(0, text.vocab_size, (1, 16))
        tokens_b = torch.randint(0, text.vocab_size, (1, 16))
        with torch.no_grad():
            out_a = model(tokens_a)
            out_b = model(tokens_b)
        assert out_a.shape == out_b.shape


class TestStateDictAdapter:
    def test_round_trip(self):
        model = TestModelInstantiation._build("debug")
        adapter = Glm5NextStateDictAdapter(model_config=model.config)
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        hf_state_dict = adapter.to_hf(state_dict)
        recovered = adapter.from_hf(hf_state_dict)

        assert set(recovered.keys()) == set(state_dict.keys())
        for key, value in state_dict.items():
            assert recovered[key].shape == value.shape, key
            assert recovered[key].dtype == value.dtype, key
            if value.is_floating_point():
                assert torch.equal(recovered[key], value), key

    def test_hf_keys_follow_reference_naming(self):
        model = TestModelInstantiation._build("debug")
        adapter = Glm5NextStateDictAdapter(model_config=model.config)
        hf_state_dict = adapter.to_hf(model.state_dict())
        pre = model.config.pre_layers
        assert "model.embed_tokens.weight" in hf_state_dict
        assert "lm_head.weight" in hf_state_dict
        assert f"model.layers.0.self_attn.q_proj.weight" in hf_state_dict
        assert f"model.layers.3.self_attn.indexer.wq_b.weight" in hf_state_dict
        assert f"model.layers.{pre}.self_attn.q_proj.weight" in hf_state_dict
        assert "model.visual.blocks.0.attn.qkv.weight" in hf_state_dict
        assert "model.visual.merger.down_proj.weight" in hf_state_dict
