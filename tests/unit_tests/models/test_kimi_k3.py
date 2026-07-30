# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Kimi K3 model: config registry, model instantiation, forward pass."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_debug_flavor_returns_model_spec(self):
        from torchtitan_npu.models.kimi_k3 import model_registry

        spec = model_registry("debug")
        assert spec.name == "kimi_k3"
        assert spec.flavor == "debug"
        assert spec.model is not None
        assert spec.parallelize_fn is not None
        assert spec.state_dict_adapter is not None

    def test_16layer_reduced_flavor_returns_model_spec(self):
        from torchtitan_npu.models.kimi_k3 import model_registry

        spec = model_registry("16layer_reduced")
        assert spec.name == "kimi_k3"
        assert spec.flavor == "16layer_reduced"

    def test_invalid_flavor_raises(self):
        from torchtitan_npu.models.kimi_k3 import model_registry

        with pytest.raises(KeyError):
            model_registry("nonexistent-flavor")

    def test_debug_config_layer_count(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs

        config = kimi_k3_configs["debug"]()
        assert len(config.layers) == 4

    def test_debug_config_hybrid_attention(self):
        """Debug model: layer 0 is MLA (dense), layers 1-3 are KDA (MoE)."""
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.attention import (
            KimiDeltaAttention,
            KimiGatedMLA,
        )

        config = kimi_k3_configs["debug"]()
        # Layer 0: MLA + dense FFN
        assert isinstance(config.layers[0].attention, KimiGatedMLA.Config)
        assert config.layers[0].feed_forward is not None
        assert config.layers[0].moe is None
        # Layers 1-3: KDA + MoE
        for i in range(1, 4):
            assert isinstance(config.layers[i].attention, KimiDeltaAttention.Config)
            assert config.layers[i].moe is not None
            assert config.layers[i].feed_forward is None


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    def test_debug_model_instantiates(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        assert model is not None
        assert len(model.layers) == 4

    def test_debug_model_parameter_count(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        total_params = sum(p.numel() for p in model.parameters())
        # Debug model should be small (< 100M params)
        assert total_params < 100_000_000
        assert total_params > 0


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_debug_model_forward(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        model.eval()

        batch_size, seq_len = 2, 64
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            logits = model(tokens)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_debug_model_backward(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        model.train()

        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        # Verify gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class TestAttentionModules:
    def test_kda_forward_shape(self):
        from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention

        config = KimiDeltaAttention.Config(dim=128, num_heads=4, head_dim=32)
        attn = KimiDeltaAttention(config)
        attn.eval()

        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_gated_mla_forward_shape(self):
        from torchtitan_npu.models.kimi_k3.attention import KimiGatedMLA

        config = KimiGatedMLA.Config(
            dim=128, n_heads=4, q_lora_rank=32, kv_lora_rank=32,
            qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        )
        attn = KimiGatedMLA(config)
        attn.eval()

        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (2, 16, 128)


# ---------------------------------------------------------------------------
# Feed-forward modules
# ---------------------------------------------------------------------------


class TestFeedForward:
    def test_situ_glu(self):
        from torchtitan_npu.models.kimi_k3.feed_forward import SituGLU

        act = SituGLU(beta=4.0, linear_beta=25.0)
        x = torch.randn(2, 16, 256)
        out = act(x)
        assert out.shape == (2, 16, 128)

    def test_moe_block_forward(self):
        from torchtitan_npu.models.kimi_k3.feed_forward import KimiSparseMoeBlock

        config = KimiSparseMoeBlock.Config(
            hidden_size=64,
            num_experts=4,
            top_k=2,
            moe_intermediate_size=32,
            num_shared_experts=1,
            routed_expert_hidden_size=32,
        )
        moe = KimiSparseMoeBlock(config)
        moe.eval()

        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = moe(x)
        assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# State dict adapter
# ---------------------------------------------------------------------------


class TestStateDictAdapter:
    def test_from_hf_maps_embedding(self):
        from torchtitan_npu.models.kimi_k3.state_dict_adapter import (
            KimiK3StateDictAdapter,
        )

        adapter = KimiK3StateDictAdapter(model_config=None)
        hf_dict = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(100, 64),
        }
        result = adapter.from_hf(hf_dict)
        assert "tok_embeddings.weight" in result
        assert "norm.weight" in result
        assert "output.weight" in result

    def test_from_hf_maps_layer_keys(self):
        from torchtitan_npu.models.kimi_k3.state_dict_adapter import (
            KimiK3StateDictAdapter,
        )

        adapter = KimiK3StateDictAdapter(model_config=None)
        hf_dict = {
            "model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
        }
        result = adapter.from_hf(hf_dict)
        assert "layers.0.attention_norm.weight" in result
        assert "layers.0.attention.o_proj.weight" in result


# ---------------------------------------------------------------------------
# Parallelize
# ---------------------------------------------------------------------------


class TestParallelize:
    def test_parallelize_fn_signature(self):
        from torchtitan_npu.models.kimi_k3.parallelize import parallelize_kimi_k3

        mock_model = MagicMock()
        mock_parallel_dims = MagicMock()
        mock_parallel_dims.dp_shard_enabled = False

        mock_ac = MagicMock()
        mock_ac.mode = "none"

        result = parallelize_kimi_k3(
            mock_model,
            parallel_dims=mock_parallel_dims,
            training=MagicMock(),
            model_converters=MagicMock(),
            parallelism=MagicMock(),
            compile_config=MagicMock(),
            ac_config=mock_ac,
            dump_folder="/tmp/test",
        )
        assert result is mock_model
