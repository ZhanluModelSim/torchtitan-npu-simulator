# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Kimi K3 model: config registry, model instantiation, forward pass."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(autouse=True)
def _stub_kda_kernel(monkeypatch):
    """Keep architecture tests independent from the optional Triton KDA package."""
    from torchtitan_npu.models.kimi_k3.attention import KimiDeltaAttention

    monkeypatch.setattr(
        KimiDeltaAttention,
        "_chunk_kda",
        lambda self, q, k, v, g, beta: v,
    )


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_baseline_uses_only_fsdp_and_ep_parallelism(self):
        from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_baseline

        parallelism = kimi_k3_baseline().parallelism
        assert parallelism.data_parallel_shard_degree == -1
        assert parallelism.expert_parallel_degree == 128
        assert parallelism.data_parallel_replicate_degree == 1
        assert parallelism.tensor_parallel_degree == 1
        assert parallelism.pipeline_parallel_degree == 1
        assert parallelism.expert_tensor_parallel_degree == 1
        assert parallelism.context_parallel_degree == 1

    def test_simulator_config_reuses_baseline_parallelism(self):
        from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_baseline
        from torchtitan_npu.simulator.config_registry import kimi_k3_full_simulate

        training_config = kimi_k3_baseline()
        simulation_config = kimi_k3_full_simulate()

        assert simulation_config.parallelism == training_config.parallelism
        assert simulation_config.compile.components == training_config.compile.components
        assert simulation_config.compile.enable is False

    def test_smoketest_enables_required_npu_converters(self):
        from torchtitan_npu.models.kimi_k3.config_registry import (
            kimi_k3_smoketest,
        )

        config = kimi_k3_smoketest()
        converter_names = {
            converter._owner._model_config.name
            for converter in config.model_converters.converters
        }
        assert converter_names == {
            "npu_rms_norm",
            "npu_rope",
            "npu_kimi_k3_moe",
        }

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
        """Debug model follows the official three-KDA-then-one-MLA pattern."""
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.attention import (
            KimiDeltaAttention,
            KimiGatedMLA,
        )

        config = kimi_k3_configs["debug"]()
        # Layer 0: KDA + dense FFN
        assert isinstance(config.layers[0].attention, KimiDeltaAttention.Config)
        assert config.layers[0].feed_forward is not None
        assert config.layers[0].moe is None
        # Layers 1-2: KDA + MoE
        for i in range(1, 3):
            assert isinstance(config.layers[i].attention, KimiDeltaAttention.Config)
            assert config.layers[i].moe is not None
            assert config.layers[i].feed_forward is None
        # Layer 3: MLA + MoE
        assert isinstance(config.layers[3].attention, KimiGatedMLA.Config)
        assert config.layers[3].moe is not None

    def test_full_config_enables_attention_residuals(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.attention import KimiGatedMLA

        config = kimi_k3_configs["full"]()
        assert config.attn_res_block_size == 12
        assert all(
            layer.layer_id == layer_id
            and layer.attn_res_block_size == config.attn_res_block_size
            for layer_id, layer in enumerate(config.layers)
        )
        assert all(
            config.layers[layer_id].attention.head_dim == 128
            for layer_id in (0, 1, 2, 90)
        )
        assert isinstance(config.layers[3].attention, KimiGatedMLA.Config)
        assert isinstance(config.layers[92].attention, KimiGatedMLA.Config)

    def test_runtime_config_enables_deterministic_moe_routing(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs

        config = kimi_k3_configs["debug"]()
        config.update_from_config(
            trainer_config=SimpleNamespace(
                debug=SimpleNamespace(moe_force_load_balance=True)
            )
        )
        assert all(
            layer.moe.debug_force_load_balance
            for layer in config.layers
            if layer.moe is not None
        )


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    def test_npu_converters_cover_all_kimi_rmsnorm_variants(self):
        from torchtitan.models.common.rmsnorm import RMSNorm

        from torchtitan_npu.converters.kernels.kimi_k3_moe import (
            NpuKimiK3MoeConverter,
            NpuKimiRMSNormGated,
        )
        from torchtitan_npu.converters.kernels.rms_norm import (
            NPURMSNorm,
            NpuRMSNormConverter,
        )
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.attention import RMSNormGated
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        model = KimiK3Model(kimi_k3_configs["debug"]())
        regular_norm_names = {
            name
            for name, module in model.named_modules()
            if isinstance(module, RMSNorm)
        }
        gated_norm_names = {
            name
            for name, module in model.named_modules()
            if isinstance(module, RMSNormGated)
        }
        model_spec = SimpleNamespace(name="kimi_k3")

        NpuRMSNormConverter(model_spec).convert(model)
        NpuKimiK3MoeConverter(model_spec).convert(model)

        assert regular_norm_names == {
            name
            for name, module in model.named_modules()
            if isinstance(module, NPURMSNorm)
        }
        assert gated_norm_names == {
            name
            for name, module in model.named_modules()
            if isinstance(module, NpuKimiRMSNormGated)
        }

    def test_debug_model_instantiates(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        assert model is not None
        assert len(model.layers) == 4
        assert model.output_attn_res is not None

    def test_debug_model_parameter_count(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        config = kimi_k3_configs["debug"]()
        model = KimiK3Model(config)
        total_params = sum(p.numel() for p in model.parameters())
        # Debug model should be small (< 100M params)
        assert total_params < 100_000_000
        assert total_params > 0

    def test_init_weights_preserves_kda_parameter_semantics(self):
        from torchtitan_npu.models.kimi_k3 import kimi_k3_configs
        from torchtitan_npu.models.kimi_k3.model import KimiK3Model

        model = KimiK3Model(kimi_k3_configs["debug"]())
        model.init_weights()
        attention = model.layers["0"].attention

        assert torch.all(attention.A_log >= 0)
        assert torch.all(attention.A_log <= torch.log(torch.tensor(16.0)))
        assert torch.count_nonzero(attention.dt_bias) == 0
        assert torch.count_nonzero(
            model.layers["1"].moe.gate.e_score_correction_bias
        ) == 0


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

    def test_debug_router_assigns_experts_round_robin(self):
        from torchtitan_npu.models.kimi_k3.feed_forward import KimiMoEGate

        gate = KimiMoEGate(
            hidden_size=16,
            num_experts=4,
            top_k=2,
            debug_force_load_balance=True,
        )
        expert_indices, _ = gate(torch.randn(1, 4, 16))
        counts = torch.bincount(expert_indices.flatten(), minlength=4)

        assert torch.equal(counts, torch.full((4,), 2))

    def test_reorderer_returns_integer_expert_counts(self):
        from torchtitan_npu.models.kimi_k3.feed_forward import (
            KimiTokenReorderer,
        )

        reorderer = KimiTokenReorderer(num_experts=4, top_k=2)
        scores = torch.ones(4, 2)
        expert_indices = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3]],
        )

        _, _, counts = reorderer(scores, expert_indices)

        assert counts.dtype == torch.int64
        assert torch.equal(counts, torch.full((4,), 2, dtype=torch.int64))


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
            "language_model.model.embed_tokens.weight": torch.randn(100, 64),
            "language_model.model.norm.weight": torch.randn(64),
            "language_model.lm_head.weight": torch.randn(100, 64),
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
            "language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "language_model.model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
            "language_model.model.layers.0.self_attn.q_conv1d.weight": torch.randn(64, 1, 4),
            "language_model.model.layers.0.self_attention_res_proj.weight": torch.randn(1, 64),
            "language_model.model.output_attn_res_norm.weight": torch.randn(64),
            "language_model.model.layers.1.block_sparse_moe.gate.weight": torch.randn(4, 64),
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.randn(32, 64),
            "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight": torch.randn(32, 64),
        }
        result = adapter.from_hf(hf_dict)
        assert "layers.0.attention_norm.weight" in result
        assert "layers.0.attention.o_proj.weight" in result
        assert "layers.0.attention.q_conv1d.conv.weight" in result
        assert "layers.0.self_attention_res.proj.weight" in result
        assert "output_attn_res.norm.weight" in result
        assert "layers.1.moe.gate.gate.weight" in result
        assert result["layers.1.moe.experts.w1"].shape == (2, 32, 64)

    def test_state_dict_adapter_round_trip_uses_official_k3_prefixes(self):
        from torchtitan_npu.models.kimi_k3.state_dict_adapter import (
            KimiK3StateDictAdapter,
        )

        adapter = KimiK3StateDictAdapter(model_config=None)
        hf_dict = {
            "language_model.model.embed_tokens.weight": torch.randn(100, 64),
            "language_model.model.layers.0.self_attn.q_conv1d.weight": torch.randn(64, 1, 4),
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.randn(32, 64),
            "language_model.model.layers.1.block_sparse_moe.experts.1.w1.weight": torch.randn(32, 64),
        }

        assert adapter.to_hf(adapter.from_hf(hf_dict)).keys() == hf_dict.keys()


# ---------------------------------------------------------------------------
# Parallelize
# ---------------------------------------------------------------------------


class TestParallelize:
    def test_selective_ac_uses_moe_save_policy(self):
        from torchtitan_npu.models.kimi_k3.parallelize import (
            parallelize_kimi_k3,
        )

        mock_model = MagicMock()
        mock_parallel_dims = SimpleNamespace(
            pp_enabled=False,
            cp_enabled=False,
            fsdp_enabled=False,
            ep_enabled=False,
            dp_replicate_enabled=False,
            get_optional_mesh=lambda dims: None,
        )
        with patch(
            "torchtitan_npu.models.kimi_k3.parallelize.apply_moe_ac"
        ) as apply_moe_ac:
            parallelize_kimi_k3(
                mock_model,
                parallel_dims=mock_parallel_dims,
                training=SimpleNamespace(seq_len=32),
                model_converters=MagicMock(),
                parallelism=SimpleNamespace(),
                compile_config=SimpleNamespace(enable=False, components=[]),
                ac_config=SimpleNamespace(mode="selective"),
                dump_folder="/tmp/test",
            )

        apply_moe_ac.assert_called_once_with(
            mock_model,
            SimpleNamespace(mode="selective"),
            model_compile_enabled=False,
            base_folder="/tmp/test",
        )

    def test_parallelize_fn_signature(self):
        from torchtitan_npu.models.kimi_k3.parallelize import parallelize_kimi_k3

        mock_model = MagicMock()
        mock_parallel_dims = SimpleNamespace(
            seq_len_divisor=2,
            pp_enabled=False,
            tp_enabled=False,
            cp_enabled=False,
            fsdp_enabled=False,
            ep_enabled=False,
            dp_replicate_enabled=False,
            get_optional_mesh=lambda dims: None,
        )

        mock_ac = SimpleNamespace(mode="none")

        result = parallelize_kimi_k3(
            mock_model,
            parallel_dims=mock_parallel_dims,
            training=SimpleNamespace(seq_len=32),
            model_converters=MagicMock(),
            parallelism=SimpleNamespace(),
            compile_config=SimpleNamespace(enable=False, components=[]),
            ac_config=mock_ac,
            dump_folder="/tmp/test",
        )
        assert result is mock_model
