# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qwen3-Next: config registry, model structure, parameter
count formula, parallel fail-fast boundaries, and MXFP8 configuration."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


class TestConfigRegistry:
    @pytest.mark.parametrize(
        "name",
        (
            "qwen3_next_smoketest",
            "qwen3_next_80b_a3b",
            "qwen3_next_80b_a3b_2layer",
            "qwen3_next_2l_tp2",
            "qwen3_next_2l_ep2",
            "qwen3_next_2l_fsdp2",
            "qwen3_next_2l_tp2ep2",
            "qwen3_next_2l_fsdp2ep2",
            "qwen3_next_2l_tp2ep2fsdp2",
            "qwen3_next_2l_mxfp8",
        ),
    )
    def test_simulator_configs_exist(self, name):
        from torchtitan_npu.simulator import config_registry

        assert hasattr(config_registry, name)

    def test_invalid_flavor_raises(self):
        from torchtitan_npu.models.qwen3_next import model_registry

        with pytest.raises(KeyError):
            model_registry("no-such-flavor")

    def test_smoketest_runs_ep2_measured_alignment(self):
        from torchtitan_npu.models.qwen3_next import config_registry

        config = config_registry.qwen3_next_smoketest()
        assert config.parallelism.expert_parallel_degree == 2
        assert config.parallelism.tensor_parallel_degree == 1

    @pytest.mark.parametrize(
        ("config_name", "tp", "ep"),
        (
            ("qwen3_next_2l_tp2", 2, 1),
            ("qwen3_next_2l_ep2", 1, 2),
            ("qwen3_next_2l_tp2ep2", 2, 2),
            ("qwen3_next_2l_fsdp2ep2", 1, 2),
        ),
    )
    def test_parallel_variant_degrees(self, config_name, tp, ep):
        from torchtitan_npu.models.qwen3_next import config_registry

        config = getattr(config_registry, config_name)()
        assert config.parallelism.tensor_parallel_degree == tp
        assert config.parallelism.expert_parallel_degree == ep


# ---------------------------------------------------------------------------
# MXFP8 quantization configuration
# ---------------------------------------------------------------------------


class TestMxfp8Config:
    def test_mxfp8_variant_enables_quant_converter(self):
        from torchtitan.components.quantization.mx import MXFP8Converter

        from torchtitan_npu.models.qwen3_next import config_registry

        config = config_registry.qwen3_next_2l_mxfp8()
        mxfp8 = [
            c
            for c in config.model_converters.converters
            if isinstance(c, MXFP8Converter.Config)
        ]
        assert len(mxfp8) == 1
        assert mxfp8[0].recipe_name == "mxfp8_rceil"
        assert mxfp8[0].fqns  # precise FQN control

    def test_bf16_variant_has_no_quant_converter(self):
        from torchtitan.components.quantization.mx import MXFP8Converter

        from torchtitan_npu.models.qwen3_next import config_registry

        config = config_registry.qwen3_next_2l_ep2()
        assert not [
            c
            for c in config.model_converters.converters
            if isinstance(c, MXFP8Converter.Config)
        ]

    def test_router_gates_never_match_mxfp8_fqns(self):
        # MXFP8Converter matches FQNs by substring containment; the
        # configured FQNs must not capture any router gate module.
        from torchtitan_npu.models.qwen3_next.config_registry import (
            _MXFP8_FQNS,
            _ROUTER_FQNS,
        )

        for fqn in _MXFP8_FQNS:
            for router in _ROUTER_FQNS:
                assert fqn not in router, (fqn, router)

    def test_mxfp8_fqns_override_rejects_router_capture(self):
        from torchtitan_npu.models.qwen3_next import config_registry

        config = config_registry.qwen3_next_2l_mxfp8()
        with pytest.raises(ValueError, match="router"):
            config.mxfp8_fqns = ["moe"]
            config.__post_init__()

    def test_mxfp8_fqns_override_requires_mxfp8_recipe(self):
        from torchtitan_npu.models.qwen3_next import config_registry

        config = config_registry.qwen3_next_2l_ep2()
        with pytest.raises(ValueError, match="MXFP8"):
            config.mxfp8_fqns = ["token_mixer"]
            config.__post_init__()

    def test_routed_experts_excluded_from_mxfp8(self):
        # The routed experts keep raw parameters (torch._grouped_mm path,
        # outside the torchao nn.Linear wrapping): their FQN is excluded
        # from the quantization scope (declared in the model contract).
        from torchtitan_npu.models.qwen3_next.config_registry import _MXFP8_FQNS

        assert not any("experts" in fqn.split(".")[-1] for fqn in _MXFP8_FQNS)
        assert not any(fqn == "moe.experts" for fqn in _MXFP8_FQNS)


# ---------------------------------------------------------------------------
# Model structure and parameter count
# ---------------------------------------------------------------------------


def _meta_model(flavor="debugmodel"):
    from torchtitan_npu.models.qwen3_next import qwen3_next_configs
    from torchtitan_npu.models.qwen3_next.model import Qwen3NextModel

    with torch.device("meta"):
        return Qwen3NextModel(qwen3_next_configs[flavor]())


class TestModelStructure:
    def test_debug_layer_layout_hybrid_gdn_attention(self):
        model = _meta_model()
        assert len(model.layers) == 4
        kinds = [
            "gdn" if layer.is_linear_attention else "attn"
            for layer in model.layers.values()
        ]
        # Every 4th layer (i % 4 == 3) is gated attention.
        assert kinds == ["gdn", "gdn", "gdn", "attn"]
        assert all(layer.moe_enabled for layer in model.layers.values())

    def test_2layer_flavor_layout(self):
        from torchtitan_npu.models.qwen3_next import qwen3_next_configs

        cfg = qwen3_next_configs["80B-A3B-2L"]()
        assert [c.layer_id for c in cfg.layers] == [0, 1]
        # GatedAttention.Config has n_heads; GatedDeltaNet.Config has
        # num_v_heads (the nested Config classes share the class name).
        assert hasattr(cfg.layers[0].token_mixer, "n_heads")
        assert hasattr(cfg.layers[1].token_mixer, "num_v_heads")
        assert all(c.moe is not None for c in cfg.layers)

    def test_2layer_param_count_matches_independent_formula(self):
        """Independent parameter-count formula for the measured 2-layer
        flavor (see MODEL_CONTRACT.md section 2)."""
        from torchtitan_npu.models.qwen3_next import qwen3_next_configs
        from torchtitan_npu.models.qwen3_next.model import Qwen3NextModel

        dim, vocab = 2048, 151936
        head_dim, n_heads, n_kv_heads = 256, 16, 2
        num_k_heads, num_v_heads, head_k, head_v = 16, 32, 128, 128
        experts, moe_inter = 512, 512

        key_dim, value_dim = num_k_heads * head_k, num_v_heads * head_v
        embed = vocab * dim
        output = dim * vocab  # lm_head weight (no bias)
        final_norm = dim
        per_layer_norm = 2 * dim
        attn = (
            dim * (n_heads * head_dim * 2)  # fused q|gate
            + 2 * dim * (n_kv_heads * head_dim)  # k, v
            + (n_heads * head_dim) * dim  # o
            + 2 * head_dim  # q/k norms
        )
        gdn = (
            dim * (2 * key_dim + 2 * value_dim)  # in_proj_qkvz
            + dim * (2 * num_v_heads)  # in_proj_ba
            + (2 * key_dim + value_dim) * 4  # depthwise conv1d kernel
            + num_v_heads  # dt_bias
            + num_v_heads  # A_log
            + head_v  # gated RMSNorm weight
            + value_dim * dim  # out_proj
        )
        moe = (
            dim * experts  # router gate
            + experts * (dim * 2 * moe_inter + moe_inter * dim)  # routed
            + 3 * dim * moe_inter  # shared expert (gate/up/down)
            + dim  # shared expert gate
        )
        expected = (
            embed
            + output
            + final_norm
            + (attn + per_layer_norm + moe)
            + (gdn + per_layer_norm + moe)
        )

        with torch.device("meta"):
            model = Qwen3NextModel(qwen3_next_configs["80B-A3B-2L"]())
        actual = sum(p.numel() for p in model.parameters())
        assert actual == expected, (actual, expected)


# ---------------------------------------------------------------------------
# Parallel fail-fast boundaries
# ---------------------------------------------------------------------------


def _fake_parallel_dims(**kwargs):
    defaults = dict(
        tp_enabled=False,
        ep_enabled=False,
        etp_enabled=False,
        cp_enabled=False,
        pp_enabled=False,
        fsdp_enabled=True,
        dp_replicate_enabled=False,
    )
    defaults.update(kwargs)
    return dataclasses.make_dataclass(
        "FakeParallelDims", [(k, bool) for k in defaults]
    )(**defaults)


def _fail_fast_kwargs():
    return dict(
        training=SimpleNamespace(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
        ),
        model_converters=None,
        parallelism=None,
        compile_config=SimpleNamespace(enable=False),
        ac_config=SimpleNamespace(mode="none"),
        dump_folder="/tmp",
    )


class TestParallelFailFast:
    def test_etp_fails_fast(self):
        from torchtitan_npu.models.qwen3_next.parallelize import (
            parallelize_qwen3_next,
        )

        model = _meta_model()
        dims = _fake_parallel_dims(etp_enabled=True, fsdp_enabled=False)
        with pytest.raises(NotImplementedError, match="[Ee]xpert [Tt]ensor"):
            parallelize_qwen3_next(model, parallel_dims=dims, **_fail_fast_kwargs())

    def test_cp_fails_fast(self):
        from torchtitan_npu.models.qwen3_next.parallelize import (
            parallelize_qwen3_next,
        )

        model = _meta_model()
        dims = _fake_parallel_dims(cp_enabled=True, fsdp_enabled=False)
        with pytest.raises(NotImplementedError, match="context parallel"):
            parallelize_qwen3_next(model, parallel_dims=dims, **_fail_fast_kwargs())

    def test_pp_fails_fast(self):
        from torchtitan_npu.models.qwen3_next.parallelize import (
            parallelize_qwen3_next,
        )

        model = _meta_model()
        dims = _fake_parallel_dims(pp_enabled=True, fsdp_enabled=False)
        with pytest.raises(NotImplementedError, match="pipeline"):
            parallelize_qwen3_next(model, parallel_dims=dims, **_fail_fast_kwargs())

    def test_tp_head_divisibility_guard(self):
        from torchtitan_npu.models.qwen3_next.parallelize import (
            _check_tp_divisibility,
        )

        model = _meta_model("80B-A3B-2L")
        # n_kv_heads=2 / num_k_heads=16 reject tp=4 on the 2L flavor.
        with pytest.raises(ValueError, match="divisible"):
            _check_tp_divisibility(model, 4)
        _check_tp_divisibility(model, 2)  # passes
