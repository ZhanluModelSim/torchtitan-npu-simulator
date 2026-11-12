# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json

import torch
from safetensors.torch import save_file

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _lora_model_config
from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter


def _adapter_and_factors():
    config = _lora_model_config(targets=["wq_a", "wo_a", "w13", "w2"])
    with torch.device("meta"):
        model = config.build()
    generator = torch.Generator().manual_seed(19)
    factors = {
        name: torch.randn(param.shape, generator=generator) * 0.03
        for name, param in model.named_parameters()
        if is_lora_parameter(name)
    }
    return DeepSeekV4StateDictAdapter(config, None), factors


def test_standard_peft_factor_layout_preserves_each_expert_and_gate_up_order():
    adapter, factors = _adapter_and_factors()
    exported = adapter.to_peft(factors)
    prefix = "base_model.model.model.layers.0.mlp.experts"
    for projection, suffix, output in (("w13", ".base_layer", 32), ("w2", "", 32)):
        a = exported[f"{prefix}{suffix}.lora_A.weight"].reshape(2, 8, -1)
        b = exported[f"{prefix}{suffix}.lora_B.weight"].reshape(output, 8, 2)
        for expert in range(2):
            torch.testing.assert_close(
                a[expert], factors[f"layers.0.feed_forward.routed_experts.{projection}.lora_a.weight"][expert]
            )
            original_b = factors[f"layers.0.feed_forward.routed_experts.{projection}.lora_b.weight"][expert]
            expected_b = torch.cat((original_b[0], original_b[1])) if projection == "w13" else original_b
            torch.testing.assert_close(b[:, :, expert], expected_b)
    metadata = adapter.peft_adapter_config()
    assert metadata["target_modules"] == []
    assert metadata["target_parameters"][-2:] == [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]
    assert metadata["r"] == 8 and metadata["lora_alpha"] == 16
    assert "model.layers.0.self_attn.o_a_proj.weight" in metadata["target_parameters"]


def test_hf_base_export_omits_standard_adapter_tensors():
    adapter, factors = _adapter_and_factors()
    state = {"layers.0.attention.wq_a.weight": torch.arange(512).reshape(16, 32).float(), **factors}
    exported = adapter.to_hf(state)
    assert exported.keys() == {"layers.0.attn.wq_a.weight"}
    torch.testing.assert_close(exported["layers.0.attn.wq_a.weight"], state["layers.0.attention.wq_a.weight"])


def test_fused_base_expert_and_shared_weights_roundtrip_exactly():
    adapter, _ = _adapter_and_factors()
    state = {
        "layers.0.feed_forward.routed_experts.w13.weight": torch.arange(2048).reshape(2, 2, 16, 32).float(),
        "layers.0.feed_forward.routed_experts.w2.weight": torch.arange(1024).reshape(2, 32, 16).float(),
        "layers.0.feed_forward.shared_experts.w13.weight": torch.arange(1024).reshape(2, 16, 32).float(),
        "layers.0.feed_forward.shared_experts.w2.weight": torch.arange(512).reshape(32, 16).float(),
    }
    exported = adapter.to_hf(state)
    assert "layers.0.ffn.experts.1.w3.weight" in exported
    assert "layers.0.ffn.shared_experts.w1.weight" in exported
    restored = adapter.from_hf(exported)
    assert restored.keys() == state.keys()
    for key, value in state.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


def test_peft_export_loads_in_transformers_and_matches_independent_merged_logits(tmp_path):
    import peft
    import transformers

    config = transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        partial_rotary_factor=0.5,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        base = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    expected = copy.deepcopy(base)
    adapter, factors = _adapter_and_factors()
    exported = adapter.to_peft(factors)
    save_file({key: value.clone() for key, value in exported.items()}, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter.peft_adapter_config()))
    layer = expected.model.layers[0]
    with torch.no_grad():
        for projection, hf in (("wq_a", "q_a_proj"), ("wo_a", "o_a_proj")):
            a = factors[f"layers.0.attention.{projection}.lora_a.weight"]
            b = factors[f"layers.0.attention.{projection}.lora_b.weight"]
            getattr(layer.self_attn, hf).weight.add_(b @ a, alpha=2)
        prefix = "layers.0.feed_forward.shared_experts"
        for index, projection in enumerate(("gate_proj", "up_proj")):
            getattr(layer.mlp.shared_experts, projection).weight.add_(
                factors[f"{prefix}.w13.lora_b.weight"][index] @ factors[f"{prefix}.w13.lora_a.weight"],
                alpha=2,
            )
        layer.mlp.shared_experts.down_proj.weight.add_(
            factors[f"{prefix}.w2.lora_b.weight"] @ factors[f"{prefix}.w2.lora_a.weight"],
            alpha=2,
        )
        prefix = "layers.0.feed_forward.routed_experts"
        for expert in range(2):
            a = factors[f"{prefix}.w13.lora_a.weight"][expert]
            b = factors[f"{prefix}.w13.lora_b.weight"][expert]
            layer.mlp.experts.gate_up_proj[expert, :16].add_(b[0] @ a, alpha=2)
            layer.mlp.experts.gate_up_proj[expert, 16:].add_(b[1] @ a, alpha=2)
            layer.mlp.experts.down_proj[expert].add_(
                factors[f"{prefix}.w2.lora_b.weight"][expert] @ factors[f"{prefix}.w2.lora_a.weight"][expert],
                alpha=2,
            )
        inputs = torch.tensor([[1, 2, 3, 4]])
        expected_logits = expected(inputs, use_cache=False).logits
        base_logits = base(inputs, use_cache=False).logits
    loaded = peft.PeftModel.from_pretrained(base, tmp_path).eval()
    loaded_state = peft.get_peft_model_state_dict(loaded)
    assert loaded_state.keys() == exported.keys()
    for key, value in exported.items():
        torch.testing.assert_close(loaded_state[key], value, rtol=0, atol=0)
    with torch.no_grad():
        actual_logits = loaded(inputs, use_cache=False).logits
    assert not torch.equal(base_logits, expected_logits)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
