# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import math
import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torchtitan_npu.models.deepseek_v4 import lora, state_dict_adapter
from tests.unit_tests.models.deepseek_v4.lora_test_utils import N_LAYERS, NUM_EXPERTS, _build_model_config, _adapter_with_converter


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture(scope="module")
def adapter():
    model_config = _build_model_config()
    return state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)


def test_to_hf_omits_local_lora_adapter_parameters(adapter):
    local_state_dict = {
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 4),
        "layers.0.moe.routed_experts.inner_experts.w13_lora_b": torch.ones(2, 4, 2, 2),
        "pre_lora_cache.weight": torch.full((2,), 7.0),
    }

    converted = adapter.to_hf(local_state_dict)
    assert set(converted) == {"pre_lora_cache.weight"}
    torch.testing.assert_close(converted["pre_lora_cache.weight"], local_state_dict["pre_lora_cache.weight"])


def test_to_peft_maps_dense_keys_and_rejects_mtp(adapter):
    local_state_dict = {
        "lm_head.lora_a.weight": torch.ones(2, 32), "lm_head.lora_b.weight": torch.ones(512, 2),
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32),
        "layers.0.attention.wq_a.lora_b.weight": torch.ones(16, 2),
        "pre_lora_cache.weight": torch.full((2,), 7.0),
    }

    peft_state_dict = adapter.to_peft(local_state_dict)

    assert set(peft_state_dict) == {
        "base_model.model.lm_head.lora_A.weight", "base_model.model.lm_head.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight", "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight",
    }

    with pytest.raises(NotImplementedError, match="MTP"):
        adapter.to_peft({"mtp_layers.0.e_proj.lora_a.weight": torch.ones(2, 32)})


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_peft_mapping_shapes_and_safetensors_round_trip(adapter, tmp_path, rank, dtype):
    shapes = {
        "w13_lora_a": (NUM_EXPERTS, rank, 32), "w13_lora_b": (NUM_EXPERTS, 16, 2, rank),
        "w2_lora_a": (NUM_EXPERTS, rank, 16), "w2_lora_b": (NUM_EXPERTS, 32, rank),
    }
    tensors = {name: torch.arange(math.prod(shape), dtype=dtype).reshape(shape) for name, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.moe.routed_experts.inner_experts.{name}": t for name, t in tensors.items()})
    prefix = "base_model.model.model.layers.0.mlp.experts"
    assert set(exported) == {
        f"{prefix}{layer}.lora_{factor}.weight" for layer in ("", ".base_layer") for factor in ("A", "B")
    }
    for layer, stem, width in ((".base_layer", "w13", 32), ("", "w2", 16)):
        a = exported[f"{prefix}{layer}.lora_A.weight"].reshape(NUM_EXPERTS, rank, width)
        b = exported[f"{prefix}{layer}.lora_B.weight"].reshape(32, rank, NUM_EXPERTS)
        for expert in range(NUM_EXPERTS):
            torch.testing.assert_close(a[expert], tensors[f"{stem}_lora_a"][expert], rtol=0, atol=0)
            expected_b = tensors[f"{stem}_lora_b"][expert]
            if stem == "w13":
                expected_b = torch.cat((expected_b[:, 0], expected_b[:, 1]))
            torch.testing.assert_close(b[:, :, expert], expected_b, rtol=0, atol=0)
    writer = dcp.HuggingFaceStorageWriter(path=str(tmp_path), save_distributed=True, enable_consolidation=True)
    dcp.save(exported, storage_writer=writer)
    files = list(tmp_path.glob("model-*.safetensors"))
    assert len(files) == 1
    loaded = load_file(str(files[0]))
    assert loaded.keys() == exported.keys()
    for name, value in exported.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("base_path", [None, "/data/base"], ids=["assets-fallback", "explicit-base"])
def test_peft_config_metadata_targets_and_base_path(base_path):
    adapter = state_dict_adapter.DeepSeekV4StateDictAdapter(_build_model_config(), hf_assets_path="/data/assets")

    config = adapter.peft_adapter_config(base_model_name_or_path=base_path)

    assert config["base_model_name_or_path"] == (base_path or "/data/assets")
    assert (config["r"], config["lora_alpha"]) == (64, 128.0)
    assert config["peft_type"] == "LORA"
    assert config["task_type"] == "CAUSAL_LM"
    assert config["target_modules"] == []
    assert {"self_attn.q_a_proj.weight", "self_attn.o_a_proj.weight", "mlp.shared_experts.gate_proj.weight",
            "lm_head.weight"} <= set(config["target_parameters"])
    indexer_modules = (
        "model.layers.0.attn.indexer.wq_b", "model.layers.0.attn.indexer.weights_proj",
        "model.layers.0.attn.indexer.compressor.wkv", "model.layers.0.attn.indexer.compressor.wgate",
    )
    for module_name in indexer_modules:
        matched = [
            target for target in config["target_parameters"] if module_name == target or module_name.endswith("." + target)
        ]
        assert not matched, f"{module_name} would receive an adapter via {matched}"


@pytest.mark.parametrize("adapt_experts", [False, True], ids=["dense", "dense-and-experts"])
def test_peft_config_targets_match_converter(adapt_experts):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        rank=4, alpha=8.0, rank_experts=4, target_modules=["attention.wo_b"], adapt_routed_experts=adapt_experts,
        include_mtp=False,
    )

    config = _adapter_with_converter(converter).peft_adapter_config()
    assert (config["r"], config["lora_alpha"]) == (4, 8.0)

    assert config["target_parameters"] == [f"model.layers.{i}.self_attn.o_b_proj.weight" for i in range(N_LAYERS)] + (
        [f"model.layers.{i}.mlp.experts.{name}" for i in range(N_LAYERS) for name in ("gate_up_proj", "down_proj")]
        if adapt_experts else []
    )


def test_peft_adapter_config_rejects_mismatched_expert_rank():
    converter_config = lora.DeepSeekV4LoRAConverter.Config(
        rank=64, rank_experts=32, target_modules=["attention.wo_b"], adapt_routed_experts=True, include_mtp=False
    )
    adapter = _adapter_with_converter(converter_config)

    with pytest.raises(ValueError, match="rank_experts"):
        adapter.peft_adapter_config()


def test_peft_export_loads_in_transformers_and_matches_merged_logits(tmp_path):
    import copy

    peft_lib = pytest.importorskip("peft", minversion="0.20.0")
    transformers = pytest.importorskip("transformers", minversion="5.17.0")
    from safetensors.torch import save_file

    config = transformers.DeepseekV4Config(
        vocab_size=64, hidden_size=32, moe_intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=4, head_dim=8, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        n_routed_experts=2, num_experts_per_tok=1, hc_mult=2, hc_sinkhorn_iters=2,
        layer_types=["sliding_attention"], mlp_layer_types=["moe"],
        partial_rotary_factor=0.5, max_position_embeddings=64,
    )
    base = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    expected = copy.deepcopy(base)
    model_config = lora.DeepSeekV4LoRAConverter.Config(
        rank=2, rank_experts=2, alpha=4.0, include_mtp=False,
        target_modules=["attention.wq_a", "attention.wo_a"],
    ).build().convert(_build_model_config(num_experts=2, num_layers=1))
    adapter = state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    shapes = {
        "attention.wq_a.lora_a.weight": (2, 32), "attention.wq_a.lora_b.weight": (16, 2),
        "attention.wo_a.lora_a.weight": (2, 16), "attention.wo_a.lora_b.weight": (16, 2),
        "moe.routed_experts.inner_experts.w13_lora_a": (2, 2, 32),
        "moe.routed_experts.inner_experts.w13_lora_b": (2, 16, 2, 2),
        "moe.routed_experts.inner_experts.w2_lora_a": (2, 2, 16),
        "moe.routed_experts.inner_experts.w2_lora_b": (2, 32, 2),
    }
    tensors = {key: torch.randn(shape) * 0.1 for key, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.{key}": value for key, value in tensors.items()})
    save_file({key: value.clone().contiguous() for key, value in exported.items()},
              str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter.peft_adapter_config()))

    block = expected.model.layers[0]
    with torch.no_grad():
        for local, hf in (("wq_a", "q_a_proj"), ("wo_a", "o_a_proj")):
            a = tensors[f"attention.{local}.lora_a.weight"]
            b = tensors[f"attention.{local}.lora_b.weight"]
            getattr(block.self_attn, hf).weight.add_(b @ a, alpha=2.0)
        routed = "moe.routed_experts.inner_experts."
        for expert in range(2):
            a = tensors[routed + "w13_lora_a"][expert]
            b = tensors[routed + "w13_lora_b"][expert]
            block.mlp.experts.gate_up_proj[expert, :16].add_(b[:, 0, :] @ a, alpha=2.0)
            block.mlp.experts.gate_up_proj[expert, 16:].add_(b[:, 1, :] @ a, alpha=2.0)
            block.mlp.experts.down_proj[expert].add_(
                tensors[routed + "w2_lora_b"][expert] @ tensors[routed + "w2_lora_a"][expert], alpha=2.0,
            )
        inputs = torch.tensor([[1, 2, 3, 4]])
        base_logits = base(inputs, use_cache=False).logits
        expected_logits = expected(inputs, use_cache=False).logits
    loaded = peft_lib.PeftModel.from_pretrained(base, tmp_path).eval()
    loaded_state = peft_lib.get_peft_model_state_dict(loaded)
    assert loaded_state.keys() == exported.keys()
    for key, value in exported.items():
        torch.testing.assert_close(loaded_state[key], value, rtol=0, atol=0)
    with torch.no_grad():
        actual_logits = loaded(inputs, use_cache=False).logits
    assert not torch.equal(base_logits, expected_logits)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("target", ["wq_a", "attention.wq_a", "layers.0.attention.wq_a"])
def test_peft_target_alias_matches_exported_parameter(target):
    adapter = _adapter_with_converter(lora.DeepSeekV4LoRAConverter.Config(
        target_modules=[target], adapt_routed_experts=False, include_mtp=False,
    ))
    metadata = adapter.peft_adapter_config()
    layers = [0] if target.startswith("layers.") else range(N_LAYERS)
    assert metadata["target_parameters"] == [f"model.layers.{i}.self_attn.q_a_proj.weight" for i in layers]
    exported = adapter.to_peft({"layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32)})
    assert set(exported) == {"base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"}


def test_peft_rejects_partially_unmapped_adapters(adapter):
    tensors = {
        "layers.0.attention.wq_a.lora_a.weight": torch.ones(2, 32),
        "layers.0.attention.indexer.wq_b.lora_a.weight": torch.ones(2, 32),
    }
    with pytest.raises(ValueError, match="Unsupported.*attention.indexer.wq_b"):
        adapter.to_peft(tensors)


@pytest.mark.parametrize("rank", [1, 3])
def test_split_expert_b_exports_same_peft_weights_as_interleaved(adapter, rank):
    prefix = "layers.0.moe.routed_experts.inner_experts."
    gate = torch.arange(NUM_EXPERTS * 16 * rank).reshape(NUM_EXPERTS, 16, rank).float()
    up = -gate - 1
    legacy = {prefix + "w13_lora_b": torch.stack((gate, up), dim=2)}
    split = {prefix + "w1_lora_b": gate, prefix + "w3_lora_b": up}

    exported = adapter.to_peft(split)
    expected = adapter.to_peft(legacy)

    assert exported.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(exported[key], expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="both gate and up"):
        adapter.to_peft({prefix + "w1_lora_b": gate})
