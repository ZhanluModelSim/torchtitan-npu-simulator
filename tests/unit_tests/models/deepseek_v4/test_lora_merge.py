# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.lora.merge_adapter import merge
from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config
from torchtitan_npu.models.deepseek_v4.lora import (
    OFFICIAL_MODULES,
    DeepSeekV4LoRAConverter,
    build_lora_merge_plan,
    merge_lora_weight,
)
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter


@pytest.fixture
def exported_adapter(tmp_path):
    model_config = (
        DeepSeekV4LoRAConverter.Config(
            rank=2,
            rank_experts=2,
            alpha=4.0,
            include_mtp=False,
            target_modules=["attention.wq_a", "attention.wq_b"],
        )
        .build()
        .convert(_build_model_config(num_experts=2, num_layers=1))
    )
    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    with torch.device("meta"):
        model = model_config.build()
    shapes = {
        name.removeprefix("layers.0."): parameter.shape
        for name, parameter in model.named_parameters()
        if name.startswith("layers.0.") and "lora_" in name
    }
    generator = torch.Generator().manual_seed(31)
    factors = {name: torch.randn(shape, generator=generator) * 0.1 for name, shape in shapes.items()}
    exported = adapter.to_peft({f"layers.0.{name}": value for name, value in factors.items()})
    directory = tmp_path / "adapter"
    _write_adapter(directory, exported, config=adapter.peft_adapter_config())

    return model, factors, adapter, directory


def _checkpoint_weights(adapter, weights, layout):
    result = adapter.to_hf(weights)
    if layout == "official":
        return result
    renamed = {}
    for name, value in result.items():
        if ".ffn.experts." in name:
            name = name.replace(".ffn.experts.", ".mlp.experts.")
        else:
            for hf, official in OFFICIAL_MODULES.items():
                name = name.replace(f".{official}.weight", f".{hf}.weight")
        renamed["model." + name] = value
    return renamed


@pytest.mark.parametrize("layout", ["official", "transformers"])
def test_current_export_merges_checkpoint_weights_and_reloads_state(exported_adapter, tmp_path, layout):
    model, factors, adapter, adapter_dir = exported_adapter
    generator = torch.Generator().manual_seed(13)
    base = {}
    for name, parameter in model.named_parameters():
        if not name.startswith("layers.0.") or "lora_" in name:
            continue
        if name.endswith(("wq_a.weight", "wq_b.weight", "w1_EFD", "w2_EDF", "w3_EFD")):
            base[name] = torch.randn(parameter.shape, generator=generator)
    expected = {name: value.clone() for name, value in base.items()}
    for local in ("wq_a", "wq_b"):
        a, b = (factors[f"attention.{local}.lora_{factor}.weight"] for factor in ("a", "b"))
        expected[f"layers.0.attention.{local}.weight"].add_(b @ a, alpha=2.0)
    prefix = "moe.routed_experts.inner_experts."
    for weight, a_name, b_name in (("w1_EFD", "w13", "w1"), ("w3_EFD", "w13", "w3"), ("w2_EDF", "w2", "w2")):
        a = factors[prefix + a_name + "_lora_a"]
        b = factors[prefix + b_name + "_lora_b"]
        expected["layers.0." + prefix + weight].add_(b @ a, alpha=2.0)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    official = _checkpoint_weights(adapter, base, layout)
    save_file({name: value.clone().contiguous() for name, value in official.items()}, base_dir / "model.safetensors")
    output = tmp_path / "merged"

    report = merge(str(base_dir), str(adapter_dir), str(output))
    written = load_file(output / "model.safetensors")
    expected_official = _checkpoint_weights(adapter, expected, layout)

    assert report["merged_count"] == 8
    assert written.keys() == expected_official.keys()
    for name, value in expected_official.items():
        torch.testing.assert_close(written[name], value, rtol=1e-6, atol=1e-7)
    if layout == "official":
        restored = adapter.from_hf(written)
        for name, value in expected.items():
            torch.testing.assert_close(restored[name].reshape_as(value), value, rtol=1e-6, atol=1e-7)
    for name, value in load_file(base_dir / "model.safetensors").items():
        torch.testing.assert_close(value, official[name], rtol=0, atol=0)


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_fused_expert_merge_matches_independent_forward(tmp_path, projection):
    from torchtitan_npu.models.deepseek_v4.lora import pack_expert_factor

    base_dir, adapter_dir, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base_dir.mkdir()
    gate_up = projection == "gate_up_proj"
    base = torch.randn(3, 8 if gate_up else 4, 4)
    a = torch.randn(3, 2, 4)
    b = torch.randn(3, 4, 2, 2) if gate_up else torch.randn(3, 4, 2)
    key = f"model.layers.0.mlp.experts.{projection}"
    module = "base_model.model.model.layers.0.mlp.experts" + (".base_layer" if gate_up else "")
    save_file({key: base}, base_dir / "model.safetensors")
    _write_adapter(
        adapter_dir,
        {
            module + ".lora_A.weight": pack_expert_factor(a, "w13_lora_a" if gate_up else "w2_lora_a"),
            module + ".lora_B.weight": pack_expert_factor(b, "w13_lora_b" if gate_up else "w2_lora_b"),
        },
    )
    merge(str(base_dir), str(adapter_dir), str(output))
    written = load_file(output / "model.safetensors")[key]
    for expert in range(3):
        factor_b = torch.cat((b[expert, :, 0], b[expert, :, 1])) if gate_up else b[expert]
        expected = base[expert] + 2 * (factor_b @ a[expert])
        torch.testing.assert_close(written[expert], expected)
        inputs = torch.randn(5, 4)
        reference = inputs @ base[expert].T + 2 * ((inputs @ a[expert].T) @ factor_b.T)
        torch.testing.assert_close(inputs @ written[expert].T, reference, rtol=1e-5, atol=1e-5)


def _write_adapter(directory, tensors, *, rank=2, alpha=4, config=None):
    directory.mkdir()
    save_file({name: value.contiguous() for name, value in tensors.items()}, directory / "adapter_model.safetensors")
    (directory / "adapter_config.json").write_text(
        json.dumps(
            config
            or {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": alpha,
            }
        )
    )


@pytest.mark.parametrize("kind", ["empty", "orphan_a", "orphan_b", "unknown"])
def test_merge_plan_rejects_incomplete_or_unmatched_adapter(kind):
    tensors = {
        "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
        "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
    }
    if kind == "empty":
        tensors.clear()
    elif kind != "unknown":
        tensors.pop(f"base_model.model.proj.lora_{'B' if kind == 'orphan_a' else 'A'}.weight")
    with pytest.raises(ValueError, match=r"Adapter|adapter"):
        build_lora_merge_plan(tensors, {} if kind == "unknown" else {"proj.weight": "model.safetensors"}, rank=2)


def test_merge_tensor_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        merge_lora_weight(torch.zeros(8, 6), torch.ones(2, 6), torch.ones(4, 2), scaling=1.0)


@pytest.mark.parametrize("location", ["base", "adapter", "existing"])
def test_merge_protects_inputs_and_existing_output(tmp_path, location):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    output = tmp_path / location / "merged"
    if location == "existing":
        output.mkdir(parents=True)
    with pytest.raises(FileExistsError if location == "existing" else ValueError):
        merge(str(base), str(adapter), str(output))


def test_merge_rejects_indexed_weight_missing_from_shard(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    save_file({"other.weight": torch.zeros(4, 4)}, base / "shard.safetensors")
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"proj.weight": "shard.safetensors"}}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    with pytest.raises(ValueError, match="missing tensors"):
        merge(str(base), str(adapter), str(tmp_path / "output"))


@pytest.mark.parametrize("declared", [False, True])
def test_quantized_input_requires_explicit_conversion(tmp_path, declared):
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({"proj.weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn)}, base / "model.safetensors")
    if declared:
        (base / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    with pytest.raises(ValueError, match=r"checkpoint|Decode"):
        merge(str(base), str(adapter), str(output))
    assert not output.exists()


def test_merge_preserves_untargeted_shards_and_metadata(tmp_path):
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({"proj.weight": torch.ones(4, 4)}, base / "weights.safetensors")
    save_file({"other.scale": torch.ones(1)}, base / "untouched.safetensors")
    index = {"weight_map": {"proj.weight": "weights.safetensors", "other.scale": "untouched.safetensors"}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    (base / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    merge(str(base), str(adapter), str(output))
    torch.testing.assert_close(load_file(output / "weights.safetensors")["proj.weight"], torch.full((4, 4), 5.0))
    for name in ("config.json", "model.safetensors.index.json", "untouched.safetensors"):
        assert (output / name).read_bytes() == (base / name).read_bytes()


@pytest.mark.parametrize("kind", ["dense_rank", "fewer_experts", "missing_expert", "missing_projection"])
def test_merge_rejects_semantic_mismatch_before_writing(tmp_path, kind):
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    if kind == "dense_rank":
        weights = {"proj.weight": torch.zeros(4, 4)}
        module, a, b = "proj", torch.ones(3, 4), torch.ones(4, 3)
    else:
        weights = {
            f"layers.0.ffn.experts.{i}.{proj}.weight": torch.zeros(4, 4)
            for i in range(3)
            for proj in ("w1", "w2", "w3")
        }
        module = "model.layers.0.mlp.experts.base_layer"
        a, b = torch.ones(4 if kind == "fewer_experts" else 6, 4), torch.ones(8, 4 if kind == "fewer_experts" else 6)
        if kind == "missing_expert":
            weights = {key: value for key, value in weights.items() if ".1." not in key}
        elif kind == "missing_projection":
            weights.pop("layers.0.ffn.experts.2.w3.weight")
    save_file(weights, base / "model.safetensors")
    _write_adapter(
        adapter, {f"base_model.model.{module}.lora_A.weight": a, f"base_model.model.{module}.lora_B.weight": b}
    )
    with pytest.raises(ValueError, match="rank|expert"):
        merge(str(base), str(adapter), str(output))
    assert not output.exists()
    assert not list(tmp_path.glob(".lora-merge-*"))


@pytest.mark.parametrize("failure", ["copy", "save"])
def test_merge_cleans_failed_output_and_can_retry(tmp_path, monkeypatch, failure):
    from scripts.lora import merge_adapter

    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({"proj.weight": torch.zeros(4, 4)}, base / "model.safetensors")
    (base / "config.json").write_text("{}")
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            merge_adapter.shutil if failure == "copy" else merge_adapter,
            "copy2" if failure == "copy" else "save_file",
            fail,
        )
        with pytest.raises(OSError, match="injected"):
            merge(str(base), str(adapter), str(output))
    assert not output.exists()
    assert not list(tmp_path.glob(".lora-merge-*"))
    merge(str(base), str(adapter), str(output))
    torch.testing.assert_close(load_file(output / "model.safetensors")["proj.weight"], torch.full((4, 4), 4.0))


def test_merge_rejects_serialized_torchao_weight(tmp_path):
    pytest.importorskip("torchao_npu")
    from torchao_npu.serialization.export import save_hf_safetensors
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    quantized = Float8Tensor(
        torch.ones(4, 4).to(torch.float8_e4m3fn), torch.ones(1), block_size=[4, 4], dtype=torch.float32
    )
    save_hf_safetensors({"proj.weight": quantized}, base)
    _write_adapter(
        adapter,
        {
            "base_model.model.proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.proj.lora_B.weight": torch.ones(4, 2),
        },
    )
    with pytest.raises(ValueError, match="Decode.*FP16/BF16/FP32"):
        merge(str(base), str(adapter), str(output))
    assert not output.exists()


def test_merge_loads_only_one_adapter_pair_at_a_time(tmp_path, monkeypatch):
    import weakref
    from scripts.lora import merge_adapter

    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "output"))
    base.mkdir()
    save_file({f"proj{i}.weight": torch.zeros(4, 4) for i in range(3)}, base / "model.safetensors")
    _write_adapter(
        adapter,
        {
            f"base_model.model.proj{i}.lora_{factor}.weight": torch.ones(shape)
            for i in range(3)
            for factor, shape in (("A", (2, 4)), ("B", (4, 2)))
        },
    )
    original_open, loaded = merge_adapter.safe_open, []

    class TrackedAdapter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            return self.handle.get_slice(key)

        def get_tensor(self, key):
            tensor = self.handle.get_tensor(key)
            loaded.append(weakref.ref(tensor))
            assert sum(ref() is not None for ref in loaded) <= 2
            return tensor

    def tracked_open(path, **kwargs):
        handle = original_open(path, **kwargs)
        return TrackedAdapter(handle) if str(path).endswith("adapter_model.safetensors") else handle

    monkeypatch.setattr(merge_adapter, "safe_open", tracked_open)
    merge(str(base), str(adapter), str(output))
    assert len(loaded) == 6
