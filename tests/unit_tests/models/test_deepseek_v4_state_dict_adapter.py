# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / "torchtitan_npu" / "models" / "deepseek_v4" / "state_dict_adapter.py"


class _BaseAdapter:
    def __init__(self, model_config, hf_assets_path):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.grouped_expert_weight_placements = {}
        self.grouped_expert_weight_shape = {}
        self.grouped_expert_weight_mesh = {}
        self.local_experts_indices = {}

    @staticmethod
    def _split_experts_weights(value, num_experts):
        assert value.shape[0] == num_experts
        return list(value.unbind(0))

    @staticmethod
    def _concatenate_expert_weights(expert_weights, abstract, layer, num_experts):
        values = expert_weights[layer][abstract]
        if len(values) != num_experts:
            return None
        return torch.stack([values[index] for index in range(num_experts)])


def _config():
    router = types.SimpleNamespace(n_hash_layers=1)
    moe = types.SimpleNamespace(num_experts=2, router=router)
    return types.SimpleNamespace(
        n_layers=2,
        compress_ratios=(4, 1),
        layers=[types.SimpleNamespace(moe=moe), types.SimpleNamespace(moe=moe)],
    )


@pytest.fixture
def adapter_class(monkeypatch):
    package_names = [
        "torchtitan_npu",
        "torchtitan_npu.models",
        "torchtitan_npu.models.deepseek_v4",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    utils = types.ModuleType("torchtitan.models.utils")
    utils.MoEStateDictAdapter = _BaseAdapter
    monkeypatch.setitem(sys.modules, "torchtitan.models.utils", utils)
    upstream = types.ModuleType("torchtitan.models.deepseek_v3.state_dict_adapter")
    upstream.DeepSeekV3StateDictAdapter = type("DeepSeekV3StateDictAdapter", (), {"get_hf_storage_reader": None})
    monkeypatch.setitem(sys.modules, "torchtitan.models.deepseek_v3.state_dict_adapter", upstream)
    model = types.ModuleType("torchtitan_npu.models.deepseek_v4.model")
    model.DeepSeekV4Model = type("DeepSeekV4Model", (), {"Config": object})
    monkeypatch.setitem(sys.modules, "torchtitan_npu.models.deepseek_v4.model", model)

    module_name = "torchtitan_npu.models.deepseek_v4.state_dict_adapter"
    spec = importlib.util.spec_from_file_location(module_name, ADAPTER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DeepSeekV4StateDictAdapter


def test_root_attn_sink_and_tid2eid_round_trip(adapter_class):
    adapter = adapter_class(_config(), hf_assets_path=None)
    hf = {
        "embed.weight": torch.arange(6).reshape(2, 3),
        "layers.0.attn.attn_sink": torch.arange(4.0),
        "layers.0.ffn.gate.tid2eid": torch.tensor([[0.0, 1.0]]),
    }
    internal = adapter.from_hf(hf)
    assert internal["layers.0.attention.attn_sink.weight"].shape == (4, 1)
    assert internal["layers.0.moe.router.tid2eid"].dtype == torch.int64
    restored = adapter.to_hf(internal)
    assert restored["layers.0.attn.attn_sink"].shape == (4,)
    assert restored["layers.0.ffn.gate.tid2eid"].dtype == torch.float32
    for key, value in hf.items():
        torch.testing.assert_close(restored[key], value)


def test_compressor_and_indexer_mappings_preserve_values(adapter_class):
    adapter = adapter_class(_config(), hf_assets_path=None)
    hf = {
        "layers.0.attn.compressor.ape": torch.randn(2, 3),
        "layers.0.attn.indexer.wq_b.weight": torch.randn(3, 4),
        "layers.0.attn.indexer.compressor.wgate.weight": torch.randn(2, 2),
    }
    restored = adapter.to_hf(adapter.from_hf(hf))
    assert restored.keys() == hf.keys()
    for key, value in hf.items():
        torch.testing.assert_close(restored[key], value)


@pytest.mark.parametrize("weight", ["w1", "w2", "w3"])
def test_routed_expert_round_trip_preserves_values(adapter_class, weight):
    adapter = adapter_class(_config(), hf_assets_path=None)
    hf = {
        f"layers.1.ffn.experts.0.{weight}.weight": torch.arange(6.0).reshape(2, 3),
        f"layers.1.ffn.experts.1.{weight}.weight": torch.arange(6.0, 12.0).reshape(2, 3),
    }
    restored = adapter.to_hf(adapter.from_hf(hf))
    assert restored.keys() == hf.keys()
    for key, value in hf.items():
        torch.testing.assert_close(restored[key], value)
