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
from torch.distributed.checkpoint import FileSystemReader, HuggingFaceStorageReader


def _ensure_package_stub(module_name: str, path: Path) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    module.__path__ = [str(path)]


def _load_module(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_torchtitan_stubs() -> None:
    if importlib.util.find_spec("torchtitan") is not None:
        return

    torchtitan = types.ModuleType("torchtitan")
    models = types.ModuleType("torchtitan.models")
    deepseek_v3 = types.ModuleType("torchtitan.models.deepseek_v3")
    utils = types.ModuleType("torchtitan.models.utils")

    class MoEStateDictAdapter:
        def __init__(self, model_args, hf_assets_path=None):
            self.model_config = model_args
            self.hf_assets_path = hf_assets_path
            self.grouped_expert_weight_placements = {}
            self.grouped_expert_weight_shape = {}
            self.grouped_expert_weight_mesh = {}
            self.local_experts_indices = {}

    class DeepSeekV3StateDictAdapter(MoEStateDictAdapter):
        pass

    deepseek_v3.DeepSeekV3StateDictAdapter = DeepSeekV3StateDictAdapter
    utils.MoEStateDictAdapter = MoEStateDictAdapter
    torchtitan.models = models
    models.deepseek_v3 = deepseek_v3
    models.utils = utils
    sys.modules.update(
        {
            "torchtitan": torchtitan,
            "torchtitan.models": models,
            "torchtitan.models.deepseek_v3": deepseek_v3,
            "torchtitan.models.utils": utils,
        }
    )


def _load_dsv4_adapter_class():
    module_name = "torchtitan_npu.models.deepseek_v4.state_dict_adapter"
    if module_name in sys.modules:
        return sys.modules[module_name].DeepSeekV4StateDictAdapter

    repo_root = Path(__file__).resolve().parents[3]
    npu_root = repo_root / "torchtitan_npu"
    _ensure_package_stub("torchtitan_npu", npu_root)
    _ensure_package_stub("torchtitan_npu.models", npu_root / "models")
    _ensure_package_stub("torchtitan_npu.models.deepseek_v4", npu_root / "models" / "deepseek_v4")
    _ensure_package_stub("torchtitan_npu.tools", npu_root / "tools")
    _ensure_torchtitan_stubs()

    _load_module("torchtitan_npu.tools.weight_utils", npu_root / "tools" / "weight_utils.py")
    module = _load_module(module_name, npu_root / "models" / "deepseek_v4" / "state_dict_adapter.py")
    return module.DeepSeekV4StateDictAdapter


DeepSeekV4StateDictAdapter = _load_dsv4_adapter_class()


def _make_model_args(*, n_layers: int = 4, num_mtp_modules: int = 2):
    model_args = types.SimpleNamespace(
        n_layers=n_layers,
        num_mtp_modules=num_mtp_modules,
        compress_ratios=(1, 1, 4, 128),
        moe_args=types.SimpleNamespace(
            n_hash_layers=3,
            use_grouped_mm=True,
            num_experts=256,
        ),
    )
    if len(model_args.compress_ratios) < model_args.n_layers:
        model_args.compress_ratios = tuple([1] * model_args.n_layers)
    return model_args


def test_adapter_init_supports_deepseek_v4_flat_config_without_layers_attribute():
    model_args = _make_model_args()
    assert not hasattr(model_args, "layers")

    DeepSeekV4StateDictAdapter(model_args, hf_assets_path=None)


def test_mtp_mapping_round_trip_preserves_each_mtp_layer_index():
    model_args = _make_model_args(n_layers=4, num_mtp_modules=2)
    adapter = DeepSeekV4StateDictAdapter(model_args, hf_assets_path=None)

    hf_input_dict = {
        "mtp.0.enorm.weight": torch.randn(2),
        "mtp.1.hnorm.weight": torch.randn(2),
        "mtp.0.e_proj.weight": torch.randn(2, 2),
        "mtp.1.h_proj.weight": torch.randn(2, 2),
        "mtp.0.norm.weight": torch.randn(2),
        "mtp.1.hc_head_fn": torch.randn(2),
        "mtp.0.hc_head_base": torch.randn(2),
        "mtp.1.hc_head_scale": torch.randn(2),
    }

    from_hf_dict = adapter.from_hf_mtp(hf_input_dict)
    assert "layers.4.enorm.weight" in from_hf_dict
    assert "layers.5.hnorm.weight" in from_hf_dict
    assert "layers.4.e_proj.weight" in from_hf_dict
    assert "layers.5.h_proj.weight" in from_hf_dict
    assert "layers.4.norm.weight" in from_hf_dict
    assert "layers.5.hc_head_fn" in from_hf_dict
    assert "layers.4.hc_head_base" in from_hf_dict
    assert "layers.5.hc_head_scale" in from_hf_dict

    to_hf_dict = adapter.to_hf_mtp(from_hf_dict)
    assert set(to_hf_dict.keys()) == set(hf_input_dict.keys())
    assert torch.equal(to_hf_dict["mtp.0.enorm.weight"], hf_input_dict["mtp.0.enorm.weight"])
    assert torch.equal(to_hf_dict["mtp.1.hnorm.weight"], hf_input_dict["mtp.1.hnorm.weight"])
    assert torch.equal(to_hf_dict["mtp.0.e_proj.weight"], hf_input_dict["mtp.0.e_proj.weight"])
    assert torch.equal(to_hf_dict["mtp.1.h_proj.weight"], hf_input_dict["mtp.1.h_proj.weight"])
    assert torch.equal(to_hf_dict["mtp.0.norm.weight"], hf_input_dict["mtp.0.norm.weight"])
    assert torch.equal(to_hf_dict["mtp.1.hc_head_fn"], hf_input_dict["mtp.1.hc_head_fn"])
    assert torch.equal(to_hf_dict["mtp.0.hc_head_base"], hf_input_dict["mtp.0.hc_head_base"])
    assert torch.equal(to_hf_dict["mtp.1.hc_head_scale"], hf_input_dict["mtp.1.hc_head_scale"])


def test_split_w13_for_mapping_preserves_values_instead_of_placeholders():
    model_args = _make_model_args()
    adapter = DeepSeekV4StateDictAdapter(model_args, hf_assets_path=None)

    fused = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    state_dict = {
        "layers.0.moe.experts.w13": fused,
        "layers.0.attention_norm.weight": torch.ones(3),
    }

    split_dict = adapter._split_w13_for_mapping(state_dict)

    expected_w1, expected_w3 = torch.chunk(fused, 2, dim=1)
    torch.testing.assert_close(split_dict["layers.0.moe.experts.w1"], expected_w1)
    torch.testing.assert_close(split_dict["layers.0.moe.experts.w3"], expected_w3)
    assert "layers.0.attention_norm.weight" in split_dict


def test_storage_reader_rejects_quantized_dcp_but_preserves_other_paths(monkeypatch, tmp_path):
    adapter_module = sys.modules[DeepSeekV4StateDictAdapter.__module__]
    adapter = DeepSeekV4StateDictAdapter(_make_model_args(), hf_assets_path=None)

    monkeypatch.setattr(adapter_module, "detect_input_format_by_path", lambda _path: "dcp")
    with pytest.raises(NotImplementedError, match="quantized DCP checkpoints is not supported"):
        adapter.get_hf_storage_reader(str(tmp_path), from_quantized=True)
    assert isinstance(adapter.get_hf_storage_reader(str(tmp_path)), FileSystemReader)

    monkeypatch.setattr(adapter_module, "detect_input_format_by_path", lambda _path: "hf")
    assert isinstance(adapter.get_hf_storage_reader(str(tmp_path)), HuggingFaceStorageReader)
    assert isinstance(
        adapter.get_hf_storage_reader(str(tmp_path), from_quantized=True),
        adapter_module.MXFP8HuggingFaceStorageReader,
    )
