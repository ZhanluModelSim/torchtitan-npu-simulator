# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PEFT completion files must describe a successfully published export."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from torchtitan_npu.models.deepseek_v4 import peft

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("failure", ["weights", "config", "rename_weights", "rename_config"])
def test_failed_export_never_publishes_completion(tmp_path, monkeypatch, failure):
    def interrupted_write(*args, **kwargs):
        raise OSError("injected export failure")

    replace = peft.os.replace
    failed_destination = "adapter_model.safetensors" if failure == "rename_weights" else "adapter_config.json"

    def interrupted_replace(source, destination):
        if Path(destination).name == failed_destination:
            interrupted_write()
        replace(source, destination)

    finalize = peft.DeepSeekV4PEFTCheckpointManager._finalize_peft_directory
    config = {"peft_type": "LORA"}
    weights = {"lora_a": torch.ones(2, 3)}
    previous = tmp_path / "step-1"
    finalize(str(previous), config, weights)
    previous_files = {path.name: path.read_bytes() for path in previous.iterdir()}
    folder = tmp_path / "step-2"
    with monkeypatch.context() as patch:
        if failure == "weights":
            patch.setattr(peft, "save_file", interrupted_write)
        elif failure == "config":
            patch.setattr(peft.json, "dump", interrupted_write)
        else:
            patch.setattr(peft.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="injected export failure"):
            finalize(str(folder), config, weights)
    assert not (folder / "adapter_config.json").exists()
    assert {path.name for path in folder.iterdir()} == (
        {"adapter_model.safetensors"} if failure == "rename_config" else set()
    )

    finalize(str(folder), config, weights)
    assert json.loads((folder / "adapter_config.json").read_text()) == config
    torch.testing.assert_close(load_file(folder / "adapter_model.safetensors"), weights)
    assert {path.name: path.read_bytes() for path in previous.iterdir()} == previous_files


@pytest.mark.parametrize("existing_file", [".metadata", "adapter_config.json"], ids=["native", "peft"])
def test_export_does_not_modify_existing_checkpoint(tmp_path, existing_file):
    folder = tmp_path / "step-1"
    folder.mkdir()
    original = folder / existing_file
    original.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError, match="already exists"):
        peft.DeepSeekV4PEFTCheckpointManager._finalize_peft_directory(
            str(folder), {"peft_type": "LORA"}, {"lora_a": torch.ones(2, 3)},
        )
    assert original.read_bytes() == b"existing checkpoint"
    assert list(folder.iterdir()) == [original]


def test_export_publishes_exact_payload_and_preserves_pending_manifest(tmp_path):
    folder = tmp_path / "step-1"
    folder.mkdir()
    pending = folder / "_checkpoint_hash_manifest.pending"
    pending.write_text("pending\n")
    weights = {"lora_a": torch.arange(6).reshape(2, 3).float()}
    config = {"peft_type": "LORA", "r": 2}
    peft.DeepSeekV4PEFTCheckpointManager._finalize_peft_directory(str(folder), config, weights)
    torch.testing.assert_close(load_file(folder / "adapter_model.safetensors"), weights, rtol=0, atol=0)
    assert json.loads((folder / "adapter_config.json").read_text()) == config
    assert pending.read_text() == "pending\n"
    assert sorted(path.name for path in folder.iterdir()) == [
        "_checkpoint_hash_manifest.pending", "adapter_config.json", "adapter_model.safetensors",
    ]


def test_stacked_shared_adapter_export_writes_independent_safetensors(tmp_path):
    from tests.unit_tests.models.deepseek_v4.lora_test_utils import _lora_model_config
    from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
    from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

    config = _lora_model_config()
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    state = {name: value for name, value in model.state_dict().items() if is_lora_parameter(name)}
    adapter = DeepSeekV4StateDictAdapter(config, hf_assets_path=None)
    expected = adapter.to_peft(state)
    gate = "base_model.model.model.layers.0.mlp.shared_experts.gate_proj.lora_A.weight"
    up = "base_model.model.model.layers.0.mlp.shared_experts.up_proj.lora_A.weight"
    assert expected[gate].data_ptr() == expected[up].data_ptr()
    manager = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
    manager.sd_adapter = adapter

    folder = tmp_path / "step-4"
    manager.finalize_peft_checkpoint(str(folder), adapter.peft_adapter_config(), state)

    restored = load_file(folder / "adapter_model.safetensors")
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    assert restored[gate].data_ptr() != restored[up].data_ptr()
    assert json.loads((folder / "adapter_config.json").read_text())["peft_type"] == "LORA"
