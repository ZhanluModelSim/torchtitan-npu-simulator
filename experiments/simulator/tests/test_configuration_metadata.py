# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native configuration metadata without runtime or model construction."""

import subprocess
import sys

from torchtitan_simulator.configuration_metadata import describe_configuration


def test_configuration_metadata_keeps_runtime_outside_model_reset():
    catalog = describe_configuration()
    assert catalog["model_reset_paths"] == [["config"], ["model_overrides"], ["num_mtp_modules"]]
    models = {model["id"]: model for model in catalog["models"]}
    debug = models["deepseek_v4_smoketest"]
    assert debug["identity"] == [{"path": ["config"], "value": "deepseek_v4_smoketest"}]
    assert debug["assignments"][-1] == {"path": ["num_mtp_modules"], "value": 0}
    fields = {tuple(field["path"]): field for field in debug["fields"]}
    assert fields[("model_overrides", "dim")]["default"] == 256
    assert fields[("model_overrides", "n_layers")]["default"] == 4
    assert fields[("model_overrides", "dim")]["group"] == "model"
    assert fields[("num_mtp_modules",)]["group"] == "model"
    for name in ("qwen3_32b_smoketest", "kimi_k3_smoketest"):
        assert models[name]["limitations"]
        assert not any(field["path"][0] == "model_overrides" for field in models[name]["fields"])
    common = {tuple(field["path"]): field for field in catalog["fields"]}
    assert common[("seq_len",)]["default"] == 128
    assert common[("memory_offload_optimizer_states",)]["default"] is False
    assert common[("world_size",)]["group"] == "parallel"


def test_configuration_metadata_does_not_enter_capture_or_replace_tensor_metadata():
    script = """
import sys
import torchtitan_simulator
assert 'torch' not in sys.modules
from torchtitan_simulator.configuration_metadata import describe_configuration
from torchtitan_simulator.config import build_meta_model
import torchtitan_simulator.config as config
config.build_meta_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError('model construction'))
describe_configuration()
assert 'torchtitan_simulator.capture.dispatch_capture' not in sys.modules
assert 'torchtitan_simulator.runtime' not in sys.modules
from torchtitan_simulator.metadata import iter_metadata_tensors, to_meta_metadata
from torchtitan_simulator.runner import simulate
assert callable(simulate) and callable(iter_metadata_tensors) and callable(to_meta_metadata)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_draft_rejects_changed_dp_and_probes_repair_without_mutating_request():
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.configuration_metadata import describe_configuration_field

    draft = SimulationConfig(world_size=4, local_batch_size=1, global_batch_size=2)
    query = describe_configuration_field(draft, ["tensor_parallel_degree"], candidates=[1, 2, True])
    assert query["batch_configuration"] is None
    assert not query["validation"]["valid"]
    assert "global_batch_size=2" in query["validation"]["reason"]
    current, repaired, wrong_type = query["field"]["choices"]
    assert not current["available"] and current["reason"] == query["validation"]["reason"]
    assert repaired["available"] and repaired["reason"] is None
    assert not wrong_type["available"] and "integer" in wrong_type["reason"]
    assert draft.tensor_parallel_degree == 1


def test_draft_closed_fields_and_nested_geometry_probe_native_combinations():
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.configuration_metadata import describe_configuration_field

    config = SimulationConfig(selective_ac_save_ops=["comm"], activation_checkpoint_mode="selective")
    result = describe_configuration_field(config, ["activation_checkpoint_mode"])
    modes = {item["value"]: item for item in result["field"]["choices"]}
    assert set(modes) == {"none", "full", "selective"}
    assert modes["selective"]["available"]
    assert not modes["full"]["available"] and "selective" in modes["full"]["reason"]
    target = describe_configuration_field(config, ["target_npu_device_type"])
    assert [(i["value"], i["available"]) for i in target["field"]["choices"]] == [("non_a5", True), ("a5", False)]
    result = describe_configuration_field(config, ["model_overrides", "moe_args", "top_k"], candidates=[1, -1])
    assert [i["available"] for i in result["field"]["choices"]] == [True, False]
    assert config.model_overrides is None


def test_draft_discovery_rejects_custom_factory_and_imports_without_loading_them(monkeypatch):
    import pytest
    from torchtitan_simulator import config as config_module
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.configuration_metadata import describe_configuration_field

    def unexpected(name):
        pytest.fail(f"discovery imported an explicit module: {name}")

    monkeypatch.setattr(config_module, "import_module", unexpected)
    with pytest.raises(ValueError, match="registered preset"):
        describe_configuration_field(SimulationConfig(config="untrusted.factory"), ["config"])
    rejected = describe_configuration_field(
        SimulationConfig(override_imports=["untrusted.module"]),
        ["config"],
        candidates=["untrusted.factory"],
    )
    assert not rejected["validation"]["valid"] and "explicit override" in rejected["validation"]["reason"]
    candidate = rejected["field"]["choices"][0]
    assert not candidate["available"] and "registered preset" in candidate["reason"]


def test_capture_dtype_field_distinguishes_build_only_from_capture_preparation():
    import pytest
    from torchtitan_simulator.config import SimulationConfig, prepare_capture_config, prepare_config
    from torchtitan_simulator.configuration_metadata import describe_configuration_field

    config = SimulationConfig(training_dtype="bfloat16")
    assert prepare_config(config).trainer.training.dtype == "bfloat16"
    with pytest.raises(ValueError, match=r"DeepSeek V4 capture requires training\.dtype=float32"):
        prepare_capture_config(config)
    field = describe_configuration_field(config, ["training_dtype"])
    choices = {choice["value"]: choice for choice in field["field"]["choices"]}
    assert not choices["bfloat16"]["available"]
    assert choices["bfloat16"]["reason"] == field["validation"]["reason"]
    assert choices["float32"]["available"]
