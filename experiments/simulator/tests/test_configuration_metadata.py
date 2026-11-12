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
