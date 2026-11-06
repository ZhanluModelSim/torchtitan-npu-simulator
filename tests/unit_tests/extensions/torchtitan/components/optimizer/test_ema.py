# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Upstream EMA scheduling through the NPU configuration and offload adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _tracked_model():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([2.0, -2.0]))
    model.frozen = torch.nn.Parameter(torch.tensor([9.0]), requires_grad=False)
    model.register_buffer("expert_bias", torch.tensor([1.0, 3.0]))
    model.register_buffer("untracked", torch.tensor([5.0]))
    return model


@pytest.mark.parametrize("offload", [False, True], ids=["resident", "cpu-offload"])
@pytest.mark.parametrize("npu_recipe", [False, True], ids=["upstream-recipe", "npu-recipe"])
def test_config_manager_preserves_ema_schedule_and_selects_device_adapter(monkeypatch, tmp_path, offload, npu_recipe):
    from torchtitan.components.optim import Optim
    from torchtitan.components.optim.ema import EMA as _EMA
    from torchtitan.config import ConfigManager
    from torchtitan.models.llama3 import model_registry
    from torchtitan.trainer import Trainer

    from torchtitan_npu.extensions.torchtitan.components.optimizer.ema import EMA
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

    source_ema = _EMA.Config(
        decay=0.5,
        start_step=1,
        update_every_n_steps=2,
        offload_to_cpu=offload,
        buffer_patterns=[r"^expert_bias$"],
    )
    source = (TrainerEx.Config if npu_recipe else Trainer.Config)(
        model=model_registry("debugmodel"),
        hf_assets_path=str(tmp_path),
        optim=Optim.Config(ema=source_ema),
    )
    module_name = "_npu_ema_config_registry"
    registry = types.ModuleType(module_name)
    registry.ema_recipe = lambda: source
    monkeypatch.setitem(sys.modules, module_name, registry)

    config = ConfigManager().parse_args(["--module", module_name, "--config", "ema_recipe"])
    model = _tracked_model()
    ema = config.optim.ema.build(model_parts=[model])

    assert type(config.optim.ema) is EMA.Config
    assert type(ema) is EMA
    assert source.optim.ema is source_ema and type(source_ema) is _EMA.Config
    assert ema.offload_to_cpu is offload
    assert (ema.start_step, ema.update_every_n_steps, ema.decay) == (1, 2, 0.5)
    assert model.frozen not in ema._param_optimizers[0].state
    assert model.untracked not in ema._buffer_optimizers[0].state
    averaged_weight = ema._param_optimizers[0].state[model.weight]["ema_params"]
    averaged_bias = ema._buffer_optimizers[0].state[model.expert_bias]["ema_params"]

    with torch.no_grad():
        model.weight.copy_(torch.tensor([10.0, 6.0]))
        model.expert_bias.copy_(torch.tensor([5.0, 11.0]))
    ema.step(1)
    ema.step(2)
    torch.testing.assert_close(averaged_weight, torch.tensor([2.0, -2.0]), rtol=0, atol=0)
    torch.testing.assert_close(averaged_bias, torch.tensor([1.0, 3.0]), rtol=0, atol=0)
    ema.step(3)
    torch.testing.assert_close(averaged_weight, torch.tensor([6.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(averaged_bias, torch.tensor([3.0, 7.0]), rtol=0, atol=0)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([14.0, 10.0]))
        model.expert_bias.copy_(torch.tensor([9.0, 15.0]))
    ema.step(4)
    torch.testing.assert_close(averaged_weight, torch.tensor([6.0, 2.0]), rtol=0, atol=0)
    ema.step(5)
    torch.testing.assert_close(averaged_weight, torch.tensor([10.0, 6.0]), rtol=0, atol=0)
    torch.testing.assert_close(averaged_bias, torch.tensor([6.0, 11.0]), rtol=0, atol=0)


@pytest.fixture
def ema_module(monkeypatch):
    # The adapter has no import side effects; isolate only unrelated package
    # activation so the CPU offload branch also runs without a CANN runtime.
    path = Path(__file__).resolve().parents[6] / "torchtitan_npu/extensions/torchtitan/components/optimizer/ema.py"
    spec = importlib.util.spec_from_file_location("_npu_ema_definition", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_cpu_offload_ema_uses_native_dynamic_decay_and_step_bias(ema_module):
    model = _tracked_model()
    ema = ema_module.EMA.Config(half_life_fraction=0.5, step_bias=1, offload_to_cpu=True).build(model_parts=[model])
    averaged = ema._param_optimizers[0].state[model.weight]["ema_params"]
    with torch.no_grad():
        model.weight.fill_(6.0)

    ema.step(1)

    # The first firing plus bias is two updates: decay = 2**(-1/(0.5*2)) = 0.5.
    torch.testing.assert_close(averaged, torch.tensor([4.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(model.weight, torch.tensor([6.0, 6.0]), rtol=0, atol=0)
    assert averaged.data_ptr() != model.weight.data_ptr()
