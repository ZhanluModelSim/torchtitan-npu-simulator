# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract tests for the partial optimizer-state patch.

Explicitly disable foreach so torch_npu's optimizer auto-selection does not
query NPU hardware on CPU-only runners.
"""

import torch
import torchtitan.components.checkpoint_utils
import torchtitan.components.optimizer
import torchtitan.components.optimizer.optimizer
import torchtitan.components.optimizer.utils
from torch.distributed import checkpoint as dcp
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig

from torchtitan_npu.patches.torchtitan.components import optimizer as optimizer_patch


def test_patch_replaces_loaded_optimizer_state_helpers():
    assert torchtitan.components.checkpoint_utils.init_optim_state is optimizer_patch.patched_init_optim_state
    assert torchtitan.components.optimizer.utils.init_optim_state is optimizer_patch.patched_init_optim_state
    assert torchtitan.components.optimizer.optimizer.init_optim_state is optimizer_patch.patched_init_optim_state


def test_init_optim_state_materializes_missing_state():
    initialized = torch.nn.Parameter(torch.ones(1))
    missing = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.AdamW([initialized, missing], foreach=False)
    initialized.grad = torch.ones_like(initialized)
    optim.step()
    saved_grad = initialized.grad
    saved_params = [param.detach().clone() for param in (initialized, missing)]
    saved_lr = optim.param_groups[0]["lr"]
    saved_state = {key: value.clone() for key, value in optim.state[initialized].items()}

    torchtitan.components.checkpoint_utils.init_optim_state(optim)

    assert initialized.grad is saved_grad
    assert missing.grad is None
    for key, value in saved_state.items():
        assert torch.equal(optim.state[initialized][key], value)
    assert optim.state[missing]["step"].item() == 0
    assert optim.state[missing]["exp_avg"].count_nonzero().item() == 0
    assert optim.state[missing]["exp_avg_sq"].count_nonzero().item() == 0
    for param, value in zip((initialized, missing), saved_params, strict=True):
        assert torch.equal(param, value)
    assert optim.param_groups[0]["lr"] == saved_lr


def test_init_optim_state_exposes_all_checkpoint_state_keys():
    initialized = torch.nn.Parameter(torch.ones(1))
    missing = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.AdamW(
        [
            {
                "params": [initialized, missing],
                "param_names": ["initialized", "missing"],
            }
        ],
        foreach=False,
    )
    initialized.grad = torch.ones_like(initialized)
    optim.step()

    torchtitan.components.checkpoint_utils.init_optim_state(optim)
    flat_state = torchtitan.components.checkpoint_utils.get_flat_optim_state_dict(optim)

    expected_state_keys = {
        f"state.{param_name}.{state_name}"
        for param_name in ("initialized", "missing")
        for state_name in ("step", "exp_avg", "exp_avg_sq")
    }
    actual_state_keys = {key for key in flat_state if key.startswith("state.")}
    assert actual_state_keys == expected_state_keys
    assert flat_state["state.missing.step"].item() == 0
    assert flat_state["state.missing.exp_avg"].count_nonzero().item() == 0
    assert flat_state["state.missing.exp_avg_sq"].count_nonzero().item() == 0


def test_materialized_state_matches_lazy_adamw_first_update():
    control_params = [torch.nn.Parameter(torch.tensor([1.0])), torch.nn.Parameter(torch.tensor([2.0]))]
    patched_params = [torch.nn.Parameter(param.detach().clone()) for param in control_params]
    control_optim = torch.optim.AdamW(control_params, lr=0.01, weight_decay=0.1, foreach=False)
    patched_optim = torch.optim.AdamW(patched_params, lr=0.01, weight_decay=0.1, foreach=False)

    for params, optim in (
        (control_params, control_optim),
        (patched_params, patched_optim),
    ):
        params[0].grad = torch.tensor([0.25])
        optim.step()
        optim.zero_grad(set_to_none=True)

    optimizer_patch.patched_init_optim_state(patched_optim)

    for params in (control_params, patched_params):
        params[0].grad = torch.tensor([-0.5])
        params[1].grad = torch.tensor([0.75])
    control_optim.step()
    patched_optim.step()

    for control_param, patched_param in zip(control_params, patched_params, strict=True):
        assert torch.equal(control_param, patched_param)
        control_state = control_optim.state[control_param]
        patched_state = patched_optim.state[patched_param]
        assert control_state.keys() == patched_state.keys()
        for state_name in control_state:
            assert torch.equal(control_state[state_name], patched_state[state_name])


def test_init_optim_state_is_idempotent_after_state_is_complete():
    param = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.AdamW([param], foreach=False)
    param.grad = torch.ones_like(param)
    optim.step()
    saved_grad = param.grad
    saved_state = {key: value.clone() for key, value in optim.state[param].items()}

    torchtitan.components.optimizer.optimizer.init_optim_state(optim)

    assert param.grad is saved_grad
    for key, value in saved_state.items():
        assert torch.equal(optim.state[param][key], value)


def _build_partial_state_optimizer():
    model = torch.nn.Linear(1, 1)
    config = OptimizersContainer.Config(
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.01})],
        implementation="for-loop",
    )
    return model, config.build(model_parts=[model])


def test_partial_optimizer_checkpoint_resume_matches_next_update(tmp_path):
    with torch.random.fork_rng():
        model, optim = _build_partial_state_optimizer()
        resumed_model, resumed_optim = _build_partial_state_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optim.step()
    optim.zero_grad(set_to_none=True)
    dcp.save({"model": model, "optimizer": optim}, checkpoint_id=tmp_path)

    dcp.load({"model": resumed_model, "optimizer": resumed_optim}, checkpoint_id=tmp_path)

    saved_state = optim.state_dict()
    loaded_state = resumed_optim.state_dict()
    assert loaded_state.keys() == saved_state.keys()
    for key, value in saved_state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(loaded_state[key], value, rtol=0, atol=0)
        else:
            assert loaded_state[key] == value
    assert loaded_state["state.bias.step"].item() == 0
    for original, restored in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
        original.grad = torch.full_like(original, 0.25)
        restored.grad = original.grad.clone()
    optim.step()
    resumed_optim.step()

    for original, restored in zip(model.parameters(), resumed_model.parameters(), strict=True):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
    for key, value in optim.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(resumed_optim.state_dict()[key], value, rtol=0, atol=0)
