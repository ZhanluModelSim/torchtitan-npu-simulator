# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic recurrent execution independent of a paper, gate or loss recipe."""

import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
import torch
from raw_reference import LoopedModel, LoopSpec, ModelConfig, RoutingPlan, load_raw_reference


def config(**updates):
    values = dict(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=0,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dtype=torch.float32,
    )
    values.update(updates)
    return ModelConfig(**values)


def spec(**updates):
    values = dict(loop_start=1, loop_end=3, max_depth=3, normalize_carry=False)
    values.update(updates)
    return LoopSpec(**values)


def inputs(device="cpu"):
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    positions = torch.arange(4, device=device).expand(2, 4)
    visible = torch.ones((4, 4), dtype=torch.bool, device=device).tril()
    return tokens, positions, visible


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(451)


def test_raw_loop_contract_has_no_paper_policy_or_recipe_dependencies():
    module = load_raw_reference()
    assert module.MODEL_CONTRACT_VERSION == 3
    assert module.STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION == 1
    assert module.LOOP_CONTRACT_VERSION == 1
    assert {field.name for field in fields(LoopSpec)} == {"loop_start", "loop_end", "max_depth", "normalize_carry"}
    assert not any("alodlm" in name.lower() or "wedlm" in name.lower() for name in vars(module))
    source = Path(module.__file__).read_text().lower()
    assert "from recipes" not in source
    assert "import recipes" not in source
    assert "alodlm" not in source


@pytest.mark.parametrize(
    "readout,expected_calls,head_calls",
    [
        ("last", [1, 3, 3, 1], 1),
        ("each_step", [1, 3, 3, 3], 3),
        ("none", [1, 3, 3, 0], 0),
    ],
)
def test_plain_fixed_depth_loop_controls_readout_cost_without_any_gate(readout, expected_calls, head_calls):
    model = LoopedModel(config(), spec())
    calls = [0, 0, 0, 0]
    heads = []

    def count(index):
        def hook(_module, _arguments):
            calls[index] += 1

        return hook

    handles = [layer.register_forward_pre_hook(count(index)) for index, layer in enumerate(model.backbone.layers)]
    handles.append(
        model.backbone.lm_head.register_forward_pre_hook(lambda _module, arguments: heads.append(arguments[0].shape))
    )
    result = model(*inputs(), readout=readout)
    for handle in handles:
        handle.remove()
    assert calls == expected_calls
    assert len(heads) == head_calls
    assert len(result.readouts) == head_calls
    assert result.state.depth == 3
    assert result.state.carry.shape == (2, 4, 8)
    assert model.layer_evaluations(steps=3, readout_count=head_calls) == sum(expected_calls)
    assert not any("exit_gate" in name or "depth_bias" in name for name, _ in model.named_parameters())
    if head_calls:
        assert result.readouts[-1].logits.shape == (2, 4, 17)
        result.readouts[-1].logits.square().mean().backward()
        assert model.backbone.embed_tokens.weight.grad is not None


def test_readout_features_can_skip_vocabulary_projection():
    model = LoopedModel(config(), spec())
    heads = []
    handle = model.backbone.lm_head.register_forward_pre_hook(lambda *_: heads.append(True))
    result = model(*inputs(), readout="each_step", project_logits=False)
    handle.remove()
    assert not heads
    assert len(result.readouts) == 3
    assert all(readout.features.shape == (2, 4, 8) and readout.logits is None for readout in result.readouts)


def test_physical_parameters_are_independent_of_requested_loop_depth():
    shallow = LoopedModel(config(), spec(max_depth=1))
    deep = LoopedModel(config(), spec(max_depth=7))
    assert shallow.state_dict().keys() == deep.state_dict().keys()
    assert sum(parameter.numel() for parameter in shallow.parameters()) == sum(
        parameter.numel() for parameter in deep.parameters()
    )
    for key, tensor in shallow.state_dict().items():
        assert tensor.shape == deep.state_dict()[key].shape


def test_readout_does_not_change_recurrent_state_even_when_coda_weights_change():
    first = LoopedModel(config(), spec())
    second = LoopedModel(config(), spec())
    second.load_state_dict(first.state_dict(), strict=True)
    with torch.no_grad():
        second.backbone.layers[3].mlp.down_proj.weight.add_(0.7)
    result1 = first(*inputs(), readout="each_step")
    result2 = second(*inputs(), readout="each_step")
    torch.testing.assert_close(result1.state.carry, result2.state.carry, atol=0, rtol=0)
    assert not torch.allclose(result1.readouts[-1].logits, result2.readouts[-1].logits)


def test_step_is_functional_and_enforces_depth_limit():
    model = LoopedModel(config(), spec(max_depth=2))
    initial = model.begin(*inputs())
    saved = initial.carry.detach().clone()
    first = model.step(initial, readout=False)
    torch.testing.assert_close(initial.carry, saved, atol=0, rtol=0)
    assert initial.depth == 0
    assert first.state.depth == 1
    assert first.state is not initial
    assert first.features is None and first.logits is None
    with pytest.raises(FrozenInstanceError):
        initial.depth = 1
    second = model.step(first.state, readout=False)
    assert second.state.depth == 2
    with pytest.raises(ValueError):
        model.step(second.state, readout=False)


def test_external_feedback_is_differentiable_and_applied_after_carry_normalization():
    model = LoopedModel(config(), spec(normalize_carry=True))
    first = model.step(model.begin(*inputs()), readout=False)
    feedback = torch.randn((2, 4, 8), requires_grad=True)
    selection = torch.tensor([[False, True, False, False], [True, False, True, False]])
    received = []
    handle = model.backbone.layers[1].register_forward_pre_hook(
        lambda _module, arguments: received.append(arguments[0])
    )
    second = model.step(first.state, feedback=feedback, feedback_mask=selection, readout=False)
    handle.remove()
    previous = first.state.carry
    normalized = previous.float() * torch.rsqrt(
        previous.float().square().mean(-1, keepdim=True) + model.backbone.config.rms_norm_eps
    )
    normalized = normalized.to(previous.dtype) * model.backbone.norm.weight
    expected_input = torch.where(selection[..., None], feedback, normalized)
    torch.testing.assert_close(received[0], expected_input, atol=0, rtol=0)
    gradient = torch.autograd.grad(second.state.carry.square().sum(), feedback)[0]
    assert gradient.isfinite().all()
    assert torch.count_nonzero(gradient[~selection]) == 0
    assert gradient[selection].abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("moe", [False, True])
def test_generic_loop_supports_dense_and_moe_with_explicit_meta_replay(device, moe):
    cfg = config(num_experts=4 if moe else 0)
    with torch.device(device):
        model = LoopedModel(cfg, spec())
    replay = RoutingPlan.balanced(8, 4, 2) if moe else None
    result = model(*inputs(device), readout="last", routing_plan=replay)
    assert result.readouts[-1].logits.shape == (2, 4, 17)
    result.readouts[-1].logits.sum().backward()
    assert model.backbone.embed_tokens.weight.grad is not None
    if device == "cpu":
        assert result.readouts[-1].logits.isfinite().all()


def test_meta_loop_with_external_feedback_captures_without_value_reads():
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.device("meta"):
        model = LoopedModel(config(num_experts=4), spec())
        tokens, positions, mask = inputs("meta")
        feedback = torch.empty((2, 4, 8))
        selection = torch.empty((2, 4), dtype=torch.bool)
    replay = RoutingPlan.balanced(8, 4, 2)

    def operation(ids, values):
        state = model.begin(ids, positions, mask, routing_plan=replay)
        return model.step(state, feedback=values, feedback_mask=selection, routing_plan=replay).logits

    graph = make_fx(operation)(tokens, feedback)
    assert graph(tokens, feedback).shape == (2, 4, 17)
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)


@pytest.mark.parametrize("arguments", [{"steps": 0}, {"steps": 4}, {"readout": "adaptive_gate"}])
def test_fixed_loop_rejects_invalid_execution_controls(arguments):
    model = LoopedModel(config(), spec())
    with pytest.raises(ValueError):
        model(*inputs(), **arguments)


def test_feedback_requires_matching_mask_shape_and_dtype():
    model = LoopedModel(config(), spec())
    state = model.begin(*inputs())
    with pytest.raises(ValueError):
        model.step(state, feedback=torch.zeros((2, 4, 8)))
    with pytest.raises(ValueError):
        model.step(state, feedback_mask=torch.ones((2, 4), dtype=torch.bool))
    with pytest.raises(ValueError):
        model.step(state, feedback=torch.zeros((2, 4, 8)), feedback_mask=torch.ones((2, 4)))


def test_legacy_standard_checkpoint_conversion_rejects_qk_norm_parameters():
    from legacy_state_dict import to_legacy_state_dict

    model = LoopedModel(config(qk_norm=True), spec())
    with pytest.raises(ValueError):
        to_legacy_state_dict(model.backbone.state_dict())


def test_copied_model_file_runs_standalone_in_isolated_interpreter(tmp_path):
    """Private consumers receive this file, not the public repository package."""
    source = Path(load_raw_reference().__file__)
    delivered = tmp_path / "model.py"
    shutil.copyfile(source, delivered)
    script = r"""
import importlib.util
import json
import sys
import torch

spec = importlib.util.spec_from_file_location("standalone_handoff_source", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.__file__ == sys.argv[1]
results = []
for device in ("cpu", "meta"):
    cfg = module.ModelConfig(
        vocab_size=17, hidden_size=8, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        intermediate_size=12,
    )
    with torch.device(device):
        standard = module.BlockDiffusionModel(cfg)
        loop = module.LoopedModel(cfg, module.LoopSpec(loop_start=1, loop_end=2, max_depth=2))
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        block = torch.tensor([[0, 0]], dtype=torch.long)
        positions = torch.arange(3).unsqueeze(0)
        visible = torch.ones((3, 3), dtype=torch.bool).tril()
    cache = standard(prompt, phase="prefill", attention_policy="causal", logits_positions=()).cache
    denoised = standard(block, cache=cache, phase="denoise")
    committed = standard(block, cache=cache, phase="commit", logits_positions=())
    recurrent = loop(prompt, positions, visible, readout="last")
    assert denoised.logits.shape == (1, 2, 17)
    assert denoised.cache is cache
    assert cache.length == 3 and committed.cache.length == 5
    assert recurrent.state.depth == 2
    assert recurrent.readouts[-1].logits.shape == (1, 3, 17)
    results.append({"device": device, "standard": list(denoised.logits.shape), "loop": list(recurrent.readouts[-1].logits.shape)})
blocked = ("torchtitan", "torchtitan_npu", "raw_reference", "recipes")
repository_modules = [name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)]
assert not repository_modules, repository_modules
print(json.dumps({"results": results, "repository_modules": repository_modules}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(delivered)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert [entry["device"] for entry in report["results"]] == ["cpu", "meta"]
    assert report["repository_modules"] == []
    assert delivered.read_bytes() == source.read_bytes()
