# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import re
import sys
from copy import deepcopy
from types import ModuleType

import pytest
import torch
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.flex_shard import Owned
from torchtitan.models.common.linear import Linear

from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4.config_registry import (
    _dsv4_muon_profile,
    _dsv4_optimizer_config,
)
from torchtitan_npu.models.deepseek_v4.lora import (
    DeepSeekV4LoRAConverter, grouped_lora_class, is_lora_parameter, linear_lora_class,
)
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

_LORA_CONVERTER = DeepSeekV4LoRAConverter.Config(rank=4, alpha=8.0, rank_experts=4)


def _converted_debug_model() -> torch.nn.Module:
    spec = model_registry("debugmodel", num_mtp_layers=1, converters=[_LORA_CONVERTER])
    with torch.device("meta"):
        model = spec.model.build()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_lora_parameter(name))
    return model


@pytest.mark.parametrize("checkpointed", [False, True])
def test_muon_group_is_non_empty_and_every_member_has_a_compute_layout(monkeypatch, checkpointed):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torchtitan.components.checkpoint_utils import canonical_fqn

    base_spec = model_registry("debugmodel", num_mtp_layers=1)
    profile = _dsv4_muon_profile(base_spec)
    compute_sharding_by_fqn = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]

    model = _converted_debug_model()
    if checkpointed:
        for key, layer in list(model.layers.items()):
            model.layers[key] = checkpoint_wrapper(layer)
    trainable = [canonical_fqn(name) for name, param in model.named_parameters() if param.requires_grad]
    frozen = [canonical_fqn(name) for name, param in model.named_parameters() if not param.requires_grad]

    assert any(name.endswith(".weight") and not is_lora_parameter(name) for name in frozen)
    assert any(is_lora_parameter(name) for name in trainable)

    config = _dsv4_optimizer_config(base_spec, lr=1e-4)
    config.name = "Muon"
    config.materialize()
    assignment = {}

    def capture_muon(param_groups, **_kwargs):
        for group in param_groups:
            assignment.update(dict.fromkeys(group["param_names"], "DistMuon"))
        return torch.optim.AdamW(param_groups)

    monkeypatch.setattr("torchtitan.components.optimizer.optimizer.build_dist_muon", capture_muon)
    container = OptimizersContainer.Config(param_groups=config.param_groups, implementation="for-loop").build(
        model_parts=[model]
    )
    for optimizer in container.optimizers:
        for group in optimizer.param_groups:
            for name in group["param_names"]:
                assignment.setdefault(name, "AdamW")
    assert set(assignment) == set(trainable)
    assert not set(assignment).intersection(frozen)
    muon_fqns = [fqn for fqn, group in assignment.items() if group == "DistMuon"]
    adamw_fqns = [fqn for fqn, group in assignment.items() if group == "AdamW"]

    expected_muon = {name for name in trainable if not name.startswith("lm_head.")}
    assert set(muon_fqns) == expected_muon
    assert all(is_lora_parameter(fqn) for fqn in muon_fqns)
    missing = [fqn for fqn in muon_fqns if fqn not in compute_sharding_by_fqn]
    assert not missing, f"DistMuon parameters without a compute layout: {missing}"

    assert all(fqn.startswith("lm_head.") for fqn in adamw_fqns)
    assert any(fqn.endswith(".w1_lora_b") for fqn in muon_fqns)
    assert any(fqn.endswith(".w3_lora_b") for fqn in muon_fqns)

    assert any(fqn.endswith(".lora_a.weight") for fqn in muon_fqns)
    assert any(fqn.endswith(".attention.wq_b.lora_b.weight") for fqn in muon_fqns)
    assert any(fqn.endswith(".w13_lora_a") for fqn in muon_fqns)
    assert any(fqn.endswith(".w2_lora_b") for fqn in muon_fqns)


@pytest.mark.parametrize("mesh_axis", ["dp_shard", "dp_shard_cp"], ids=["dp", "dp-cp"])
def test_dsv4_muon_policy_resolves_combined_cp_storage_mesh(tmp_path, mesh_axis):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from torchtitan.distributed.flex_shard.dist_muon import _resolve_storage_to_compute_transition

    dist.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=(mesh_axis,))
        converters = [DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)]
        spec = model_registry("debugmodel", converters=converters)
        profile = _dsv4_muon_profile(spec)
        layouts = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]
        with torch.random.fork_rng(devices=[]):
            module = spec.model.layers[0].attention.wq_a.build()
            parameter = module.lora_a.weight
        fqn = "layers.0.attention.wq_a.lora_a.weight"
        tensor = distribute_tensor(parameter.detach(), mesh, [Shard(0)])
        transition = _resolve_storage_to_compute_transition(
            fqn,
            tensor,
            tensor.shape,
            None,
            layouts[fqn],
        )
        assert isinstance(transition.compute_sharding, Owned)
    finally:
        dist.destroy_process_group()


@pytest.fixture
def ao(monkeypatch):
    # CPU tests replace the numerical kernels below; isolate only CANN registration.
    # No torch.ops implementation is supplied, so an unmocked kernel still fails.
    monkeypatch.setitem(sys.modules, "cann_ops_nn", ModuleType("cann_ops_nn"))
    pytest.importorskip("torchao_npu")
    module = importlib.import_module("torchtitan_npu.extensions.adaptor.torchao.converter")
    monkeypatch.setattr(module, "_ensure_cann_ops_loaded", lambda: None)
    return module


def _lora_config(ao, kind):
    common = dict(rank=2, alpha=4.0, chunk_rows=2)
    if kind == "dense":
        return linear_lora_class(Linear).Config(in_features=4, out_features=8, bias=True, **common)
    if kind == "batched":
        return linear_lora_class(BatchedLinear).Config(n_heads=2, in_features=4, out_features=8, **common)
    return grouped_lora_class(ao.GroupedExperts).Config(dim=4, hidden_dim=8, num_experts=2, **common)


def _block_fp8_policy(ao):
    return ao.ParamSwapConfig(weight_config=ao.BlockMXQuantizeConfig(), activation_config=ao.MXQuantizeConfig())


def _convert(ao, config, policy=None):
    return ao.NpuQuantizeConverter.Config(base_config=policy or _block_fp8_policy(ao)).build().convert(config)


@pytest.mark.parametrize("kind", ["dense", "batched", "grouped"])
def test_quantization_preserves_adapter_weights_and_gradients(ao, kind, monkeypatch):
    from torch import _grouped_mm

    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    block = importlib.import_module("torchao_npu.wrapper_tensors.block_mx_wrapper_tensor")
    calls = []

    def matmul(a, b, *_configs):
        calls.append("base")
        return torch.matmul(a, b)

    def grouped(a, b, offsets, *_configs):
        if _configs:
            calls.append("base")
        parts, start = [], 0
        for expert, end in enumerate(offsets.tolist()):
            parts.append(torch.mm(a[start:end], b[expert]))
            start = end
        return torch.cat(parts)

    monkeypatch.setattr(block, "to_block_mx_then_mm", matmul)
    monkeypatch.setattr(block, "to_block_mx_then_bmm", matmul)
    monkeypatch.setattr(block, "to_block_mx_then_grouped_mm", grouped)

    def grouped_mm(self, *, A, B_t, offs):
        if isinstance(B_t, block.BlockMXTrainingWeightWrapperTensor):
            return _grouped_mm(A, B_t, offs=offs)
        return grouped(A, B_t, offs)

    config = _convert(ao, _lora_config(ao, kind))
    module = deepcopy(config).build().to(torch.bfloat16)
    if kind == "grouped":
        monkeypatch.setattr(type(module), "_grouped_mm", grouped_mm)
    adapters, bases = [], []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.normal_(std=0.1)
            if is_lora_parameter(name):
                adapters.append(parameter)
            else:
                # The model's parallelize hook freezes base weights after construction.
                parameter.requires_grad_(False)
                bases.append(parameter)
    assert adapters and all(isinstance(p, torch.nn.Parameter) and p.requires_grad for p in adapters)
    assert all(not isinstance(p, wrappers.BaseTrainingWeightWrapperTensor) for p in adapters)
    assert all(isinstance(p, wrappers.BaseTrainingWeightWrapperTensor) for p in bases)
    x = torch.randn((5, 2, 4) if kind == "batched" else (5, 4), dtype=torch.bfloat16, requires_grad=True)
    args = (torch.tensor([3, 2]),) if kind == "grouped" else ()
    module(x, *args).float().square().sum().backward()
    assert len(calls) == (3 if kind == "grouped" else 1)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapters)
    assert all(p.grad is None for p in bases)


def test_non_lora_quantization_keeps_original_policy(ao, monkeypatch):
    policy = _block_fp8_policy(ao)
    config = _convert(ao, Linear.Config(in_features=4, out_features=8), policy)
    received = []
    monkeypatch.setattr(ao, "quantize_", lambda _module, applied_policy, **_kwargs: received.append(applied_policy))
    config.build()
    assert len(received) == 1 and received[0] is policy


def test_parameter_filter_is_composed_without_mutating_shared_policy(ao):
    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    policy = _block_fp8_policy(ao)

    def bias_only(_parameter, name):
        return name == "bias"

    policy.params_filter_fn = bias_only
    original_filter = policy.params_filter_fn
    module = _convert(ao, _lora_config(ao, "dense"), policy).build()
    assert policy.params_filter_fn is original_filter
    assert isinstance(module.bias, wrappers.BaseTrainingWeightWrapperTensor)
    assert not isinstance(module.weight, wrappers.BaseTrainingWeightWrapperTensor)
    assert all(
        not isinstance(p, wrappers.BaseTrainingWeightWrapperTensor)
        for name, p in module.named_parameters()
        if is_lora_parameter(name)
    )


@pytest.mark.parametrize("excluded_base", [None, "w2_EDF"], ids=["all-base-weights", "preserve-base-filter"])
def test_grouped_quantization_keeps_all_split_adapters_floating(ao, excluded_base):
    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    policy = _block_fp8_policy(ao)

    def base_filter(_parameter, name):
        return name != excluded_base

    policy.params_filter_fn = base_filter
    module = _convert(ao, _lora_config(ao, "grouped"), policy).build()
    parameters = dict(module.named_parameters())
    adapters = {"w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b"}
    bases = {"w1_EFD", "w2_EDF", "w3_EFD"}
    assert parameters.keys() == adapters | bases
    for name in adapters:
        parameter = parameters[name]
        assert parameter.requires_grad and parameter.is_floating_point()
        assert not isinstance(parameter, wrappers.BaseTrainingWeightWrapperTensor), name
    for name in bases:
        assert isinstance(parameters[name], wrappers.BaseTrainingWeightWrapperTensor) == (name != excluded_base), name
    assert policy.params_filter_fn is base_filter


def test_module_replacement_policy_is_rejected_for_lora(ao):
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

    with pytest.raises(ValueError, match="Float8DynamicActivationFloat8WeightConfig module replacement"):
        _convert(ao, _lora_config(ao, "dense"), Float8DynamicActivationFloat8WeightConfig())


@pytest.mark.parametrize("recipe", ["all_mxfp8", "mix", "all_block_fp8"])
def test_registered_quantized_lora_accepts_compile_configuration(ao, recipe):
    from torchtitan.tools import utils

    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    wrappers = importlib.import_module("torchao_npu.wrapper_tensors")
    lora = DeepSeekV4LoRAConverter.Config(rank=2, rank_experts=2)
    config = deepseek_v4_debugmodel(converters=[lora])
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = recipe
    config.compile.enable = True
    config.extension.quantization.enable_fsdp_prequantize = recipe != "all_mxfp8"
    config.model_spec = ao.apply_quantization_converter(
        config.model_spec,
        config.extension.quantization,
        model_compile_enabled=True,
    )
    config.model_spec.model.update_from_config(config=config)
    with torch.device("meta"), utils.set_default_dtype(torch.bfloat16):
        model = config.model_spec.model.build()
    adapters = [p for name, p in model.named_parameters() if is_lora_parameter(name)]
    assert adapters and all(isinstance(p, torch.nn.Parameter) and p.requires_grad for p in adapters)
    assert all(not isinstance(p, wrappers.BaseTrainingWeightWrapperTensor) for p in adapters)


@pytest.mark.parametrize("empty_layers", [False, True])
def test_muon_without_lora_keeps_non_matrix_parameters_in_adamw(empty_layers):
    spec = model_registry("debugmodel", num_mtp_layers=0)
    if empty_layers:
        spec.model = deepcopy(spec.model)
        spec.model.n_layers = 0
        spec.model.layers = []
    pattern = re.compile(_dsv4_muon_profile(spec).muon_pattern)

    for name in ("norm.weight", "tok_embeddings.weight", "lm_head.weight", "layers.0.attention.norm.weight"):
        assert pattern.search(name) is None, name
    assert pattern.search("layers.0.attention.wq_a.weight") is not None
