# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import re
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC

from scripts.lora.merge_adapter import merge
from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import preprocess
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models import deepseek_v4_1 as registry
from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4_1 import _make_v41_config, parallelize
from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    _v41_muon_profile,
    _v41_optimizer_config,
    deepseek_v4_1_debugmodel_lora,
    deepseek_v4_1_flash_lora,
    deepseek_v4_1_flash_text_lora,
)
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable
from torchtitan_npu.models.deepseek_v4_1.lora import DeepSeekV41LoRAConverter
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41MultimodalModel
from torchtitan_npu.models.deepseek_v4_1.vision.data import build_image_token_layout
from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import DeepSeekV41DataLoader
from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import AlignedHuggingfaceDataloader


@pytest.mark.parametrize(
    "recipe,flavor,vision,loader_type",
    [
        (deepseek_v4_1_flash_lora, "deepseek_v4_1_flash", True, DeepSeekV41DataLoader.Config),
        (deepseek_v4_1_flash_text_lora, "deepseek_v4_1_flash_text", False, AlignedHuggingfaceDataloader.Config),
    ],
    ids=["multimodal", "text"],
)
def test_flash_lora_recipes_select_model_loader_and_checkpoint(recipe, flavor, vision, loader_type):
    config = recipe()
    model = config.model_spec.model
    assert config.model_spec.flavor == flavor
    assert isinstance(model, DeepSeekV41MultimodalModel.Config) == vision
    assert model.dim == 5120 and model.n_layers == 40
    assert type(config.dataloader) is loader_type
    assert model.lora.rank == model.lora.rank_experts == 16
    assert isinstance(config.checkpoint, DeepSeekV41LoRACheckpointManager.Config)
    assert config.checkpoint.enable and config.checkpoint.save_training_state
    assert not config.checkpoint.last_save_model_only


@pytest.fixture
def tiny_registry(monkeypatch):
    make_config = _make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(dim=8, n_heads=2, head_dim=8, rope_head_dim=4, q_lora_rank=8,
                      o_lora_rank=4, n_groups=1, index_n_heads=2, index_head_dim=4,
                      moe_inter_dim=16, vision_dim=8, vision_heads=2, vision_inter_dim=16,
                      vocab_size=64)
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    with torch.random.fork_rng(devices=[]):
        yield


def _parallelize_cpu(model, monkeypatch, tmp_path, checkpointed):
    # Only device placement is replaced; AC, table setup and freezing run normally.
    monkeypatch.setattr(parallelize, "apply_fsdp_to_decoder", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "torchtitan_npu.extensions.torchtitan.distributed.fsdp.apply_fsdp_to_vision_encoder", lambda *args, **kwargs: None,
    )
    mesh = SimpleNamespace(size=lambda: 1)
    dims = SimpleNamespace(ep_enabled=False, dp_replicate_enabled=False, pp_enabled=False,
                           ep=1, get_mesh=lambda names: mesh)
    return parallelize.parallelize_deepseek_v4_1(
        model, parallel_dims=dims, training=TrainingConfig(),
        parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
        compile_config=CompileConfig(enable=False),
        ac_config=FullAC.Config() if checkpointed else None, dump_folder=str(tmp_path),
    )


@pytest.mark.parametrize("checkpointed", [False, True], ids=["eager", "full-ac"])
def test_lora_training_freezes_engram_and_native_dcp_resumes(tiny_registry, monkeypatch, tmp_path, checkpointed):
    options = DeepSeekV41LoRAConverter.Config(rank=8, alpha=16, rank_experts=8)
    spec = registry.model_registry("deepseek_v4_1_debugmodel_text", converters=[options])
    base = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel_text").model)
    model = build_cpu_model(spec.model)
    # Initialization order differs with adapters. Copy the same base for zero-delta parity.
    model.load_state_dict(base.state_dict(), strict=False)
    _parallelize_cpu(model, monkeypatch, tmp_path, checkpointed)
    optimizer_config = _v41_optimizer_config(spec)
    optimizer_config.materialize()
    container = optimizer_config.build(model_parts=[model])
    optimizer = container.optimizers[0]
    spec.post_optimizer_build_fn(container, [model], None)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    tables = [m for m in model.modules() if isinstance(m, HostEngramTable)]
    assert len(tables) == 2
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for n, p in model.named_parameters() if is_lora_parameter(n)
    }
    biases = {n: b.clone() for n, b in model.named_buffers() if n.endswith("expert_bias_E")}
    tokens = torch.arange(16).reshape(1, -1)
    positions = torch.arange(16).reshape(1, -1)

    def forward(target):
        tokens_in, _, kwargs = preprocess(target, tokens, tokens, positions)
        return target(tokens_in, **kwargs)

    with torch.no_grad():
        torch.testing.assert_close(forward(model), forward(base), rtol=0, atol=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if is_lora_parameter(n)}
    forward(model).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
    for name, expected in frozen.items():
        torch.testing.assert_close(model.get_parameter(name), expected, rtol=0, atol=0)
        assert model.get_parameter(name).grad is None
    for name, expected in biases.items():
        torch.testing.assert_close(model.get_buffer(name), expected, rtol=0, atol=0)
    assert all(table.pending_sparse_grad() is None for table in tables)
    assert all(not table(tokens, positions).requires_grad for table in tables)

    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.save(state, checkpoint_id=tmp_path / "checkpoint")
    expected_state = copy.deepcopy(state)
    forward(model).square().mean().backward()
    optimizer.step()
    expected_output = forward(model).detach()
    loaded = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.load(loaded, checkpoint_id=tmp_path / "checkpoint")
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    torch.testing.assert_close(loaded["model"], expected_state["model"], rtol=0, atol=0)
    torch.testing.assert_close(loaded["optimizer"]["state"], expected_state["optimizer"]["state"], rtol=0, atol=0)
    assert loaded["optimizer"]["param_groups"] == expected_state["optimizer"]["param_groups"]
    optimizer.zero_grad(set_to_none=True)
    forward(model).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(forward(model), expected_output, rtol=0, atol=0)


def test_lora_host_table_stays_frozen_after_state_initialization(tiny_registry, monkeypatch, tmp_path):
    base = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel_text").model)
    spec = registry.model_registry(
        "deepseek_v4_1_debugmodel_text", converters=[DeepSeekV41LoRAConverter.Config()]
    )
    model = build_cpu_model(spec.model)
    _parallelize_cpu(model, monkeypatch, tmp_path, False)
    model.to_empty(device=torch.device("cpu"))
    model.init_states(buffer_device=torch.device("cpu"))
    table = next(module for module in model.modules() if isinstance(module, HostEngramTable))
    base_table = next(module for module in base.modules() if isinstance(module, HostEngramTable))
    assert type(base_table) is HostEngramTable
    table.load_state_dict(base_table.state_dict())
    tokens = torch.arange(16).reshape(1, -1)
    positions = torch.arange(16).reshape(1, -1)
    frozen_rows = table(tokens, positions)
    trainable_rows = base_table(tokens, positions)
    torch.testing.assert_close(frozen_rows, trainable_rows, rtol=0, atol=0)
    assert not frozen_rows.requires_grad
    assert table.pending_sparse_grad() is None
    trainable_rows.square().sum().backward()
    assert base_table.pending_sparse_grad() is not None
    assert torch.is_grad_enabled()


def test_multimodal_lora_updates_adapters_with_frozen_vision_and_engram(tiny_registry, monkeypatch, tmp_path):
    spec = registry.model_registry("deepseek_v4_1_debugmodel", converters=[DeepSeekV41LoRAConverter.Config()])
    model = build_cpu_model(spec.model)
    base = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel").model)
    model.load_state_dict(base.state_dict(), strict=False)
    _parallelize_cpu(model, monkeypatch, tmp_path, True)
    assert isinstance(model, DeepSeekV41MultimodalModel)
    assert model.vision_encoder is not None
    assert model.lora_config.rank == 64
    assert model.layers["1"].engram is not None
    ids, types, feature_ids = build_image_token_layout([(3, 3)], span_start=2, vocab_size=64)
    tokens = torch.arange(128).remainder(32).unsqueeze(0)
    image_span_end = ids.numel()
    tokens[:, 2:image_span_end] = ids[2:]
    token_types = torch.full_like(tokens, -1)
    token_types[:, : types.numel()] = types
    indices = torch.full_like(tokens, -1)
    indices[:, : feature_ids.numel()] = feature_ids
    inputs = dict(
        positions=torch.arange(128).unsqueeze(0),
        pixel_values=torch.linspace(-1, 1, 9 * 588).reshape(1, 9, 588),
        image_grid=torch.tensor([[3, 3]]),
        image_feature_indices=indices,
        token_types=token_types,
    )

    def forward(target):
        modality = {name: value for name, value in inputs.items() if name != "positions"}
        tokens_in, _, kwargs = preprocess(target, tokens, tokens, inputs["positions"], **modality)
        return target(tokens_in, **kwargs)

    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert any(n.startswith("vision_encoder.") for n in frozen)
    assert any(".engram." in n for n in frozen)
    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], foreach=False)
    with torch.no_grad():
        torch.testing.assert_close(forward(model), forward(base), rtol=0, atol=0)
    forward(model).square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(model.get_parameter(n), value) for n, value in adapters.items())
    for name, expected in frozen.items():
        torch.testing.assert_close(model.get_parameter(name), expected, rtol=0, atol=0)
        assert model.get_parameter(name).grad is None
    assert all(table.pending_sparse_grad() is None for table in model.modules() if isinstance(table, HostEngramTable))


def test_v41_adapter_only_resume_restores_engram_and_next_step(tiny_registry, monkeypatch, tmp_path):
    recipe = deepseek_v4_1_debugmodel_lora()
    recipe.checkpoint.initial_load_path = str(tmp_path / "base")
    recipe.checkpoint.folder = "checkpoint"
    recipe.checkpoint.interval = 1
    recipe.checkpoint.keep_latest_k = 0
    recipe.checkpoint.async_mode = "disabled"
    base = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel_text").model)
    dcp.save(base.state_dict(), checkpoint_id=recipe.checkpoint.initial_load_path)

    def build_training_state():
        model = build_cpu_model(recipe.model_spec.model)
        _parallelize_cpu(model, monkeypatch, tmp_path, False)
        config = copy.deepcopy(recipe.optimizer)
        config.materialize()
        optimizers = config.build(model_parts=[model])
        recipe.model_spec.post_optimizer_build_fn(optimizers, [model], None)
        manager = recipe.checkpoint.build(
            dataloader=None, model_parts=[model], optimizers=optimizers,
            lr_schedulers=None, states={}, base_folder=str(tmp_path),
            sd_adapter=recipe.model_spec.state_dict_adapter(recipe.model_spec.model, None),
        )
        assert isinstance(manager, DeepSeekV41LoRACheckpointManager)
        assert manager.periodic_save_adapter_only and manager.save_training_state
        return model, optimizers, manager

    tokens = torch.arange(16).reshape(1, -1)

    def train_step(model, optimizers):
        optimizers.zero_grad()
        tokens_in, _, kwargs = preprocess(model, tokens, tokens, tokens)
        loss = model(tokens_in, **kwargs).square().mean()
        loss.backward()
        optimizers.step()
        return loss.detach()

    model, optimizers, source = build_training_state()
    target = None
    try:
        assert source.load()
        train_step(model, optimizers)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizers.state_dict())
        assert source.save(1)
        keys = dcp.FileSystemReader(tmp_path / "checkpoint" / "step-1").read_metadata().state_dict_metadata
        adapters = {n for n, _ in model.named_parameters() if is_lora_parameter(n)}
        frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
        buffers = set(model.state_dict()) - {n for n, _ in model.named_parameters()}
        assert adapters and adapters <= keys.keys()
        assert buffers and buffers <= keys.keys()
        assert not frozen.intersection(keys)
        assert any(".engram." in n for n in frozen)
        assert any(n.startswith("optimizer.") for n in keys)
        expected_loss = train_step(model, optimizers)
        expected_next_model = copy.deepcopy(model.state_dict())
        expected_next_optimizer = copy.deepcopy(optimizers.state_dict())

        restored, restored_optimizers, target = build_training_state()
        with torch.no_grad():
            for parameter in restored.parameters():
                parameter.zero_()
        assert target.load(step=1)
        torch.testing.assert_close(restored.state_dict(), expected_model, rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizers.state_dict(), expected_optimizer, rtol=0, atol=0)
        torch.testing.assert_close(train_step(restored, restored_optimizers), expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(restored.state_dict(), expected_next_model, rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizers.state_dict(), expected_next_optimizer, rtol=0, atol=0)
    finally:
        source.close()
        if target is not None:
            target.close()


def _load_fixed_base(config, model, adapter, path, tmp_path, *, from_hf=False):
    config = copy.deepcopy(config)
    config.initial_load_path = str(path)
    config.initial_load_in_hf = from_hf
    config.keep_latest_k = 0
    config.async_mode = "disabled"
    manager = config.build(
        dataloader=None, model_parts=[model], optimizers=None, lr_schedulers=None,
        states={}, base_folder=str(tmp_path), sd_adapter=adapter,
    )
    try:
        assert manager.load()
    finally:
        manager.close()


def test_lora_parallelization_freezes_base_with_lora_substring(tiny_registry, monkeypatch, tmp_path):
    recipe = deepseek_v4_1_debugmodel_lora()
    model = build_cpu_model(recipe.model_spec.model)
    model.register_parameter("pre_lora_cache", torch.nn.Parameter(torch.ones(2)))

    _parallelize_cpu(model, monkeypatch, tmp_path, False)

    assert not model.pre_lora_cache.requires_grad
    assert model.get_parameter("layers.0.attention.wq_a.lora_a.weight").requires_grad


def test_recipe_loads_fixed_base_without_overwriting_adapters(tiny_registry, tmp_path):

    recipe = deepseek_v4_1_debugmodel_lora()
    assert isinstance(recipe.checkpoint, DeepSeekV41LoRACheckpointManager.Config)
    assert recipe.optimizer.name == "AdamW"
    assert recipe.checkpoint.enable and not recipe.checkpoint.last_save_model_only
    model = build_cpu_model(recipe.model_spec.model)
    model.register_parameter("pre_lora_cache", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    original = copy.deepcopy(model.state_dict())
    base = {name: torch.full_like(value, 0.25) for name, value in original.items() if not is_lora_parameter(name)}
    path = str(tmp_path / "base")
    dcp.save(base, checkpoint_id=path)
    adapter = recipe.model_spec.state_dict_adapter(recipe.model_spec.model, None)
    _load_fixed_base(recipe.checkpoint, model, adapter, path, tmp_path)
    torch.testing.assert_close(model.pre_lora_cache, base["pre_lora_cache"], rtol=0, atol=0)
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, original[name] if is_lora_parameter(name) else base[name], rtol=0, atol=0)


def test_peft_merge_reloads_with_frozen_engram(tiny_registry, monkeypatch, tmp_path):


    torch.manual_seed(12)
    options = DeepSeekV41LoRAConverter.Config(rank=8, alpha=16, rank_experts=8)
    spec = registry.model_registry("deepseek_v4_1_debugmodel_text", converters=[options])
    model = build_cpu_model(spec.model)
    _parallelize_cpu(model, monkeypatch, tmp_path, False)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if is_lora_parameter(name):
                parameter.normal_(std=0.01)
    adapter = spec.state_dict_adapter(spec.model, None)
    base_state = adapter.to_hf(model.state_dict())
    base, exported, output = (tmp_path / name for name in ("base", "adapter", "merged"))
    base.mkdir()
    exported.mkdir()
    save_file({name: value.contiguous().clone() for name, value in base_state.items()}, base / "model.safetensors")
    metadata = {"model_type": "deepseek_v4_1", "engram": {"hash_seed": 0}}
    (base / "config.json").write_text(json.dumps(metadata))
    (base / "token_id_map.json").write_text("[0, 1, 2]\n")
    save_file(adapter.to_peft(model.state_dict()), exported / "adapter_model.safetensors")
    (exported / "adapter_config.json").write_text(json.dumps(adapter.peft_adapter_config()))
    merge(str(base), str(exported), str(output))
    merged = load_file(output / "model.safetensors")
    for name, value in base_state.items():
        if ".engram." in name:
            torch.testing.assert_close(merged[name], value, rtol=0, atol=0)
    assert (output / "config.json").read_bytes() == (base / "config.json").read_bytes()
    assert (output / "token_id_map.json").read_bytes() == (base / "token_id_map.json").read_bytes()
    merged_model = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel_text").model)
    merged_model.load_state_dict(adapter.from_hf(merged))
    frozen_base = build_cpu_model(registry.model_registry("deepseek_v4_1_debugmodel_text").model)
    frozen_base.load_state_dict(adapter.from_hf(base_state))
    tokens = torch.arange(16).reshape(1, -1)
    positions = torch.arange(16).reshape(1, -1)
    with torch.no_grad():
        tokens_in, _, kwargs = preprocess(model, tokens, tokens, positions)
        expected = model(tokens_in, **kwargs)
        tokens_in, _, kwargs = preprocess(merged_model, tokens, tokens, positions)
        actual = merged_model(tokens_in, **kwargs)
        tokens_in, _, kwargs = preprocess(frozen_base, tokens, tokens, positions)
        unchanged = frozen_base(tokens_in, **kwargs)
    assert not torch.allclose(expected, unchanged, rtol=1e-4, atol=1e-6)
    assert (actual - expected).abs().max() < (unchanged - expected).abs().max()
    # Routed-expert GEMMs use BF16; merge rounds their base weights once.
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)


def test_muon_profile_covers_split_adapters_without_engram_optimizer(tiny_registry, monkeypatch, tmp_path):


    spec = registry.model_registry(
        "deepseek_v4_1_debugmodel_text", converters=[DeepSeekV41LoRAConverter.Config(rank=8, rank_experts=8)]
    )
    model = build_cpu_model(spec.model)
    profile = _v41_muon_profile(spec)
    layouts = profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]
    expected = {name for name, _ in model.named_parameters() if is_lora_parameter(name) and ".indexer." not in name and not name.startswith("lm_head.")}
    assert expected
    assert {name for name, _ in model.named_parameters() if is_lora_parameter(name) and re.search(profile.muon_pattern, name)} == expected
    assert expected <= layouts.keys()
    for name in expected:
        if ".inner_experts." in name:
            assert layouts[name].shard_order_by_tensor_dim[0] == ("ep", "efsdp")
    optimizer = _v41_optimizer_config(spec)
    optimizer.name = "Muon"
    optimizer.materialize()
    assert all(group.optimizer_name != "SparseAdam" for group in optimizer.param_groups)
    _parallelize_cpu(model, monkeypatch, tmp_path, False)
    assigned = set()

    def capture_muon(param_groups, **kwargs):
        for group in param_groups:
            assigned.update(group["param_names"])
        return torch.optim.AdamW(param_groups)

    monkeypatch.setattr("torchtitan.components.optimizer.optimizer.build_dist_muon", capture_muon)
    container = optimizer.build(model_parts=[model])
    assert assigned == expected
    actual = {id(p) for item in container.optimizers for group in item.param_groups for p in group["params"]}
    assert actual == {id(p) for name, p in model.named_parameters() if is_lora_parameter(name)}


@pytest.mark.parametrize("base_key", ["weight", "pre_lora_cache.weight"], ids=["plain-base", "lora-substring-base"])
def test_resume_rejects_checkpoint_without_adapters(tiny_registry, tmp_path, base_key):

    manager = object.__new__(DeepSeekV41LoRACheckpointManager)
    manager.initial_load_path = str(tmp_path / "base")
    resume = str(tmp_path / "resume")
    dcp.save({base_key: torch.ones(2)}, checkpoint_id=resume)
    with pytest.raises(ValueError, match="resume requires adapter tensors"):
        manager.dcp_load({base_key: torch.empty(2)}, resume)


@pytest.mark.parametrize("flavor", ["deepseek_v4_1_debugmodel_text", "deepseek_v4_1_debugmodel"], ids=["text", "multimodal"])
def test_hf_base_load_keeps_adapters_and_engram(tiny_registry, tmp_path, flavor):
    spec = registry.model_registry(
        flavor, converters=[DeepSeekV41LoRAConverter.Config(rank=8, rank_experts=8)]
    )
    model = build_cpu_model(spec.model)
    adapter = spec.state_dict_adapter(spec.model, None)
    base = tmp_path / "hf-base"
    base.mkdir()
    model.register_parameter("pre_lora_cache", torch.nn.Parameter(torch.full((2,), 7.0), requires_grad=False))
    expected_hf = {name: value.clone().contiguous() for name, value in adapter.to_hf(model.state_dict()).items()}
    torch.testing.assert_close(expected_hf["pre_lora_cache"], torch.full((2,), 7.0), rtol=0, atol=0)
    save_file(expected_hf, base / "model.safetensors")
    expected_base = adapter.from_hf(expected_hf)
    expected_vision = {
        name: value.clone() for name, value in model.state_dict().items()
        if name.startswith(("vision_encoder.", "image_marker_embeddings."))
    }
    assert bool(expected_vision) == (flavor == "deepseek_v4_1_debugmodel")
    expected_adapters = {name: value.clone() for name, value in model.state_dict().items() if is_lora_parameter(name)}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not is_lora_parameter(name):
                parameter.zero_()
    recipe = deepseek_v4_1_debugmodel_lora()
    _load_fixed_base(recipe.checkpoint, model, adapter, base, tmp_path, from_hf=True)
    state = model.state_dict()
    torch.testing.assert_close(state["pre_lora_cache"], torch.full((2,), 7.0), rtol=0, atol=0)
    for name, expected in {**expected_base, **expected_adapters, **expected_vision}.items():
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0)
