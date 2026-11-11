# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Current checkpoints through real managers, adapters, DCP and Stateful consumers."""

import copy
import json
from contextlib import closing
from dataclasses import dataclass

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torchtitan.components.checkpointer import EMA as EMA_KEY
from torchtitan.components.optim import EMA, AdamW, LRSchedulersContainer, OptimizersContainer

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config
from torchtitan_npu.extensions.components import validation
from torchtitan_npu.extensions.torchtitan.components import checkpoint as checkpoint_extension
from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform, is_lora_parameter
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter


@dataclass
class _Counter:
    step: int = 0

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


def _model_config(monkeypatch, family):
    if family == "v4":
        config = _build_model_config(num_layers=2, num_experts=2)
    else:
        from torchtitan_npu.models import deepseek_v4_1 as registry

        make = registry._make_v41_config

        def tiny(**kwargs):
            kwargs.update(
                n_layers=2,
                vocab_size=64,
                compress_ratios=(0, 0),
                kv_source_layers=(0,),
                index_source_layers=(0,),
                candidate_source_layer=0,
                dim=8,
                n_heads=2,
                head_dim=8,
                rope_head_dim=4,
                q_lora_rank=8,
                o_lora_rank=4,
                n_groups=1,
                index_n_heads=2,
                index_head_dim=4,
                index_topk=2,
                moe_inter_dim=16,
                num_experts=2,
                top_k=1,
                hc_mult=2,
                sinkhorn_iters=2,
                max_seq_len=32,
                vision_dim=8,
                vision_heads=2,
                vision_inter_dim=16,
            )
            return make(**kwargs)

        monkeypatch.setattr(registry, "_make_v41_config", tiny)
        config = registry.model_registry("deepseek_v4_1_stripped")
        for layer in config.layers:
            layer.engram = None
    return deepseek_v4_lora_transform(rank=8, alpha=16).transform(config)


def _model(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = copy.deepcopy(config).build()
        model.init_states(buffer_device=torch.device("cpu"))
    model.register_buffer("resume_probe", torch.tensor([0.0, 1.0]))
    # Current adapter-only saves also preserve explicitly trainable base tensors.
    model.norm.weight.requires_grad_(True)
    return model


def _adapter(config, family):
    owner = DeepSeekV4StateDictAdapter if family == "v4" else DeepSeekV41StateDictAdapter
    return owner(config, None)


def _write_hf_base(path, model, adapter):
    path.mkdir()
    state = {key: value for key, value in model.state_dict().items() if not is_lora_parameter(key)}
    hf_state = {key: value.detach().clone().contiguous() for key, value in adapter.to_hf(state).items()}
    save_file(hf_state, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in hf_state.values())},
                "weight_map": dict.fromkeys(hf_state, "model.safetensors"),
            }
        )
    )
    validation._write_manifest(path)


def _write_native_base(path, model):
    dcp.save(
        {key: value for key, value in model.state_dict().items() if not is_lora_parameter(key)}, checkpoint_id=path
    )
    validation._write_manifest(path)


def _manager(path, base, config, family, model, *, save_training_state=True, verify=False, initial_hf=True, keep=0):
    optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=".*", lr=0.02, weight_decay=0.0, fused=False)]
    ).build(model_parts=[model])
    scheduler = LRSchedulersContainer.Config(warmup_steps=0).build(optimizers=optimizer, training_steps=20)
    ema = EMA.Config(decay=0.5, buffer_patterns=[r"^resume_probe$"]).build(model_parts=[model])
    owner = DeepSeekV4PEFTCheckpointManager if family == "v4" else DeepSeekV41LoRACheckpointManager
    manager = owner.Config(
        folder=str(path),
        interval=1,
        keep_latest_k=keep,
        initial_load_path=str(base),
        initial_load_in_hf=initial_hf,
        last_save_in_peft=False,
        save_training_state=save_training_state,
        extensions=checkpoint_extension.CheckpointExtensions(verify_hash_manifest=verify),
    ).build(
        model_parts=[model],
        optimizers=optimizer,
        lr_schedulers=scheduler,
        dataloader=_Counter(),
        ema=ema,
        states={"train_state": _Counter()},
        sd_adapter=_adapter(config, family),
    )
    return manager


def test_peft_retention_preserves_resume_point_and_uses_normal_cleanup(tmp_path, monkeypatch):
    config = _model_config(monkeypatch, "v4")
    root = tmp_path / "checkpoints"
    base = tmp_path / "base"
    _write_native_base(base, _model(config))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def new_manager():
        return _manager(root, base, config, "v4", _model(config), verify=True, initial_hf=False, keep=2)

    def retained():
        return {path.name for path in root.iterdir()}

    with closing(new_manager()) as manager:
        manager.states["train_state"].step = manager.states["dataloader"].step = 1
        assert manager.save(1)
        weights_only = root / "step-3"
        adapters = {
            name: value for name, value in manager.states["model"].state_dict().items() if is_lora_parameter(name)
        }
        dcp.save(adapters, checkpoint_id=weights_only)
        validation._write_manifest(weights_only)
        manager._finalize_peft_directory(str(root / "step-4"), {"peft_type": "LORA"}, {"a": torch.ones(1)})
        assert manager._find_load_step() == 1
    assert retained() == {"step-1", "step-3", "step-4"}

    with closing(new_manager()) as manager:
        assert manager.load()
        assert manager.states["train_state"].step == manager.states["dataloader"].step == 1
        native_save = manager.dcp_save

        def interrupted_save(*args, **kwargs):
            native_save(*args, **kwargs)
            raise OSError("injected failure before manifest completion")

        with monkeypatch.context() as patch:
            patch.setattr(manager, "dcp_save", interrupted_save)
            with pytest.raises(OSError, match="before manifest completion"):
                manager.save(5)
        assert (root / "step-5" / ".metadata").is_file()
        assert not validation.is_checkpoint_manifest_complete(root / "step-5", required=False)
        assert manager._find_load_step() == 1
    assert retained() == {"step-1", "step-4", "step-5"}

    with closing(new_manager()) as manager:
        manager.states["train_state"].step = manager.states["dataloader"].step = 5
        assert manager.save(5)
        assert validation.is_checkpoint_manifest_complete(root / "step-5", required=True)
        assert manager._find_load_step() == 5
        assert manager._find_load_step(max_step=4) == 1
    assert retained() == {"step-1", "step-4", "step-5"}

    with closing(new_manager()) as manager:
        assert manager.load()
        assert manager.states["train_state"].step == manager.states["dataloader"].step == 5
        assert manager.save(6)
        assert manager._find_load_step() == 6
    assert retained() == {"step-5", "step-6"}


def test_peft_retention_preserves_native_resume_when_newer_dcp_lacks_adapters(tmp_path, monkeypatch):
    config = _model_config(monkeypatch, "v4")
    root = tmp_path / "checkpoints"
    base = tmp_path / "base"
    _write_native_base(base, _model(config))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    with closing(_manager(root, base, config, "v4", _model(config), initial_hf=False, keep=2)) as manager:
        manager.states["train_state"].step = manager.states["dataloader"].step = 1
        assert manager.save(1)
        saved = manager._flattened_model_states_sd()
        dcp.save(
            {name: value for name, value in saved.items() if not is_lora_parameter(name)}, checkpoint_id=root / "step-2"
        )

        assert manager.load()
        assert manager.states["train_state"].step == 1
        with pytest.raises(ValueError, match="requires adapter tensors"):
            manager.load(step=2)

        manager._purge_stale_checkpoints(saving_step=3)
        manager._finalize_peft_directory(str(root / "step-3"), {"peft_type": "LORA"}, {"a": torch.ones(1)})

    assert {path.name for path in root.iterdir()} == {"step-1", "step-2", "step-3"}


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("load_step", [-1, 0])
def test_native_step_zero_seed_initializes_base_without_overwriting_adapters(tmp_path, monkeypatch, family, load_step):
    config = _model_config(monkeypatch, family)
    root = tmp_path / "checkpoints"
    seed = root / "step-0"
    original = _model(config)
    _write_native_base(seed, original)
    expected = {name: value for name, value in _clone_state(original).items() if not is_lora_parameter(name)}
    model = _model(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.75)
        model.resume_probe.zero_()
    adapters = {name: value for name, value in _clone_state(model).items() if is_lora_parameter(name)}

    with closing(_manager(root, seed, config, family, model, initial_hf=False, verify=True)) as manager:
        assert manager.load(step=load_step)

    actual = model.state_dict()
    _assert_state({name: actual[name] for name in expected}, expected)
    _assert_state({name: actual[name] for name in adapters}, adapters)


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("keep,full_save", [(2, False), (3, False), (2, True)])
def test_adapter_only_retention_keeps_initial_seed_and_resumes_latest_state(
    tmp_path, monkeypatch, family, keep, full_save
):
    config = _model_config(monkeypatch, family)
    root = tmp_path / "checkpoints"
    seed = root / "step-0"
    _write_native_base(seed, _model(config))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    with closing(_manager(root, seed, config, family, _model(config), initial_hf=False, keep=keep)) as manager:
        assert manager.load(step=0)
        for step in (1, 2, 3):
            if step == 3 and full_save:
                manager.periodic_save_adapter_only = False
            with torch.no_grad():
                for parameter in manager.states["model"].model[0].parameters():
                    if parameter.requires_grad:
                        parameter.add_(0.25)
            manager.states["train_state"].step = manager.states["dataloader"].step = step
            assert manager.save(step)
        expected = _clone_state(manager.states["model"].model[0])

    assert (seed / ".metadata").is_file()
    with closing(_manager(root, seed, config, family, _model(config), initial_hf=False, keep=keep)) as manager:
        assert manager._find_load_step() == 3
        assert manager.load()
        assert manager.states["train_state"].step == manager.states["dataloader"].step == 3
        _assert_state(manager.states["model"].state_dict(), expected)


@pytest.mark.parametrize("family", ["v4", "v41"])
def test_automatic_selection_rejects_step_zero_without_adapters_unless_it_is_initial_base(
    tmp_path, monkeypatch, family
):
    config = _model_config(monkeypatch, family)
    model = _model(config)
    root = tmp_path / "checkpoints"
    base = tmp_path / "base"
    _write_native_base(base, model)
    _write_native_base(root / "step-0", model)

    with closing(_manager(root, base, config, family, model, initial_hf=False)) as manager:
        with pytest.raises(ValueError, match="requires adapter tensors"):
            manager.load()


def _clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value, key


def _record_ema(monkeypatch, manager):
    calls = []
    original = manager.states[EMA_KEY].load_state_dict

    def restore(state):
        calls.append(bool(state))
        return original(state)

    monkeypatch.setattr(manager.states[EMA_KEY], "load_state_dict", restore)
    return calls


@pytest.mark.parametrize("family", ["v4", "v41"])
def test_real_load_entry_reads_hf_by_layer_preserves_adapters_and_reseeds_ema_once(tmp_path, monkeypatch, family):
    config = _model_config(monkeypatch, family)
    original = _model(config)
    expected = _clone_state(original)
    base = tmp_path / "base"
    _write_hf_base(base, original, _adapter(config, family))
    model = _model(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.fill_(0.75 if is_lora_parameter(name) else 0)
        model.resume_probe.zero_()
    adapters = {key: value.clone() for key, value in model.state_dict().items() if is_lora_parameter(key)}
    manager = _manager(tmp_path / "run", base, config, family, model, verify=True)
    calls, events = [], []
    to_hf = manager.sd_adapter.to_hf

    def convert(state):
        events.append("read")
        calls.append(tuple(state))
        return to_hf(state)

    verify = checkpoint_extension.verify_checkpoint_manifest

    def verify_source(path):
        events.append(("verify", str(path)))
        return verify(path)

    monkeypatch.setattr(manager.sd_adapter, "to_hf", convert)
    monkeypatch.setattr(checkpoint_extension, "verify_checkpoint_manifest", verify_source)
    ema_calls = _record_ema(monkeypatch, manager)
    try:
        assert manager.load()
        assert len(calls) == 3
        assert events[0] == ("verify", str(base))
        assert events.count(("verify", str(base))) == 1
        for group in calls:
            assert not any(is_lora_parameter(key) for key in group)
            assert len({".".join(key.split(".")[:2]) for key in group if key.startswith("layers.")}) <= 1
        actual = model.state_dict()
        _assert_state(actual, {key: adapters.get(key, value) for key, value in expected.items()})
        assert ema_calls == [False]
        for optimizer in manager.states[EMA_KEY].optimizers:
            for tensor, state in optimizer.state.items():
                torch.testing.assert_close(state["ema_params"], tensor, rtol=0, atol=0)
    finally:
        manager.close()


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("save_training_state", [False, True])
@pytest.mark.parametrize("initial_hf", [False, True], ids=["native-base", "hf-base"])
def test_current_adapter_only_save_load_restores_real_stateful_consumers(
    tmp_path, monkeypatch, family, save_training_state, initial_hf
):
    config = _model_config(monkeypatch, family)
    model = _model(config)
    base = tmp_path / "base"
    if initial_hf:
        _write_hf_base(base, model, _adapter(config, family))
    else:
        _write_native_base(base, model)
    source = _manager(
        tmp_path / "run",
        base,
        config,
        family,
        model,
        save_training_state=save_training_state,
        verify=True,
        initial_hf=initial_hf,
    )
    try:
        assert source.load()
        before_update = _clone_state(model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.25)
        source.states["optimizer"].step()
        source.states["optimizer"].zero_grad()
        source.states["lr_scheduler"].step()
        source.states["train_state"].step = 7
        source.states["dataloader"].step = 11
        model.resume_probe.add_(5)
        source.states[EMA_KEY].step(1)
        expected_model = _clone_state(model)
        assert any(
            not torch.equal(expected_model[name], before_update[name])
            for name in expected_model
            if is_lora_parameter(name)
        )
        expected_optimizer = copy.deepcopy(source.states["optimizer"].state_dict())
        expected_scheduler = copy.deepcopy(source.states["lr_scheduler"].state_dict())
        expected_ema = copy.deepcopy(source.states[EMA_KEY].state_dict())
        assert source.save(7)
        saved = set(dcp.FileSystemReader(tmp_path / "run" / "step-7").read_metadata().state_dict_metadata)
        assert "resume_probe" in saved and "norm.weight" in saved
        assert "tok_embeddings.weight" not in saved
        assert any(is_lora_parameter(key) for key in saved)
    finally:
        source.close()

    restored = _model(config)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
        restored.resume_probe.zero_()
    target = _manager(
        tmp_path / "run",
        base,
        config,
        family,
        restored,
        save_training_state=save_training_state,
        verify=True,
        initial_hf=initial_hf,
    )
    fresh_optimizer = copy.deepcopy(target.states["optimizer"].state_dict())
    fresh_scheduler = copy.deepcopy(target.states["lr_scheduler"].state_dict())
    ema_calls = _record_ema(monkeypatch, target)
    events = []
    verify = checkpoint_extension.verify_checkpoint_manifest

    def verify_source(path):
        events.append(str(path))
        return verify(path)

    monkeypatch.setattr(checkpoint_extension, "verify_checkpoint_manifest", verify_source)
    try:
        assert target.load()
        _assert_state(restored.state_dict(), expected_model)
        assert events == [str(tmp_path / "run" / "step-7"), str(base)]
        assert target.states["train_state"].step == (7 if save_training_state else 0)
        assert target.states["dataloader"].step == (11 if save_training_state else 0)
        if save_training_state:
            _assert_state(target.states["optimizer"].state_dict(), expected_optimizer)
            _assert_state(target.states["lr_scheduler"].state_dict(), expected_scheduler)
            _assert_state(target.states[EMA_KEY].state_dict(), expected_ema)
            assert ema_calls == [True]
        else:
            _assert_state(target.states["optimizer"].state_dict(), fresh_optimizer)
            _assert_state(target.states["lr_scheduler"].state_dict(), fresh_scheduler)
            assert ema_calls == [False]
            for optimizer in target.states[EMA_KEY].optimizers:
                for tensor, state in optimizer.state.items():
                    torch.testing.assert_close(state["ema_params"], tensor, rtol=0, atol=0)
    finally:
        target.close()


@pytest.mark.parametrize("corrupt", ["base", "adapter"])
def test_real_load_rejects_either_bad_manifest_before_any_model_restore(tmp_path, monkeypatch, corrupt):
    config = _model_config(monkeypatch, "v4")
    model = _model(config)
    base = tmp_path / "base"
    _write_hf_base(base, model, _adapter(config, "v4"))
    manager = _manager(tmp_path / "run", base, config, "v4", model, verify=True)
    try:
        assert manager.save(3)
        folder = base if corrupt == "base" else tmp_path / "run" / "step-3"
        artifact = next(path for path in folder.iterdir() if path.suffix in {".safetensors", ".distcp"})
        with artifact.open("ab") as output:
            output.write(b"corruption")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(-7)
            model.resume_probe.fill_(-9)
        before = _clone_state(model)
        with pytest.raises(validation.CheckpointManifestError, match="hash mismatch"):
            manager.load()
        _assert_state(model.state_dict(), before)
    finally:
        manager.close()


def test_real_hf_entry_forwards_quantized_flag_to_reader_boundary(tmp_path, monkeypatch):
    config = _model_config(monkeypatch, "v4")
    model = _model(config)
    base = tmp_path / "base"
    _write_hf_base(base, model, _adapter(config, "v4"))
    manager = _manager(tmp_path / "run", base, config, "v4", model)
    manager.initial_load_in_hf_quantized = True
    flags = []
    reader = manager.sd_adapter.get_hf_storage_reader

    def read(path, from_quantized=False):
        flags.append(from_quantized)
        # Fixture weights are plain safetensors. This tests flag forwarding,
        # while the real reader still restores all tensors; no quantization claim.
        return reader(path, False)

    monkeypatch.setattr(manager.sd_adapter, "get_hf_storage_reader", read)
    try:
        assert manager.load()
        assert flags == [True, True, True]
    finally:
        manager.close()


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("missing", ["all", "one"])
def test_real_resume_rejects_missing_adapters_before_loading_any_live_state(tmp_path, monkeypatch, family, missing):
    config = _model_config(monkeypatch, family)
    model = _model(config)
    run = tmp_path / "run"
    saved = _clone_state(model)
    adapter_names = sorted(name for name in saved if is_lora_parameter(name))
    assert len(adapter_names) > 1
    for name in adapter_names if missing == "all" else adapter_names[:1]:
        del saved[name]
    # A complete base must not make an incomplete adapter resume acceptable.
    saved = {name: value + 3 for name, value in saved.items()}
    dcp.save(saved, checkpoint_id=run / "step-3")
    manager = _manager(run, tmp_path / "base", config, family, model)
    before = _clone_state(model)
    ema_before = copy.deepcopy(manager.states[EMA_KEY].state_dict())
    loads = []
    original_load = dcp.load

    def record_load(*args, **kwargs):
        loads.append(kwargs.get("checkpoint_id"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(dcp, "load", record_load)
    try:
        with pytest.raises(ValueError, match="requires adapter tensors"):
            manager.load()
        assert loads == []
        _assert_state(model.state_dict(), before)
        _assert_state(manager.states[EMA_KEY].state_dict(), ema_before)
        assert manager.states["train_state"].step == manager.states["dataloader"].step == 0
    finally:
        manager.close()


def test_native_no_model_load_restores_stateful_only_without_reseeding_ema(tmp_path, monkeypatch):
    config = _model_config(monkeypatch, "v4")
    model = _model(config)
    manager = _manager(tmp_path / "run", tmp_path / "base", config, "v4", model)
    folder = tmp_path / "dataloader"
    dcp.save({"dataloader": _Counter(13)}, checkpoint_id=folder)
    before = _clone_state(model)
    ema_calls = _record_ema(monkeypatch, manager)
    try:
        manager._load_checkpoint(
            {"dataloader": manager.states["dataloader"]}, str(folder), from_hf=False, from_quantized=False
        )
        assert manager.states["dataloader"].step == 13
        assert ema_calls == []
        _assert_state(model.state_dict(), before)
    finally:
        manager.close()
