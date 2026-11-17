# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch import distributed as dist, multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan_npu.models.deepseek_v4 import lora, peft, state_dict_adapter
from tests.unit_tests.models.deepseek_v4.lora_test_utils import _adapter_with_converter


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def _make_checkpoint_fixture(tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base):
    class RoutingState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("expert_bias_E", torch.zeros(3))
            self.register_buffer("tokens_per_expert_E", torch.zeros(3), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)
            self.pre_lora_cache = torch.nn.Parameter(torch.full((2,), 11.0), requires_grad=False)
            if trainable_base:
                self.base.requires_grad_(True)
                self.frozen_extra = torch.nn.Parameter(torch.full((2,), 7.0), requires_grad=False)
            self.lora_a = torch.nn.Parameter(torch.ones(2, 3))
            self.lora_b = torch.nn.Parameter(torch.ones(3, 2))
            self.moe = checkpoint_wrapper(RoutingState()) if wrapper_scope == "submodule" else RoutingState()

    def build_model():
        model = Model()
        if wrapper_scope == "block-fp8":
            pytest.importorskip("torchao_npu")
            from torchtitan_npu.extensions.adaptor.torchao.converter import _block_fp8_param_swap
            from torchao_npu.wrapper_tensors.block_mx_wrapper_tensor import BlockMXTrainingWeightWrapperTensor

            policy = _block_fp8_param_swap()
            model.base = torch.nn.Parameter(
                BlockMXTrainingWeightWrapperTensor(model.base.detach(), policy.weight_config, policy.activation_config),
                requires_grad=False,
            )
        return checkpoint_wrapper(model) if wrapper_scope == "model" else model

    class StepState:
        def __init__(self, step):
            self.step = step

        def state_dict(self):
            return {"step": torch.tensor(self.step)}

        def load_state_dict(self, state):
            self.step = int(state["step"])

    def manager(model, step):
        result = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
        result.ema_optimizer = None
        result.stager = None
        result.verify_hash_manifest = False
        result.periodic_save_adapter_only = True
        result.save_training_state = save_training_state
        result.initial_load_path = str(tmp_path / "base")
        result.initial_load_in_hf = False
        result.initial_load_in_hf_quantized = False
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], foreach=False)
        if step or save_training_state or checkpoint_format == "full_model":
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, float(step))
            optimizer.step()
            optimizer.zero_grad()
        result.states = {"model": ModelWrapper(model), "optimizer": optimizer, "train_state": StepState(step)}
        return result

    return build_model, manager


def _floating_base_state(model):
    # Pretrained base weights are plain tensors, even when training uses wrappers.
    return {
        key: value.to_tensor() if hasattr(value, "to_tensor") else value
        for key, value in model.state_dict().items()
        if key not in {"lora_a", "lora_b"}
    }


def _check_periodic_dcp_round_trip(
    tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base=False
):
    build_model, manager = _make_checkpoint_fixture(
        tmp_path, save_training_state, checkpoint_format, wrapper_scope, trainable_base
    )
    original = build_model()
    source = manager(original, 9)
    dcp.save(
        _floating_base_state(original),
        checkpoint_id=source.initial_load_path,
    )
    with torch.no_grad():
        if trainable_base:
            original.get_parameter("base").fill_(5)
        original.get_parameter("lora_a").fill_(2)
        original.get_parameter("lora_b").fill_(3)
        original.state_dict()["moe.expert_bias_E"].copy_(torch.tensor([-0.1, 0.2, -0.1]))
    selected = source._flattened_model_states_sd()
    expected_keys = {"lora_a", "lora_b", "moe.expert_bias_E"}
    if trainable_base:
        expected_keys.add("base")
    if save_training_state:
        expected_keys |= {"optimizer", "train_state"}
    assert set(selected) == expected_keys
    if checkpoint_format == "legacy_adapter":
        selected.pop("moe.expert_bias_E")
    elif checkpoint_format == "full_model":
        selected.update(original.state_dict())
        selected.update({key: value for key, value in source.states.items() if key != "model"})
    checkpoint = str(tmp_path / "step-9")
    dcp.save(selected, checkpoint_id=checkpoint)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    assert ("base" in metadata.state_dict_metadata) == (trainable_base or checkpoint_format == "full_model")
    assert "moe.tokens_per_expert_E" not in metadata.state_dict_metadata

    restored = build_model()
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    target = manager(restored, 0)
    _load_base_then_adapter(target, restored, checkpoint)
    assert type(restored.get_parameter("base")) is type(original.get_parameter("base"))

    for key, tensor in original.state_dict().items():
        expected = (
            torch.zeros_like(tensor) if key == "moe.expert_bias_E" and checkpoint_format == "legacy_adapter" else tensor
        )
        assert torch.equal(expected, restored.state_dict()[key]), key
    _check_restored_training_state((original, restored), (source, target), save_training_state, checkpoint_format)


def _load_base_then_adapter(target, restored, checkpoint):
    initial_adapters = {key: restored.state_dict()[key].clone() for key in ("lora_a", "lora_b")}
    target.dcp_load(restored.state_dict(), target.initial_load_path)
    for key, value in initial_adapters.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        restored.get_parameter("base").zero_()
    target.dcp_load(target._flattened_model_states_sd(target.states), checkpoint)


def _check_restored_training_state(models, managers, save_training_state, checkpoint_format):
    original, restored = models
    source, target = managers
    restores_training_state = save_training_state or checkpoint_format == "full_model"
    assert target.states["train_state"].step == (9 if restores_training_state else 0)
    restored_optimizer = target.states["optimizer"].state_dict()["state"]
    if restores_training_state:
        for parameter_id, state in source.states["optimizer"].state_dict()["state"].items():
            for key, tensor in state.items():
                assert torch.equal(tensor, restored_optimizer[parameter_id][key])
        for model, checkpoint_manager in ((original, source), (restored, target)):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, 0.25)
            checkpoint_manager.states["optimizer"].step()
        for expected, actual in zip(original.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert not restored_optimizer


@pytest.mark.parametrize(
    "save_training_state,checkpoint_format,wrapper_scope",
    [
        (False, "adapter_buffers", "plain"),
        (True, "adapter_buffers", "plain"),
        (True, "adapter_buffers", "block-fp8"),
        (False, "legacy_adapter", "plain"),
        (True, "legacy_adapter", "plain"),
        (False, "full_model", "plain"),
        (True, "full_model", "plain"),
        (False, "adapter_buffers", "model"),
        (False, "adapter_buffers", "submodule"),
    ],
)
def test_periodic_dcp_round_trip_keeps_base_out_and_restores_selected_state(
    tmp_path, save_training_state, checkpoint_format, wrapper_scope
):
    _check_periodic_dcp_round_trip(tmp_path, save_training_state, checkpoint_format, wrapper_scope)


@pytest.mark.parametrize("wrapper_scope", ["plain", "model"], ids=["plain", "checkpoint-wrapped"])
def test_periodic_dcp_restores_reactivated_base_weights_and_optimizer(tmp_path, wrapper_scope):
    _check_periodic_dcp_round_trip(tmp_path, True, "adapter_buffers", wrapper_scope, trainable_base=True)


def _distributed_worker(rank, root, action, argument):
    dist.init_process_group(
        "gloo", init_method=(Path(root) / "store").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        message = None
        try:
            action(rank, root, argument)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
        Path(root, f"rank{rank}.json").write_text(json.dumps(message))
    finally:
        dist.destroy_process_group()


def _run_distributed(tmp_path, action, argument):
    mp.spawn(_distributed_worker, args=(str(tmp_path), action, argument), nprocs=2, join=True)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]


def _check_export_guard(rank, root, invalid_rank):
    model = torch.nn.Module()
    model.register_parameter("base", torch.nn.Parameter(torch.ones(2), requires_grad=rank == invalid_rank))
    model.register_parameter("lora_a", torch.nn.Parameter(torch.ones(2)))
    manager = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
    manager.states = {"model": SimpleNamespace(model=[model])}
    manager._ensure_peft_exportable()


@pytest.mark.parametrize("invalid_rank", [None, 1], ids=["adapter-only", "remote-trainable-base"])
def test_peft_export_guard_agrees_across_ranks(tmp_path, invalid_rank):
    messages = _run_distributed(tmp_path, _check_export_guard, invalid_rank)
    if invalid_rank is None:
        assert messages == [None, None]
    else:
        assert all(message and message.startswith("ValueError:") for message in messages)
        assert all(message and "trainable non-LoRA parameters: base" in message for message in messages)
        assert messages[0] == messages[1]


class _MappingStubAdapter(state_dict_adapter.DeepSeekV4StateDictAdapter):
    def __init__(self, fail_finalization):
        self.fail_finalization = fail_finalization

    def to_peft(self, state_dict):
        return state_dict

    def peft_adapter_config(self, **kwargs):
        return {"peft_type": "LORA", "fail_finalization": self.fail_finalization}


class _FinalSaveCheckpoint(peft.DeepSeekV4PEFTCheckpointManager):
    @staticmethod
    def _finalize_peft_directory(checkpoint_id, adapter_config, state_dict):
        if adapter_config["fail_finalization"]:
            raise OSError("injected finalization failure")
        peft.DeepSeekV4PEFTCheckpointManager._finalize_peft_directory(checkpoint_id, adapter_config, state_dict)

    def _create_checkpoint_id(self, curr_step):
        return self.checkpoint_dir


def _save_peft(rank, root, fail_finalization):
    model = torch.nn.Module()
    mesh = init_device_mesh("cpu", (2,))
    adapter = distribute_tensor(torch.arange(6).reshape(2, 3).float(), mesh, [Shard(0)])
    model.register_parameter("lora_a", torch.nn.Parameter(adapter))
    manager = object.__new__(_FinalSaveCheckpoint)
    manager.states = {"model": ModelWrapper(model)}
    manager.last_save_in_peft = True
    manager.peft_base_model_name_or_path = None
    manager.sd_adapter = _MappingStubAdapter(fail_finalization)
    manager.initial_load_path = ""
    manager.initial_load_in_hf = False
    manager.export_dtype = torch.float32
    manager.checkpoint_dir = str(Path(root) / "checkpoint")
    def unexpected_dcp_save(*args, **kwargs):
        raise AssertionError("PEFT export must not use DCP save")

    original_save = dcp.save
    try:
        dcp.save = unexpected_dcp_save
        manager._save_last_step(1)
    finally:
        dcp.save = original_save


@pytest.mark.parametrize("fail_finalization", [False, True], ids=["adapter-file", "rank-zero-io-failure"])
def test_final_peft_save_finishes_consistently_across_ranks(tmp_path, fail_finalization):
    messages = _run_distributed(tmp_path, _save_peft, fail_finalization)
    if fail_finalization:
        assert all(
            message and "PEFT checkpoint finalization failed: OSError: injected" in message for message in messages
        )
        assert messages[0] == messages[1]
    else:
        assert messages == [None, None]
        saved = load_file(tmp_path / "checkpoint" / "adapter_model.safetensors")
        torch.testing.assert_close(saved["lora_a"], torch.arange(6).reshape(2, 3).float(), rtol=0, atol=0)


@pytest.mark.parametrize("peft_export", [False, True])
def test_hf_export_rejected_before_checkpoint_initialization(monkeypatch, peft_export):
    def unexpected_init(self, config, **kwargs):
        pytest.fail("Invalid export configuration reached checkpoint initialization")
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", unexpected_init)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(
        enable=True, last_save_in_peft=peft_export, last_save_in_hf=True,
    )
    with pytest.raises(ValueError, match="HF|last_save_in_hf"):
        config.build()


@pytest.mark.parametrize("invalid", ["mtp", "rank"])
def test_peft_configuration_rejected_before_checkpoint_initialization(monkeypatch, invalid):
    converter = lora.DeepSeekV4LoRAConverter.Config(
        target_modules=["attention.wq_a"], include_mtp=invalid == "mtp",
        rank=2, rank_experts=3 if invalid == "rank" else 2,
    )
    adapter = _adapter_with_converter(converter)
    def unexpected_init(self, config, **kwargs):
        pytest.fail("Invalid export configuration reached checkpoint initialization")
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", unexpected_init)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(enable=True)
    with pytest.raises((ValueError, NotImplementedError), match="MTP|rank_experts|Unsupported"):
        config.build(sd_adapter=adapter)


def test_native_checkpoint_allows_mtp_and_mixed_ranks(monkeypatch):
    adapter = _adapter_with_converter(lora.DeepSeekV4LoRAConverter.Config(
        rank=2, rank_experts=3, target_modules=["attention.wq_a"],
    ))
    def initialize(self, config, **kwargs):
        self.initial_load_path = "/base"
    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", initialize)
    config = peft.DeepSeekV4PEFTCheckpointManager.Config(enable=True, last_save_in_peft=False)
    config.build(sd_adapter=adapter)


def test_hf_base_load_limits_expert_reassembly_to_one_layer(monkeypatch):
    calls = []

    def load(self, state, checkpoint_id, from_hf=False, from_quantized=False):
        assert checkpoint_id == "/base"
        assert from_hf and from_quantized
        calls.append(dict(state))

    monkeypatch.setattr(peft.NPUCheckpointManager, "dcp_load", load)
    manager = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
    state = {name: torch.ones(1) for name in (
        "tok_embeddings.weight", "layers.0.attention.wq_a.weight",
        "pre_lora_cache.weight",
        "layers.0.moe.routed_experts.inner_experts.w1_EFD",
        "layers.1.moe.routed_experts.inner_experts.w1_EFD",
        "mtp_layers.0.attention.wq_a.weight", "layers.0.attention.wq_a.lora_a.weight",
    )}
    manager.dcp_load(state, "/base", from_hf=True, from_quantized=True)
    expected = set(state) - {"layers.0.attention.wq_a.lora_a.weight"}
    assert len(calls) == 4
    assert {name for call in calls for name in call} == expected
    assert sum(len(call) for call in calls) == len(expected)
    for call in calls:
        assert len({".".join(name.split(".")[:2]) for name in call if "layers." in name}) <= 1
        assert all(value is state[name] for name, value in call.items())


@pytest.mark.parametrize("folder,base_folder", [("s3://bucket/run", ""), ("checkpoint", "s3://bucket/run")])
def test_remote_peft_destination_rejected_before_checkpoint_initialization(monkeypatch, folder, base_folder):
    def unexpected_init(*args, **kwargs):
        pytest.fail("Remote PEFT destination reached checkpoint initialization")

    monkeypatch.setattr(peft.NPUCheckpointManager, "__init__", unexpected_init)
    with pytest.raises(ValueError, match="local checkpoint folder"):
        config = peft.DeepSeekV4PEFTCheckpointManager.Config(enable=True, folder=folder)
        config.build(base_folder=base_folder)


@pytest.mark.parametrize("from_hf,explicit,expected", [
    (False, None, None), (True, None, "/base"), (False, "org/base", "org/base"),
])
def test_peft_metadata_uses_hf_base_and_never_native_checkpoint(monkeypatch, from_hf, explicit, expected):
    manager = object.__new__(_FinalSaveCheckpoint)
    model = torch.nn.Module()
    model.register_parameter("lora_a", torch.nn.Parameter(torch.ones(2)))
    manager.states = {"model": ModelWrapper(model)}
    manager.last_save_in_peft = True
    manager.peft_base_model_name_or_path = explicit
    manager.initial_load_path = "/base"
    manager.initial_load_in_hf = from_hf
    manager.export_dtype = torch.float32
    manager.sd_adapter = _MappingStubAdapter(False)
    captured = {}
    monkeypatch.setattr(manager.sd_adapter, "peft_adapter_config", lambda **kwargs: captured.update(kwargs) or {})
    monkeypatch.setattr(peft.dcp, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "finalize_peft_checkpoint", lambda *args: None)
    manager.checkpoint_dir = "/output"
    manager._save_last_step(1)
    assert captured["base_model_name_or_path"] == expected


def test_adapter_state_selection_keeps_trainable_state_and_routing_buffers():
    state = {name: torch.ones(1) for name in (
        "base.weight", "pre_lora_cache.weight", "lora_a", "layers.0.w13_lora_a",
        "layers.0.w1_lora_b", "layers.0.w3_lora_b", "layers.0.w2_lora_b",
        "expert_bias_E", "optimizer", "unfrozen",
    )}
    selected = peft.select_adapter_only_state(
        state, {"optimizer"}, model_buffer_keys={"expert_bias_E"}, model_trainable_keys={"unfrozen"},
    )
    assert set(selected) == {
        "lora_a", "layers.0.w13_lora_a", "layers.0.w1_lora_b", "layers.0.w3_lora_b",
        "layers.0.w2_lora_b", "expert_bias_E", "optimizer", "unfrozen",
    }


def test_resume_rejects_legacy_interleaved_expert_optimizer(tmp_path):
    checkpoint = tmp_path / "legacy"
    dcp.save({"layers.0.moe.routed_experts.inner_experts.w13_lora_b": torch.ones(2, 3, 2, 2)}, checkpoint_id=checkpoint)
    manager = object.__new__(peft.DeepSeekV4PEFTCheckpointManager)
    manager.initial_load_path = None

    with pytest.raises(ValueError, match="former AdamW state"):
        manager.dcp_load({}, str(checkpoint))
