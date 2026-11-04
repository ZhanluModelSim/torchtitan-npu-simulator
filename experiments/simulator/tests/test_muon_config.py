# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public optimizer selection must reach the native model-owned policy."""

import re
from dataclasses import asdict

import pytest
from torchtitan_simulator.cli import main
from torchtitan_simulator.config import SimulationConfig, prepare_config
from torchtitan_simulator.model_overrides import ModelOverrides


@pytest.mark.parametrize("steps", [None, 3], ids=["recipe-default", "explicit-iterations"])
def test_dsv4_muon_selection_materializes_expert_and_fallback_groups(steps):
    prepared = prepare_config(SimulationConfig(optimizer_name="Muon", muon_ns_steps=steps))
    muon, fallback = prepared.trainer.optimizer.param_groups

    assert muon.optimizer_name == "DistMuon"
    assert muon.optimizer_kwargs["ns_steps"] == (10 if steps is None else 3)
    assert muon.optimizer_kwargs["fused"] is False
    assert muon.optimizer_kwargs["foreach"] is False
    for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
        assert re.search(muon.pattern, f"layers.0.moe.routed_experts.inner_experts.{name}")
    assert not re.search(muon.pattern, "tok_embeddings.weight")
    assert fallback.optimizer_name == "AdamW" and fallback.pattern == r".*"
    assert fallback.optimizer_kwargs["fused"] is True


def test_muon_geometry_overrides_rebuild_compute_layout_before_materialization():
    prepared = prepare_config(SimulationConfig(optimizer_name="Muon", model_overrides=ModelOverrides(n_layers=2)))
    layouts = prepared.trainer.optimizer.optimizer_factory_kwargs_by_name["DistMuon"]["compute_sharding_by_fqn"]

    assert "layers.1.moe.routed_experts.inner_experts.w1_EFD" in layouts
    assert not any(name.startswith("layers.2.") for name in layouts)


@pytest.mark.parametrize("preset", ["deepseek_v4_smoketest", "kimi_k3_smoketest"], ids=["dsv4", "kimi"])
def test_explicit_adamw_preserves_recipe_groups(preset):
    default = prepare_config(SimulationConfig(config=preset))
    selected = prepare_config(SimulationConfig(config=preset, optimizer_name="AdamW"))

    assert [asdict(group) for group in selected.trainer.optimizer.param_groups] == [
        asdict(group) for group in default.trainer.optimizer.param_groups
    ]
    assert all(group.optimizer_name == "AdamW" for group in selected.trainer.optimizer.param_groups)


def test_cli_muon_options_reach_native_config(monkeypatch):
    from torchtitan_simulator import runner

    # Isolate the expensive capture; config selection and materialization remain real.
    monkeypatch.setattr(runner, "simulate", prepare_config)
    prepared = main(["--optimizer.name", "Muon", "--optimizer.muon-ns-steps", "7"])

    assert prepared.trainer.optimizer.param_groups[0].optimizer_name == "DistMuon"
    assert prepared.trainer.optimizer.param_groups[0].optimizer_kwargs["ns_steps"] == 7


def test_native_muon_factory_keeps_selection_and_applies_simulator_limits(monkeypatch):
    from torchtitan_npu.models.deepseek_v4 import config_registry

    original = config_registry.deepseek_v4_debugmodel

    def native_muon(*, num_mtp_layers=0):
        trainer = original(num_mtp_layers=num_mtp_layers)
        trainer.optimizer.name = "Muon"
        return trainer

    monkeypatch.setattr(config_registry, "deepseek_v4_debugmodel", native_muon)
    prepared = prepare_config(SimulationConfig(muon_ns_steps=6))

    assert prepared.trainer.optimizer.param_groups[0].optimizer_name == "DistMuon"
    assert prepared.trainer.optimizer.param_groups[0].optimizer_kwargs["ns_steps"] == 6
    with pytest.raises(ValueError, match="TP=1 and CP=1"):
        prepare_config(SimulationConfig(world_size=2, context_parallel_degree=2))


def test_explicit_adamw_rejects_already_materialized_muon_factory(monkeypatch):
    from torchtitan_npu.models.deepseek_v4 import config_registry

    original = config_registry.deepseek_v4_debugmodel

    def materialized_muon(*, num_mtp_layers=0):
        trainer = original(num_mtp_layers=num_mtp_layers)
        trainer.optimizer.name = "Muon"
        trainer.optimizer.materialize()
        return trainer

    monkeypatch.setattr(config_registry, "deepseek_v4_debugmodel", materialized_muon)
    with pytest.raises(ValueError, match="factory already materialized Muon"):
        prepare_config(SimulationConfig(optimizer_name="AdamW"))


def test_native_muon_factory_rejects_replica_dedup(monkeypatch):
    from torchtitan_npu.models.deepseek_v4 import config_registry

    original = config_registry.deepseek_v4_debugmodel

    def replica_dedup_muon(*, num_mtp_layers=0):
        trainer = original(num_mtp_layers=num_mtp_layers)
        trainer.optimizer.name = "Muon"
        trainer.optimizer.muon_enable_hsdp_replica_dedup = True
        return trainer

    monkeypatch.setattr(config_registry, "deepseek_v4_debugmodel", replica_dedup_muon)
    with pytest.raises(ValueError, match="HSDP replica dedup"):
        prepare_config(SimulationConfig())


@pytest.mark.parametrize(
    "options, message",
    [
        ({"config": "kimi_k3_smoketest", "optimizer_name": "Muon"}, "native Muon profile"),
        ({"optimizer_name": "Muon", "world_size": 2, "tensor_parallel_degree": 2}, "TP=1 and CP=1"),
        ({"optimizer_name": "Muon", "enable_quantized_training": True}, "quantized training"),
        (
            {"optimizer_name": "Muon", "memory_offload_optimizer_states": True},
            "Muon optimizer-state offload requires a dedicated",
        ),
        ({"muon_ns_steps": 3}, "requires optimizer_name=Muon"),
        ({"optimizer_name": "Muon", "muon_ns_steps": 0}, "positive integer"),
        (
            {
                "optimizer_name": "Muon",
                "override_imports": ["torchtitan_npu.override.common.optimizer.swap_optimizer"],
            },
            "swap or virtual optimizer",
        ),
    ],
    ids=[
        "kimi",
        "tp",
        "quantized",
        "optimizer-state-offload",
        "unused-muon-option",
        "invalid-iterations",
        "native-swap",
    ],
)
def test_unsupported_optimizer_requests_fail_explicitly(options, message):
    with pytest.raises(ValueError, match=message):
        prepare_config(SimulationConfig(**options))
