# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed geometry changes must reconstruct every native consumer consistently."""

from dataclasses import fields, is_dataclass

import pytest
from torchtitan_simulator.cli import main
from torchtitan_simulator.config import SimulationConfig, prepare_config
from torchtitan_simulator.model_overrides import ModelOverrides, MoEOverrides


def _geometry_snapshot(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _geometry_snapshot(getattr(value, item.name))
            for item in fields(value)
            if item.name not in {"param_init", "sharding_config"}
        }
    if isinstance(value, (list, tuple)):
        return [_geometry_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: _geometry_snapshot(item) for key, item in value.items()}
    return None if callable(value) else value


@pytest.mark.parametrize(
    "preset",
    [
        "deepseek_v4_smoketest",
        "deepseek_v4_flash_baseline_bf16",
        "deepseek_v4_pro_baseline_bf16",
        "deepseek_v4_pro_20t_baseline_bf16",
    ],
)
def test_empty_override_preserves_the_native_config_tree(preset):
    original = prepare_config(SimulationConfig(config=preset, num_mtp_modules=1))
    expected = _geometry_snapshot(original.trainer.model_spec.model)
    rebuilt = prepare_config(SimulationConfig(config=preset, num_mtp_modules=1, model_overrides=ModelOverrides()))
    assert _geometry_snapshot(rebuilt.trainer.model_spec.model) == expected
    assert _geometry_snapshot(original.trainer.model_spec.model) == expected


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
def test_typed_cli_geometry_reaches_quantization_loss_backward_and_optimizer(mode, tmp_path):
    workload = main(
        [
            "--extension.quantization.enable-quantized-training",
            "--training.num-mtp-modules",
            "1",
            "--activation-checkpoint.mode",
            mode,
            "--simulation.world-size",
            "2",
            "--model-overrides.dim",
            "128",
            "--model-overrides.vocab-size",
            "512",
            "--model-overrides.n-layers",
            "3",
            "--model-overrides.n-heads",
            "8",
            "--model-overrides.q-lora-rank",
            "64",
            "--model-overrides.o-lora-rank",
            "64",
            "--model-overrides.moe-inter-dim",
            "128",
            "--model-overrides.moe-args.top-k",
            "2",
            "--simulation.output-dir",
            str(tmp_path),
            "--simulation.output-formats",
            "json",
            "mem",
        ]
    )
    annotations = workload.iteration.schedule.annotations
    layouts = annotations["parameter_layouts"]
    expected = {
        "tok_embeddings.weight": (512, 128),
        "lm_head.weight": (512, 128),
        "layers.0.attention.wq_a.weight": (64, 128),
        "layers.0.attention.wq_b.weight": (4096, 64),
        # BatchedLinear stores [groups*out_features, per_group_in], then
        # unflattens the group axis for the native BMM.
        "layers.0.attention.wo_a.weight": (128, 2048),
        "layers.0.moe.routed_experts.inner_experts.w1_EFD": (4, 128, 128),
        "mtp_layers.0.e_proj.weight": (128, 128),
    }
    for name, shape in expected.items():
        assert tuple(layouts[name]["original_shape"]) == shape
    assert set(annotations["gradient_validation"]) == set(layouts)
    geometry = annotations["model_configuration"]
    assert geometry["dim"] == 128 and geometry["vocab_size"] == 512
    assert geometry["num_layers"] == 3 and geometry["num_mtp_layers"] == 1
    assert geometry["overrides"]["moe_args"]["top_k"] == 2
    assert len(annotations["quantization_configuration"]["quantized_parameters"]) == 45
    raw = annotations["memory_plan"].raw_events
    assert any(event.raw_op_type == "npu.npu_quant_matmul.default" for event in raw)
    assert any(event.raw_op_type == "aten._fused_adamw_.default" for event in raw)
    assert (tmp_path / "simulation_result.json").is_file()


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"dim": -1}, "invalid value"),
        ({"n_layers": 5}, "cover every main layer"),
        ({"n_heads": 7}, "divisible by o_groups"),
        ({"index_head_dim": 96}, "power of two"),
        ({"compress_ratios": (2, 2, 2, 2)}, "native ratios"),
        ({"moe_args": MoEOverrides(num_experts=2)}, "top_k exceeds"),
        ({"norm_eps": float("nan")}, "finite"),
    ],
)
def test_invalid_typed_geometry_is_rejected_before_model_construction(changes, message):
    with pytest.raises(ValueError, match=message):
        prepare_config(SimulationConfig(model_overrides=ModelOverrides(**changes)))


def test_pro_geometry_can_introduce_plain_rope_without_reusing_compressed_theta():
    prepared = prepare_config(
        SimulationConfig(
            config="deepseek_v4_pro_baseline_bf16", model_overrides=ModelOverrides(n_layers=1, compress_ratios=(1,))
        )
    )
    assert prepared.trainer.model_spec.model.layers[0].attention.rope.theta == 10000.0


def test_legacy_boolean_cli_alias_preserves_explicit_false(monkeypatch):
    import torchtitan_simulator.cli as cli

    original = cli.build_meta_model
    observed = []

    def inspect_model(prepared):
        model = original(prepared)
        observed.extend(layer.moe.router.route_norm for layer in model.layers.values())
        return model

    monkeypatch.setattr(cli, "build_meta_model", inspect_model)
    report = main(
        ["--build-only", "--config", "deepseek_v4_flash_baseline_bf16", "--model-overrides.moe-args.no-route-norm"]
    )
    assert report["scope"] == "model_build_only"
    assert observed and not any(observed)


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
def test_legacy_zero_compression_runs_as_native_window_attention(quantized):
    from torchtitan_simulator.runner import simulate

    common = dict(
        world_size=4,
        expert_parallel_degree=2,
        num_mtp_modules=1,
        local_batch_size=1,
        enable_quantized_training=quantized,
    )
    runs = []
    for plain_ratio in (0, 1):
        overrides = ModelOverrides(n_layers=3, dim=128, vocab_size=512, compress_ratios=(plain_ratio, 4, 128))
        prepared = prepare_config(SimulationConfig(**common, model_overrides=overrides))
        model = prepared.trainer.model_spec.model
        assert model.compress_ratios == (1, 4, 128)
        assert model.layers[0].attention.compress_ratio == 1
        assert model.layers[0].attention.compressor is None
        assert model.layers[0].attention.indexer is None
        runs.append(simulate(SimulationConfig(**common, model_overrides=overrides)).iteration.schedule.annotations)
    before, reference = runs
    assert before["gradient_validation"] == reference["gradient_validation"]
    assert before["quantization_configuration"] == reference["quantization_configuration"]
    assert [(event.raw_op_type, event.execution_kind) for event in before["memory_plan"].raw_events] == [
        (event.raw_op_type, event.execution_kind) for event in reference["memory_plan"].raw_events
    ]
    assert before["memory_plan"].peak_active_bytes == reference["memory_plan"].peak_active_bytes


def test_20t_preset_retains_plain_last_layer_with_native_ratio():
    prepared = prepare_config(SimulationConfig(config="deepseek_v4_pro_20t_baseline_bf16", num_mtp_modules=1))
    model = prepared.trainer.model_spec.model
    assert model.n_layers == 96
    assert model.compress_ratios == (128, 4) * 47 + (128, 1)
    assert model.layers[-1].attention.compress_ratio == 1
    assert model.layers[-1].attention.compressor is None
