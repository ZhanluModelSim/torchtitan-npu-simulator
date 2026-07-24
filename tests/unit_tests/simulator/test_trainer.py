# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan_npu.simulator.trainer import (
    SimulationTrainer,
    _capture_num_micro_batches,
    _l0_csv_filename,
    _strip_hardware_dependent_model_converters,
    run_simulation_step,
)


@pytest.fixture
def fake_world():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29913"
    dist.init_process_group("fake", rank=0, world_size=4)
    yield
    dist.destroy_process_group()


def _build_parallel_dims(world_size: int):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=world_size, cp=1, tp=1, pp=1, ep=1, etp=1, world_size=world_size
    )
    parallel_dims.build_mesh()
    return parallel_dims


def test_capture_num_micro_batches_uses_pp_schedule_not_gradient_accumulation():
    assert _capture_num_micro_batches(
        pp_enabled=True,
        pp_schedule=SimpleNamespace(_n_microbatches=4),
        gradient_accumulation_steps=1,
    ) == 4
    assert _capture_num_micro_batches(
        pp_enabled=False,
        pp_schedule=None,
        gradient_accumulation_steps=3,
    ) == 3
    with pytest.raises(RuntimeError, match="_n_microbatches"):
        _capture_num_micro_batches(
            pp_enabled=True,
            pp_schedule=SimpleNamespace(_n_microbatches=0),
            gradient_accumulation_steps=1,
        )


def test_run_simulation_step_produces_complete_workload_graph(fake_world):
    parallel_dims = _build_parallel_dims(4)

    model = nn.Linear(8, 8, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        pred = model(input_dict["input"])
        loss = pred.sum() / global_valid_tokens
        loss.backward()
        return loss

    input_dict = {"input": torch.randn(2, 8, device="meta")}
    labels = torch.randint(0, 10, (2, 8), device="meta")

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict=input_dict,
        labels=labels,
        optimizer_step=optimizer.step,
        lr_scheduler_step=lr_scheduler.step,
        local_batch_size=2,
        seq_len=8,
    )

    assert graph.num_iterations == 1
    assert {"s0_F", "s0_B", "s0_OPTIMIZER"} <= graph.step_templates.keys()
    assert {template.step_type for template in graph.step_templates.values()} >= {
        "F",
        "B",
        "OPTIMIZER",
    }
    schedule = graph.iteration.schedule
    assert {instance.step_ref for instance in schedule.instances} == {
        "s0_F",
        "s0_B",
        "s0_OPTIMIZER",
    }
    assert {instance.pipeline_stage for instance in schedule.instances} == {0}
    assert schedule.annotations["rank_table"]["world_size"] == 4


def test_run_simulation_step_captures_optimizer_phase(fake_world):
    parallel_dims = _build_parallel_dims(4)
    model = nn.Linear(4, 4, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        loss = model(input_dict["input"]).sum() / global_valid_tokens
        loss.backward()
        return loss

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict={"input": torch.randn(2, 4, device="meta")},
        labels=torch.randint(0, 10, (2, 4), device="meta"),
        optimizer_step=optimizer.step,
        lr_scheduler_step=lambda: None,
        local_batch_size=2,
        seq_len=4,
    )
    optimizer_ops = graph.step_templates["s0_OPTIMIZER"].nodes
    assert len(optimizer_ops) > 0


def test_run_simulation_step_can_disable_memory_tracking(fake_world):
    parallel_dims = _build_parallel_dims(4)
    model = nn.Linear(4, 4, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        loss = model(input_dict["input"]).sum() / global_valid_tokens
        loss.backward()
        return loss

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict={"input": torch.randn(2, 4, device="meta")},
        labels=torch.randint(0, 10, (2, 4), device="meta"),
        optimizer_step=optimizer.step,
        lr_scheduler_step=lambda: None,
        local_batch_size=2,
        seq_len=4,
        enable_memory_tracking=False,
    )

    assert "memory_plan" not in graph.iteration.schedule.annotations
    assert "memory_summary" not in graph.iteration.schedule.annotations
    assert all(step.param_mem == 0 for step in graph.step_templates.values())
    assert all(step.peak_active_mem == 0 for step in graph.step_templates.values())


@pytest.mark.parametrize(
    ("output_formats", "expect_memory_export"),
    [([], False), (["json", "csv", "text", "html", "trace"], False), (["mem"], True)],
)
def test_memory_export_requires_explicit_mem_format(tmp_path, monkeypatch, output_formats, expect_memory_export):
    exported: list[object] = []
    memory_plan = object()
    monkeypatch.setattr(
        "torchtitan_npu.simulator.trainer.export_memory_plan",
        lambda plan, _out_dir: exported.append(plan),
    )
    monkeypatch.setattr(
        "torchtitan_npu.simulator.trainer.export_json",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "torchtitan_npu.simulator.trainer.export_kernel_summary_csv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "torchtitan_npu.simulator.trainer.write_text_summary",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "torchtitan_npu.simulator.trainer.export_html",
        lambda *_args: None,
    )

    schedule = SimpleNamespace(
        annotations={"memory_plan": memory_plan},
        export_l1_schedule_csv=lambda *_args, **_kwargs: None,
    )
    workload_graph = SimpleNamespace(
        iteration=SimpleNamespace(schedule=schedule),
        schedule_plan=None,
        step_templates={},
        export_schedule_csv=lambda *_args, **_kwargs: None,
    )
    trainer = SimpleNamespace(
        workload_graph=workload_graph,
        simulation_config=SimpleNamespace(
            output_dir=str(tmp_path),
            output_formats=output_formats,
            csv_max_ranks=None,
        ),
        config=SimpleNamespace(comm=SimpleNamespace(mode="fake_backend")),
    )

    SimulationTrainer._export(trainer)

    assert exported == ([memory_plan] if expect_memory_export else [])


def test_l0_csv_filename_preserves_virtual_stage_templates():
    from collections import Counter

    counts = Counter({"F": 2, "B": 1})

    assert _l0_csv_filename("s0_F", "F", counts) == "step_s0_F_l0_ops.csv"
    assert _l0_csv_filename("s3_F", "F", counts) == "step_s3_F_l0_ops.csv"
    assert _l0_csv_filename("s0_B", "B", counts) == "step_B_l0_ops.csv"


def test_simulation_trainer_config_build_dispatches_to_simulation_trainer():
    # Regression test for the Configurable._owner auto-wiring mechanism
    # this design relies on (verified against the pinned torchtitan
    # source): `SimulationTrainerConfig().build()` must construct a
    # SimulationTrainer, not a plain Trainer, even though
    # `SimulationTrainer.Config = SimulationTrainerConfig` uses simple
    # attribute assignment rather than nested `class Config:` syntax.
    from torchtitan_npu.simulator.trainer import SimulationTrainerConfig

    assert SimulationTrainerConfig._owner.__name__ == "SimulationTrainer"


def _fake_converter_config(name: str):
    # Mirrors the real `_owner`/`_model_config.name` shape read by both
    # `torchtitan_npu.converters.registry.has_npu_converter` and
    # `_strip_hardware_dependent_model_converters`: the dynamically
    # generated converter class carries `_model_config` (whose `.name` is
    # the registered patch name), and its Config carries `_owner` pointing
    # back at that converter class.
    owner = SimpleNamespace(_model_config=SimpleNamespace(name=name))
    return SimpleNamespace(_owner=owner)


def test_strip_hardware_dependent_model_converters_removes_nothing():
    # Updated expectation (was: strips npu_smla only). SMLA is no longer stripped either --
    # SimulationTrainer now installs SimSMLAConverter via apply_smla_shims() instead (Task 4),
    # so npu_smla stays in the converters list and gets a real (shim) implementation rather
    # than being dropped to the base class. _HARDWARE_DEPENDENT_CONVERTER_NAMES is now empty.
    config = SimpleNamespace(
        model_converters=SimpleNamespace(
            converters=[
                _fake_converter_config("npu_rms_norm"),
                _fake_converter_config("npu_mhc_pre"),
                _fake_converter_config("npu_mhc_post"),
                _fake_converter_config("npu_smla"),
                _fake_converter_config("npu_gmm"),
            ]
        )
    )
    _strip_hardware_dependent_model_converters(config)
    remaining_names = {c._owner._model_config.name for c in config.model_converters.converters}
    assert remaining_names == {"npu_rms_norm", "npu_mhc_pre", "npu_mhc_post", "npu_smla", "npu_gmm"}


def test_strip_hardware_dependent_model_converters_handles_missing_or_empty_converters():
    # must not raise when model_converters/converters is absent or empty
    _strip_hardware_dependent_model_converters(SimpleNamespace())
    _strip_hardware_dependent_model_converters(SimpleNamespace(model_converters=None))
    _strip_hardware_dependent_model_converters(SimpleNamespace(model_converters=SimpleNamespace(converters=[])))


def test_simulation_config_defaults_target_npu_device_type_to_non_a5():
    from torchtitan_npu.simulator.trainer import SimulationConfig

    config = SimulationConfig(output_dir="./out")
    assert config.target_npu_device_type == "non_a5"
    assert config.enable_memory_tracking is True
    assert config.world_size is None
