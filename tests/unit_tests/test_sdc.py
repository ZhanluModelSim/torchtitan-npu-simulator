# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch._inductor.config as inductor_config
from torch._inductor.custom_graph_pass import get_custom_graph_passes
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torchtitan.trainer import Trainer

from torchtitan_npu.extensions.compile import sdc_checksum
from torchtitan_npu.extensions.components.sdc.sdc import SDC
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


class TorchNpuCheckerFake:
    def __init__(self, *, selected: list[bool] | None = None) -> None:
        self.hook_enabled: int | None = None
        self.with_checksum = False
        self.cooldown = 5
        self.strikes_num = 3
        self.strikes_window = 480
        self.checksum_cooldown = 180
        self.upper_thresh1 = 1_000_000
        self.upper_thresh2 = 100
        self.grad_sample_interval = 3
        self.check_stat: dict[str, dict[str, float | int]] = {}
        self.matmul_dtype_supported = False
        self.stream_initializations = 0
        self.startups = 0
        self.detected: list[tuple[torch.Tensor, str]] = []
        self._selected = iter(selected or [])

    def set_matmul_hook_enable(self, value: int) -> None:
        self.hook_enabled = value

    def set_with_checksum(self, value: bool) -> None:
        self.with_checksum = value

    def set_cooldown(self, value: int) -> None:
        self.cooldown = value

    def set_strikes_num(self, value: int) -> None:
        self.strikes_num = value

    def set_strikes_window(self, value: int) -> None:
        self.strikes_window = value

    def set_checksum_cooldown(self, value: int) -> None:
        self.checksum_cooldown = value

    def set_upper_thresh1(self, value: int) -> None:
        self.upper_thresh1 = value

    def set_upper_thresh2(self, value: int) -> None:
        self.upper_thresh2 = value

    def set_grad_sample_interval(self, value: int) -> None:
        self.grad_sample_interval = value

    def init_stream(self) -> None:
        self.stream_initializations += 1

    def parameter_filtering(self) -> bool:
        return next(self._selected)

    def _startup(self) -> None:
        self.startups += 1

    def _detect_grad(self, gradient: torch.Tensor, state_key: str) -> None:
        self.detected.append((gradient, state_key))


def _install_torch_npu(
    monkeypatch,
    *,
    selected: list[bool] | None = None,
) -> tuple[TorchNpuCheckerFake, list[tuple[str, str]]]:
    checker = TorchNpuCheckerFake(selected=selected)
    state_events: list[tuple[str, str]] = []

    torch_npu = ModuleType("torch_npu")
    vars(torch_npu)["asd"] = SimpleNamespace(asd=SimpleNamespace(matmul_check=checker))
    vars(torch_npu)["_C"] = SimpleNamespace(
        _npu_set_module_train_state=lambda state: state_events.append(("model", state)),
        _npu_set_call_state=lambda state: state_events.append(("call", state)),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    return checker, state_events


def test_sdc_config_rejects_checksum_without_gradient_detection() -> None:
    with pytest.raises(ValueError, match=r"requires --sdc.gradient-enabled=true"):
        SDC.Config(with_checksum=True)


@pytest.mark.parametrize(
    "hccl_mode",
    [
        pytest.param(-1, id="below-range"),
        pytest.param(4, id="above-range"),
        pytest.param(1.5, id="non-integer"),
    ],
)
def test_sdc_config_rejects_invalid_hccl_modes(hccl_mode: Any) -> None:
    with pytest.raises(ValueError, match="must be 0, 1, 2 or 3"):
        SDC.Config(hccl_mode=hccl_mode)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        pytest.param("cooldown", 0, id="cooldown-below-range"),
        pytest.param("cooldown", 1.5, id="cooldown-non-integer"),
        pytest.param("strikes_num", 0, id="strikes-num-below-range"),
        pytest.param("strikes_window", 0, id="strikes-window-below-range"),
        pytest.param("checksum_cooldown", 0, id="checksum-cooldown-below-range"),
        pytest.param("upper_thresh1", 2, id="upper-thresh1-below-range"),
        pytest.param("upper_thresh2", 2, id="upper-thresh2-below-range"),
        pytest.param("grad_sample_interval", 0, id="grad-sample-interval-below-range"),
    ],
)
def test_sdc_config_rejects_invalid_gradient_tuning(option: str, value: Any) -> None:
    with pytest.raises(ValueError, match="must be an integer greater than or equal to"):
        SDC.Config(gradient_enabled=True, **{option: value})


def test_sdc_config_rejects_non_default_tuning_without_gradient_detection() -> None:
    with pytest.raises(ValueError, match=r"non-default --sdc gradient tuning requires"):
        SDC.Config(cooldown=6)


# Gradient monitoring


def _compiled_trainer_config(
    *,
    compile_enabled: bool = True,
    backend: str = "inductor",
    pipeline_parallel_degree: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        compile=SimpleNamespace(enable=compile_enabled, components=("model",), backend=backend),
        parallelism=SimpleNamespace(pipeline_parallel_degree=pipeline_parallel_degree),
    )


def _build_gradient_sdc(model_parts: list[Any], accumulation_steps: int = 1, **options: Any) -> SDC:
    return SDC.Config(gradient_enabled=True, **options).build(
        trainer_config=_compiled_trainer_config(),
        model_parts=model_parts,
        gradient_accumulation_steps=accumulation_steps,
    )


@pytest.fixture(autouse=True)
def _isolate_native_sdc_environment(monkeypatch):
    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize(
    ("accumulation_steps", "expected_submissions"),
    [
        pytest.param(1, [1, 2, 3], id="every-step"),
        pytest.param(3, [0, 0, 1, 1], id="every-third-step"),
    ],
)
def test_sdc_submits_only_at_configured_accumulation_boundaries(
    monkeypatch,
    accumulation_steps: int,
    expected_submissions: list[int],
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    checker, _ = _install_torch_npu(monkeypatch, selected=[True] * 4)
    sdc = _build_gradient_sdc([model], accumulation_steps)

    for expected in expected_submissions:
        sdc.finalize_sdc_step()
        assert len(checker.detected) == expected

    gradient, name = checker.detected[0]
    assert torch.equal(gradient, model.weight.grad)
    assert name == "weight_backward"


@pytest.mark.parametrize(
    ("dtype", "expected_flag"),
    [
        pytest.param(torch.bfloat16, True, id="bf16"),
        pytest.param(torch.float32, True, id="fp32"),
        pytest.param(torch.float16, False, id="fp16-excluded"),
    ],
)
def test_sdc_marks_only_native_checksum_dtypes(monkeypatch, dtype: torch.dtype, expected_flag: bool) -> None:
    checker, _ = _install_torch_npu(monkeypatch, selected=[True])
    model = torch.nn.Linear(2, 2, bias=False, dtype=dtype)

    _build_gradient_sdc([model])

    assert checker.matmul_dtype_supported is expected_flag


# Checksum graph instrumentation


@pytest.mark.parametrize(
    ("dtype", "device", "is_inference", "expected_checks"),
    [
        pytest.param(torch.bfloat16, "npu", False, 1, id="bf16-training"),
        pytest.param(torch.float32, "npu", False, 1, id="fp32-training"),
        pytest.param(torch.float16, "npu", False, 0, id="fp16-unsupported"),
        pytest.param(torch.bfloat16, "cpu", False, 0, id="cpu-excluded"),
        pytest.param(torch.float32, "npu", True, 0, id="inference-excluded"),
    ],
)
def test_checksum_pass_instruments_native_supported_training_dtypes(
    monkeypatch,
    dtype: torch.dtype,
    device: str,
    is_inference: bool,
    expected_checks: int,
) -> None:
    # Fake tensors provide NPU graph metadata without executing an NPU kernel.
    monkeypatch.setattr(inductor_config, "post_grad_custom_post_pass", None)
    fake_mode = FakeTensorMode()
    value = FakeTensor(fake_mode, torch.empty((2, 2), device="meta", dtype=dtype), torch.device(device))
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    output = graph.call_function(torch.ops.aten.mm.default, args=(left, right))
    graph.output(output)
    for node in (left, right, output):
        node.meta["val"] = value

    sdc_checksum.install_checksum_pass()
    passes = get_custom_graph_passes(inductor_config.post_grad_custom_post_pass)
    assert len(passes) == 1
    passes[0](graph, is_inference=is_inference)
    graph.lint()

    checks = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.torchtitan_npu.sdc_matmul_checksum.default
    ]
    assert len(checks) == expected_checks
    if checks:
        assert checks[0].args == (left, right, output)
        assert checks[0].meta["val"] is None


# SDC facade


def test_sdc_build_finishes_runtime_initialization_in_one_call(monkeypatch) -> None:
    checker, state_events = _install_torch_npu(monkeypatch, selected=[True])
    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    checksum_installs: list[bool] = []
    checksum_module = ModuleType("torchtitan_npu.extensions.compile.sdc_checksum")

    def install_checksum_pass() -> None:
        checksum_installs.append(True)

    vars(checksum_module)["install_checksum_pass"] = install_checksum_pass
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.compile.sdc_checksum", checksum_module)
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16)))

    SDC.Config(
        gradient_enabled=True,
        with_checksum=True,
        hccl_mode=2,
        cooldown=6,
        strikes_num=4,
        strikes_window=481,
        checksum_cooldown=181,
        upper_thresh1=1_000_001,
        upper_thresh2=101,
        grad_sample_interval=4,
    ).build(
        trainer_config=_compiled_trainer_config(),
        model_parts=[model],
        gradient_accumulation_steps=2,
    )

    assert checker.hook_enabled == 1
    assert checker.with_checksum is True
    assert checker.cooldown == 6
    assert checker.strikes_num == 4
    assert checker.strikes_window == 481
    assert checker.checksum_cooldown == 181
    assert checker.upper_thresh1 == 1_000_001
    assert checker.upper_thresh2 == 101
    assert checker.grad_sample_interval == 4
    assert checker.stream_initializations == 1
    assert checker.startups == 1
    assert checker.matmul_dtype_supported is True
    assert checker.check_stat == {
        "weight_backward": {"avg": 0.0, "pre_val": 0.0, "step": 0, "none_zero_step": 0},
    }
    assert os.environ["NPU_ASD_ENABLE"] == "2"
    assert state_events == [("model", "train"), ("call", "backward")]
    assert checksum_installs == [True]


@pytest.mark.parametrize(
    ("trainer_options", "message"),
    [
        pytest.param({"compile_enabled": False}, r"requires compile.enable=true", id="compile-disabled"),
        pytest.param({"backend": "aot_eager"}, r"requires compile.backend='inductor'", id="non-inductor-backend"),
        pytest.param({"pipeline_parallel_degree": 2}, r"does not support pipeline parallelism", id="pipeline-parallel"),
    ],
)
def test_active_sdc_validates_compile_before_runtime_changes(
    monkeypatch,
    trainer_options: dict[str, Any],
    message: str,
) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    trainer_config = _compiled_trainer_config(**trainer_options)

    with pytest.raises(ValueError, match=message):
        SDC.Config(gradient_enabled=True).build(
            trainer_config=trainer_config, model_parts=[], gradient_accumulation_steps=1
        )

    assert checker.hook_enabled is None
    assert checker.startups == 0
    assert state_events == []


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        pytest.param({"NPU_ASD_ENABLE": "4"}, r"invalid native SDC switch", id="invalid-native-value"),
        pytest.param(
            {"NPU_ASD_ENABLE": "2"},
            r"cannot be combined with enabled NPU_ASD_ENABLE",
            id="native-hccl-enabled",
        ),
        pytest.param(
            {"NPU_ASD_CONFIG": "enable:false,enable:true"},
            r"cannot be combined with enabled NPU_ASD_CONFIG",
            id="last-enable-entry-wins",
        ),
    ],
)
def test_compiled_sdc_rejects_enabled_native_environment(
    monkeypatch,
    environment: dict[str, str],
    message: str,
) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        SDC.Config(gradient_enabled=True).build(
            trainer_config=_compiled_trainer_config(), model_parts=[], gradient_accumulation_steps=1
        )

    assert checker.hook_enabled is None
    assert checker.startups == 0
    assert state_events == []


# Trainer integration


def _patch_trainer_sdc_flow(monkeypatch) -> tuple[list[object], TrainerEx.Config]:
    events: list[object] = []

    def finalize_sdc_step() -> None:
        events.append("sdc-after-step")

    def build_sdc(
        *, trainer_config: object, model_parts: list[Any], gradient_accumulation_steps: int
    ) -> SimpleNamespace:
        events.append(("sdc-create", trainer_config, tuple(model_parts), gradient_accumulation_steps))
        return SimpleNamespace(finalize_sdc_step=finalize_sdc_step)

    config = cast(
        "TrainerEx.Config",
        SimpleNamespace(
            sdc=SimpleNamespace(build=build_sdc),
            anticipatory=SimpleNamespace(enable=False),
            extension=SimpleNamespace(quantization=SimpleNamespace(enable_quantized_training=False)),
            training=SimpleNamespace(extension=SimpleNamespace(allow_hf32=True)),
        ),
    )

    def base_init(trainer: TrainerEx, received_config: object) -> None:
        events.append(("base-init", received_config))
        trainer.init_distributed()
        trainer.model_parts = ["model"]
        trainer.gradient_accumulation_steps = 2

    def init_distributed(_trainer: Trainer) -> SimpleNamespace:
        events.append("base-distributed")
        return SimpleNamespace()

    def forward_backward_step(_trainer: Trainer, **kwargs: object) -> str:
        events.append(("base-step", kwargs))
        return "base-result"

    monkeypatch.setattr(TrainerEx.__base__, "__init__", base_init)
    monkeypatch.setattr("torchtitan_npu.extensions.torchtitan.trainer.set_allow_hf32", lambda _allow_hf32: None)
    monkeypatch.setattr(TrainerEx.__base__, "init_distributed", init_distributed)
    monkeypatch.setattr(TrainerEx.__base__, "forward_backward_step", forward_backward_step)
    return events, config


def test_trainer_initializes_sdc_once_after_model_setup(monkeypatch) -> None:
    events, config = _patch_trainer_sdc_flow(monkeypatch)

    trainer = cast("TrainerEx", object.__new__(TrainerEx))
    TrainerEx.__init__(trainer, config)
    result = trainer.forward_backward_step(
        input_dict={"tokens": "batch"},  # type: ignore[dict-item]
        labels=cast("torch.Tensor", "labels"),
        global_valid_tokens=cast("torch.Tensor", "valid-tokens"),
    )

    assert result == "base-result"
    assert events == [
        ("base-init", config),
        "base-distributed",
        ("sdc-create", config, ("model",), 2),
        (
            "base-step",
            {
                "input_dict": {"tokens": "batch"},
                "labels": "labels",
                "global_valid_tokens": "valid-tokens",
            },
        ),
        "sdc-after-step",
    ]


def test_trainer_failed_backward_does_not_advance_gradient_accumulation(monkeypatch) -> None:
    checker, _ = _install_torch_npu(monkeypatch, selected=[True])
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
    model.weight.grad = torch.ones_like(model.weight)
    sdc = _build_gradient_sdc([model], accumulation_steps=2)
    trainer = cast("TrainerEx", object.__new__(TrainerEx))
    vars(trainer)["_sdc"] = sdc
    vars(trainer)["anticipatory_schedule"] = None
    outcomes = iter(["ok", "failure", "ok"])

    def forward_backward_step(_trainer: Trainer, *_args: Any, **_kwargs: Any) -> str:
        outcome = next(outcomes)
        if outcome == "failure":
            raise RuntimeError("backward failed")
        return outcome

    monkeypatch.setattr(TrainerEx.__base__, "forward_backward_step", forward_backward_step)

    assert trainer.forward_backward_step() == "ok"
    assert checker.detected == []
    with pytest.raises(RuntimeError, match="backward failed"):
        trainer.forward_backward_step()
    assert checker.detected == []
    assert trainer.forward_backward_step() == "ok"

    assert len(checker.detected) == 1
    assert checker.detected[0][1] == "weight_backward"
    assert torch.equal(checker.detected[0][0], torch.ones(2, 2))
