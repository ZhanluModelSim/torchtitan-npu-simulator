# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run the core SDC smoke scenarios in a fresh Python process each."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.smoke

logger = logging.getLogger(__name__)

_THIS_FILE = Path(__file__).resolve()
_NO_NPU_EXIT_CODE = 77
_CONTROL_ENVIRONMENT = (
    "NPU_ASD_CONFIG",
    "NPU_ASD_ENABLE",
    "PERF_DUMP_CONFIG",
)
# Compiled SDC uses typed ``SDC.Config`` values. Only torch-npu's eager
# controls remain process environment variables.
_SCENARIO_ENVIRONMENT = {
    "real-bf16-gradient": {"NPU_ASD_CONFIG": "enable:false", "NPU_ASD_ENABLE": "0"},
    "eager-gradient-controls": {"NPU_ASD_CONFIG": "enable:true"},
}
_COMPILED_GRADIENT_TUNING = {
    "cooldown": 6,
    "strikes_num": 4,
    "strikes_window": 481,
    "checksum_cooldown": 181,
    "upper_thresh1": 1_000_001,
    "upper_thresh2": 101,
    "grad_sample_interval": 4,
}


def _compiled_sdc_config(scenario: str) -> Any:
    from torchtitan_npu.extensions.components.sdc import SDC

    return {
        "compiled-gradient-controls": SDC.Config(gradient_enabled=True, **_COMPILED_GRADIENT_TUNING),
        "compiled-all-controls": SDC.Config(gradient_enabled=True, with_checksum=True, hccl_mode=1),
    }[scenario]


def _get_private_runtime_interface(target: Any, name: str) -> Any:
    """Read a private runtime interface required for an NPU smoke observation."""

    return getattr(target, name)


def _set_private_runtime_interface(target: Any, name: str, value: Any) -> None:
    """Replace a private runtime interface while an NPU smoke observation runs."""

    setattr(target, name, value)


# Pytest launcher


def _run_worker(scenario: str) -> None:
    repo_root = _THIS_FILE.parents[3]
    environment = os.environ.copy()
    # PyTorch treats CI as a compiler assertion mode; keep this smoke test
    # focused on SDC behavior while the child process runs the real graph.
    environment.pop("CI", None)
    # AscendC compile workers must not fork after the NPU is initialized.
    environment["TORCHINDUCTOR_WORKER_START"] = "spawn"
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(path for path in (str(repo_root), python_path) if path)

    completed = subprocess.run(
        [sys.executable, str(_THIS_FILE), "--worker", scenario],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=120,
    )

    if completed.returncode == _NO_NPU_EXIT_CODE:
        pytest.skip(completed.stdout.strip() or "isolated SDC worker has no available Ascend NPU")
    assert completed.returncode == 0, (
        f"isolated SDC worker failed for {scenario!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if scenario == "fp32-checksum-results":
        branches = [line for line in completed.stdout.splitlines() if line.startswith("FP32 native checksum branch:")]
        assert len(branches) == 1, "The FP32 worker must report its actual native checksum branch"
        print(branches[0])


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("compiled-gradient-controls", id="gradient"),
        pytest.param("compiled-all-controls", id="gradient-checksum-hccl"),
    ],
)
def test_compiled_controls_are_typed_without_installing_the_eager_wrapper(scenario: str) -> None:
    _run_worker(scenario)


def test_eager_sdc_controls_remain_owned_by_torch_npu() -> None:
    _run_worker("eager-gradient-controls")


def test_invalid_compiled_sdc_business_config_is_rejected() -> None:
    _run_worker("invalid-uncompiled-model")


def test_checksum_custom_op_declares_its_output_mutation() -> None:
    _run_worker("checksum-schema-declares-output-mutation")


def test_checksum_custom_op_propagates_torch_npu_output_mutation() -> None:
    _run_worker("checksum-output-mutation")


def test_compiled_consumer_observes_checksum_output_mutation() -> None:
    _run_worker("checksum-compiled-consumer-observes-mutation")


def test_compiled_gradient_checks_real_bf16_gradients() -> None:
    _run_worker("real-bf16-gradient")


def test_compiled_hccl_covers_forward_and_backward_collectives() -> None:
    _run_worker("global-hccl-scope")


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_three_gradient_strikes_activate_checksum_without_recompiling(dtype: str) -> None:
    scenario = "three-strikes-enable-checksum"
    _run_worker(scenario if dtype == "bf16" else f"{scenario}-fp32")


def test_fp32_checksum_preserves_compiled_results_and_detects_output_faults() -> None:
    _run_worker("fp32-checksum-results")


# Worker setup


def _set_scenario_environment(scenario: str) -> None:
    for name in _CONTROL_ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ.update(_SCENARIO_ENVIRONMENT.get(scenario, {}))


def _config(*, compiled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        compile=SimpleNamespace(
            enable=compiled,
            components=("model",) if compiled else (),
            backend="inductor",
        ),
        parallelism=SimpleNamespace(pipeline_parallel_degree=1),
        model_spec=SimpleNamespace(model=SimpleNamespace(enable_weight_tying=False)),
    )


def _initialize_process_group(torch: Any, device: Any) -> None:
    if "RANK" in os.environ:
        torch.npu.set_device(device)
        torch.distributed.init_process_group(backend="hccl")
        return
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.npu.set_device(device)
    torch.distributed.init_process_group(
        backend="hccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


def _configure_checker(checker: Any, *, checksum: bool) -> None:
    checker.set_strikes_num(3)
    checker.set_strikes_window(480)
    checker.set_cooldown(0)
    checker.set_checksum_cooldown(0)
    checker.set_upper_thresh1(1_000_000)
    checker.set_upper_thresh2(100)
    checker.set_with_checksum(checksum)


def _run_sdc_step(sdc: Any, step: Any, *args: Any) -> Any:
    result = step(*args)
    sdc.finalize_sdc_step()
    return result


def _wait_for(torch: Any, predicate: Any, *, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        torch.npu.synchronize()
        time.sleep(0.1)
    return predicate()


# Worker scenarios


def _check_compiled_controls(torch: Any, torch_npu: Any, scenario: str) -> None:
    from torchtitan_npu.extensions.components.sdc import SDC

    expected = {
        "compiled-gradient-controls": (True, False, False),
        "compiled-all-controls": (True, True, True),
    }[scenario]

    config = _compiled_sdc_config(scenario)
    SDC(config, _config(), model_parts=[], gradient_accumulation_steps=1)

    gradient_enabled, checksum_enabled, hccl_enabled = expected
    checker = torch_npu.asd.asd.matmul_check
    assert checker.get_matmul_hook_enable() == int(gradient_enabled)
    assert checker.get_with_checksum() == checksum_enabled
    assert (os.environ.get("NPU_ASD_ENABLE", "0") != "0") == hccl_enabled
    assert os.environ.get("NPU_ASD_ENABLE", "0") == str(config.hccl_mode)
    assert torch.nn.Module.__call__.__module__ != "torch_npu.utils._step"
    if scenario == "compiled-gradient-controls":
        assert {
            "cooldown": checker.get_cooldown(),
            "strikes_num": checker.get_strikes_num(),
            "strikes_window": checker.get_strikes_window(),
            "checksum_cooldown": checker.get_checksum_cooldown(),
            "upper_thresh1": checker.get_upper_thresh1(),
            "upper_thresh2": checker.get_upper_thresh2(),
            "grad_sample_interval": checker.get_grad_sample_interval(),
        } == _COMPILED_GRADIENT_TUNING


def _check_eager_controls(torch: Any) -> None:
    assert torch.nn.Module.__call__.__module__ == "torch_npu.utils._step"


def _check_invalid_config() -> None:
    from torchtitan_npu.extensions.components.sdc import SDC

    try:
        SDC.Config(gradient_enabled=True).build(
            trainer_config=_config(compiled=False),
            model_parts=[],
            gradient_accumulation_steps=1,
        )
    except (RuntimeError, ValueError) as error:
        assert "compile.enable" in str(error)
        return
    raise AssertionError("SDC accepted a configuration without a compiled model")


def _check_checksum_schema_declares_output_mutation(torch: Any) -> None:
    import torchtitan_npu.ops.misc.sdc_checksum  # noqa: F401

    # Dispatcher alias metadata is available only through a private runtime API.
    schema = _get_private_runtime_interface(torch.ops.torchtitan_npu.sdc_matmul_checksum.default, "_schema")
    output_argument = schema.arguments[2]

    assert output_argument.name == "output"
    assert output_argument.alias_info is not None
    assert output_argument.alias_info.is_write


def _install_checksum_output_mutator(torch: Any, torch_npu: Any, device: Any) -> tuple[Any, list[int]]:
    torch.npu.set_device(device)
    checker = torch_npu.asd.asd.matmul_check
    checker.checksum_result = torch.tensor(False, device=device)
    checksum_calls: list[int] = []

    def mutate_output(_left: Any, _right: Any, output: Any) -> Any:
        checksum_calls.append(output.data_ptr())
        output.fill_(3)
        return torch.tensor(False, device=output.device)

    torch_npu.matmul_checksum = mutate_output
    return checker, checksum_calls


def _check_checksum_output_mutation(torch: Any, torch_npu: Any) -> None:
    import torchtitan_npu.ops.misc.sdc_checksum  # noqa: F401

    device = torch.device("npu:0")
    checker, checksum_calls = _install_checksum_output_mutator(torch, torch_npu, device)
    checksum_op = torch.ops.torchtitan_npu.sdc_matmul_checksum.default
    left = torch.zeros((1, 2), device=device, dtype=torch.bfloat16)
    right = torch.zeros((2, 2), device=device, dtype=torch.bfloat16)
    eager_output = torch.mm(left, right)

    checker.checksum_enable = True
    checksum_op(left, right, eager_output)
    torch.npu.synchronize()

    assert torch.equal(eager_output, torch.full_like(eager_output, 3))
    assert len(checksum_calls) == 1


def _check_compiled_consumer_observes_checksum_output_mutation(torch: Any, torch_npu: Any) -> None:
    from torchtitan_npu.extensions.compile.sdc_checksum import install_checksum_pass

    device = torch.device("npu:0")
    checker, checksum_calls = _install_checksum_output_mutator(torch, torch_npu, device)

    class MatmulConsumer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((2, 2), device=device, dtype=torch.bfloat16))

        def forward(self, value: Any) -> Any:
            return torch.mm(value, self.weight) + 1

    install_checksum_pass()
    compiled = torch.compile(MatmulConsumer(), backend="inductor")
    left = torch.zeros((1, 2), device=device, dtype=torch.bfloat16)
    checker.checksum_enable = False

    gate_closed_output = compiled(left)
    torch.npu.synchronize()

    assert torch.equal(gate_closed_output, torch.ones_like(gate_closed_output))
    assert checksum_calls == []

    checker.checksum_enable = True
    gate_open_output = compiled(left)
    torch.npu.synchronize()

    assert torch.equal(gate_open_output, torch.full_like(gate_open_output, 4))
    assert len(checksum_calls) >= 1


def _check_fp32_checksum_results(torch: Any, torch_npu: Any) -> None:
    from torchtitan_npu.extensions.components.sdc import SDC

    device = torch.device("npu:0")
    _initialize_process_group(torch, device)
    left_cpu = (torch.arange(128, dtype=torch.float64).reshape(8, 16) % 13 - 6) / 16
    right_cpu = (torch.arange(128, dtype=torch.float64).reshape(16, 8) % 11 - 5) / 16
    upstream_cpu = (torch.arange(64, dtype=torch.float64).reshape(8, 8) % 7 + 1) / 8
    reference_left = left_cpu.clone().requires_grad_()
    reference_right = right_cpu.clone().requires_grad_()
    reference_output = torch.mm(reference_left, reference_right) + 1
    reference_output.backward(upstream_cpu)

    class MatmulConsumer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(right_cpu.to(device=device, dtype=torch.float32))

        def forward(self, value: Any) -> Any:
            return torch.mm(value, self.weight) + 1

    model = MatmulConsumer()
    SDC(
        SDC.Config(gradient_enabled=True, with_checksum=True),
        _config(),
        model_parts=[model],
        gradient_accumulation_steps=1,
    )
    checker = torch_npu.asd.asd.matmul_check
    assert checker.matmul_dtype_supported, "SDC must initialize FP32 checksum capability"
    checker.checksum_result = torch.tensor(False, device=device)
    checksum_calls: list[tuple[Any, Any, Any, bool]] = []
    original_checksum = torch_npu.matmul_checksum

    def checksum(left: Any, right: Any, output: Any) -> Any:
        result = original_checksum(left, right, output)
        checksum_calls.append((left.detach().cpu(), right.detach().cpu(), output.detach().cpu(), bool(result)))
        return result

    torch_npu.matmul_checksum = checksum
    compiled = torch.compile(model, backend="inductor")
    for gate in (False, True):
        checker.checksum_enable = gate
        checker.checksum_result.fill_(False)
        checksum_calls.clear()
        model.zero_grad(set_to_none=True)
        left = left_cpu.to(device=device, dtype=torch.float32).requires_grad_()
        output = compiled(left)
        output.backward(upstream_cpu.to(device=device, dtype=torch.float32))
        torch.npu.synchronize()

        torch.testing.assert_close(output.detach().cpu().double(), reference_output.detach(), rtol=0, atol=1e-5)
        torch.testing.assert_close(left.grad.cpu().double(), reference_left.grad, rtol=0, atol=1e-5)
        torch.testing.assert_close(model.weight.grad.cpu().double(), reference_right.grad, rtol=0, atol=1e-5)
        assert not bool(checker.checksum_result)
        assert bool(checksum_calls) is gate
        if gate:
            received_left, received_right, received_output, _ = checksum_calls[0]
            assert received_left.dtype == received_right.dtype == received_output.dtype == torch.float32
            torch.testing.assert_close(received_left.double(), left_cpu, rtol=0, atol=0)
            torch.testing.assert_close(received_right.double(), right_cpu, rtol=0, atol=0)
            torch.testing.assert_close(received_output.double(), torch.mm(left_cpu, right_cpu), rtol=0, atol=1e-5)
            assert all(not call[3] for call in checksum_calls)

    # Keep fault injection outside autograd; exercise the production op's native result accumulation.
    left = left_cpu.to(device=device, dtype=torch.float32)
    right = right_cpu.to(device=device, dtype=torch.float32)
    expected = torch.mm(left_cpu, right_cpu)
    corrupted = expected.to(device=device, dtype=torch.float32)
    corrupted[0, 0] += 1024
    checker.checksum_result.fill_(False)
    torch.ops.torchtitan_npu.sdc_matmul_checksum.default(left, right, corrupted)
    torch.npu.synchronize()
    assert bool(checker.checksum_result)
    assert checksum_calls[-1][3]

    # The native fallback repairs output; V-ABFT reports faults without this repair contract.
    checksum_module = sys.modules[original_checksum.__module__]
    abft_available = _get_private_runtime_interface(checksum_module, "_abft_available_cache")
    assert type(abft_available) is bool, "The native checksum branch must have been probed"
    print(f"FP32 native checksum branch: {'V-ABFT' if abft_available else 'Python fallback'}")
    if abft_available is False:
        torch.testing.assert_close(corrupted.cpu().double(), expected, rtol=0, atol=1e-5)
    torch.ops.torchtitan_npu.sdc_matmul_checksum.default(left, right, expected.to(device=device, dtype=torch.float32))
    torch.npu.synchronize()
    assert not checksum_calls[-1][3]
    assert bool(checker.checksum_result), "A later healthy matmul must not erase an earlier fault"


def _check_real_bf16_gradient(torch: Any, torch_npu: Any) -> None:
    from torchtitan_npu.extensions.components.sdc import SDC

    device = torch.device("npu:0")
    _initialize_process_group(torch, device)
    checker = torch_npu.asd.asd.matmul_check
    detected: list[tuple[str, Any]] = []
    # The smoke test wraps torch-npu's NPU-runtime private checker hook.
    original_detect_grad = _get_private_runtime_interface(checker, "_detect_grad")

    def detect_grad(gradient: Any, name: str) -> Any:
        detected.append((name, gradient.dtype))
        return original_detect_grad(gradient, name)

    _set_private_runtime_interface(checker, "_detect_grad", detect_grad)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    compiled = torch.compile(model, backend="inductor")

    def forward_backward(value: Any, target: Any) -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = (compiled(value).float() - target.float()).square().mean()
        loss.backward()
        return loss

    sdc = SDC(SDC.Config(gradient_enabled=True), _config(), model_parts=[model], gradient_accumulation_steps=1)
    _configure_checker(checker, checksum=False)
    loss = _run_sdc_step(
        sdc,
        forward_backward,
        torch.randn(1, 16, device=device, dtype=torch.bfloat16),
        torch.zeros(1, 8, device=device, dtype=torch.bfloat16),
    )
    torch.npu.synchronize()

    assert torch.isfinite(loss.float()).item()
    assert detected
    assert all(dtype == torch.bfloat16 for _, dtype in detected)


def _check_global_hccl_scope(torch: Any, torch_npu: Any) -> None:
    from torchtitan_npu.extensions.components.sdc import SDC

    state_events: list[tuple[str, str]] = []
    # The smoke test observes torch-npu's NPU-runtime private state API.
    npu_extension = _get_private_runtime_interface(torch_npu, "_C")
    original_call_state = _get_private_runtime_interface(npu_extension, "_npu_set_call_state")
    original_model_state = _get_private_runtime_interface(npu_extension, "_npu_set_module_train_state")

    def record_call_state(state: str) -> None:
        state_events.append(("call", state))
        original_call_state(state)

    def record_model_state(state: str) -> None:
        state_events.append(("model", state))
        original_model_state(state)

    _set_private_runtime_interface(npu_extension, "_npu_set_call_state", record_call_state)
    _set_private_runtime_interface(npu_extension, "_npu_set_module_train_state", record_model_state)
    device = torch.device("npu:0")
    _initialize_process_group(torch, device)
    # Exercise initialization-time communication before enabling typed SDC.
    initial_value = torch.ones(16, device=device, dtype=torch.bfloat16)
    torch.distributed.all_reduce(initial_value)
    torch.npu.synchronize()
    assert state_events == []

    class GlobalScopeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(16, 8, bias=False, device=device, dtype=torch.bfloat16)

        def forward(self, value: Any) -> Any:
            torch.distributed.all_reduce(value)
            return self.projection(value)

    model = GlobalScopeModel()
    sdc = SDC(SDC.Config(hccl_mode=2), _config(), model_parts=[model], gradient_accumulation_steps=1)
    assert os.environ["NPU_ASD_ENABLE"] == "2"
    assert state_events == [("model", "train"), ("call", "backward")]
    assert torch.nn.Module.__call__.__module__ != "torch_npu.utils._step"
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def all_reduce_gradient(gradient: Any) -> Any:
        torch.distributed.all_reduce(gradient.contiguous())
        return gradient

    model.projection.weight.register_hook(all_reduce_gradient)

    def forward_backward(value: Any, target: Any) -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(value).float() - target.float()).square().mean()
        loss.backward()
        return loss

    initial_events = list(state_events)
    loss = _run_sdc_step(
        sdc,
        forward_backward,
        torch.randn(1, 16, device=device, dtype=torch.bfloat16),
        torch.zeros(1, 8, device=device, dtype=torch.bfloat16),
    )
    torch.npu.synchronize()

    assert torch.isfinite(loss.float()).item()
    assert state_events == initial_events
    assert state_events == [("model", "train"), ("call", "backward")]


def _build_matmul(torch: Any, device: Any, *, dtype: Any) -> Any:
    class Matmul(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16, 8, device=device, dtype=dtype))

        def forward(self, value: Any) -> Any:
            return torch.mm(value, self.weight)

    return Matmul()


class _ThreeStrikesStepRunner:
    def __init__(self, torch: Any, device: Any, *, dtype: Any) -> None:
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self.model = _build_matmul(torch, device, dtype=dtype)
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self._compiled = torch.compile(self.model, backend="inductor")

    def run(self, sdc: Any, inject_fault: bool) -> None:
        self._optimizer.zero_grad(set_to_none=True)
        value = self._torch.randn(1, 16, device=self._device, dtype=self._dtype)
        target = self._torch.zeros(1, 8, device=self._device, dtype=self._dtype)
        loss = (self._compiled(value).float() - target.float()).square().mean()
        loss.backward()
        self._torch.npu.synchronize()
        if inject_fault:
            self.model.weight.grad.fill_(1.0e30)
            self._torch.npu.synchronize()
        sdc.finalize_sdc_step()


def _install_three_strikes_observers(
    torch: Any,
    torch_npu: Any,
    checksum_pass: Any,
    checker: Any,
) -> tuple[list[int], list[int], list[bool]]:
    resets: list[int] = []
    pass_installs: list[int] = []
    dispatches: list[bool] = []
    original_reset = torch.compiler.reset
    original_checksum = torch_npu.matmul_checksum
    original_install_checksum_pass = checksum_pass.install_checksum_pass

    def reset() -> None:
        resets.append(1)
        original_reset()

    def checksum(left: Any, right: Any, output: Any) -> Any:
        dispatches.append(bool(checker.checksum_enable))
        return original_checksum(left, right, output)

    def install_checksum_pass() -> None:
        pass_installs.append(1)
        original_install_checksum_pass()

    torch.compiler.reset = reset
    torch_npu.matmul_checksum = checksum
    checksum_pass.install_checksum_pass = install_checksum_pass
    return resets, pass_installs, dispatches


def _assert_three_strikes_activated(
    torch: Any,
    checker: Any,
    resets: list[int],
    dispatches: list[bool],
) -> None:
    assert _wait_for(torch, lambda: len(checker.history_abnormal_list) >= 3, timeout=25), checker.history_abnormal_list
    assert all(item.get("striked") for item in checker.history_abnormal_list[-3:])
    assert resets == []
    assert dispatches == []
    assert _wait_for(torch, lambda: bool(checker.checksum_enable), timeout=25)


def _check_three_strikes_enable_checksum(torch: Any, torch_npu: Any, *, dtype: Any) -> None:
    import torchtitan_npu.extensions.compile.sdc_checksum as checksum_pass
    from torchtitan_npu.extensions.components.sdc import SDC

    device = torch.device("npu:0")
    _initialize_process_group(torch, device)
    checker = torch_npu.asd.asd.matmul_check
    resets, pass_installs, dispatches = _install_three_strikes_observers(torch, torch_npu, checksum_pass, checker)

    runner = _ThreeStrikesStepRunner(torch, device, dtype=dtype)
    sdc = SDC(
        SDC.Config(gradient_enabled=True, with_checksum=True),
        _config(),
        model_parts=[runner.model],
        gradient_accumulation_steps=1,
    )
    assert checker.matmul_dtype_supported, "SDC must initialize the native checksum capability"
    _configure_checker(checker, checksum=True)
    checker.set_cooldown(0.05)
    for _ in range(3):
        runner.run(sdc, inject_fault=True)
        time.sleep(3.2)

    _assert_three_strikes_activated(torch, checker, resets, dispatches)
    runner.run(sdc, inject_fault=False)

    assert resets == []
    assert pass_installs == [1]
    assert dispatches
    assert all(dispatches)


def _run_scenario(torch: Any, torch_npu: Any, scenario: str) -> None:
    if scenario.startswith("compiled-"):
        _check_compiled_controls(torch, torch_npu, scenario)
    elif scenario.startswith("eager-"):
        _check_eager_controls(torch)
    elif scenario.startswith("invalid-"):
        _check_invalid_config()
    elif scenario == "checksum-schema-declares-output-mutation":
        _check_checksum_schema_declares_output_mutation(torch)
    elif scenario == "checksum-output-mutation":
        _check_checksum_output_mutation(torch, torch_npu)
    elif scenario == "checksum-compiled-consumer-observes-mutation":
        _check_compiled_consumer_observes_checksum_output_mutation(torch, torch_npu)
    elif scenario == "fp32-checksum-results":
        _check_fp32_checksum_results(torch, torch_npu)
    elif scenario == "real-bf16-gradient":
        _check_real_bf16_gradient(torch, torch_npu)
    elif scenario == "global-hccl-scope":
        _check_global_hccl_scope(torch, torch_npu)
    elif scenario in ("three-strikes-enable-checksum", "three-strikes-enable-checksum-fp32"):
        dtype = torch.float32 if scenario.endswith("-fp32") else torch.bfloat16
        _check_three_strikes_enable_checksum(torch, torch_npu, dtype=dtype)
    else:
        raise ValueError(f"unknown SDC smoke scenario: {scenario}")


def _worker_main(scenario: str) -> int:
    _set_scenario_environment(scenario)
    try:
        import torch
        import torch_npu
    except (ImportError, OSError, RuntimeError) as error:
        logger.warning("Ascend NPU runtime is unavailable: %s", error)
        return _NO_NPU_EXIT_CODE
    if not torch.npu.is_available():
        logger.warning("Ascend NPU runtime is unavailable: torch.npu.is_available() is false")
        return _NO_NPU_EXIT_CODE

    _run_scenario(torch, torch_npu, scenario)
    logger.info("SDC smoke scenario passed: %s", scenario)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit(f"usage: {_THIS_FILE.name} --worker <scenario>")
    raise SystemExit(_worker_main(sys.argv[2]))
