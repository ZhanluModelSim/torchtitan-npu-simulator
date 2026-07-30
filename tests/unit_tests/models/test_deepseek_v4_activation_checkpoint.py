# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torchtitan.distributed.activation_checkpoint as activation_checkpoint
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.config import ActivationCheckpointConfig

from torchtitan_npu.models.deepseek_v4.activation_checkpoint import (
    _resolve_save_ops,
    apply_deepseek_v4_ac,
)
from torchtitan_npu.patches.torchao_npu.mx_linear import NpuMXFP8MM
from torchtitan_npu.simulator.capture.checkpoint_execution import (
    install_checkpoint_execution_tracking,
)
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker


EXPECTED_SAVE_OPS = {
    "aten._grouped_mm.default",
    "npu.npu_quant_matmul.default",
    "npu.npu_grouped_matmul.default",
}


class _MXFP8Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(16, 32, device="meta", dtype=torch.bfloat16)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return NpuMXFP8MM.apply(inputs, self.weight)


class _ModelWithLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _MXFP8Block()})

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](inputs)


def _capture_mxfp8_checkpoint(mode: str) -> list:
    model = _ModelWithLayers()
    apply_deepseek_v4_ac(
        model,
        ActivationCheckpointConfig(mode=mode),
        model_compile_enabled=False,
        base_folder=".",
    )
    assert install_checkpoint_execution_tracking([model]) == 1

    boundary = StepBoundaryTracker()
    capture = OpDispatchCapture(phase_provider=lambda: boundary.current_phase)
    inputs = torch.empty(
        8,
        32,
        device="meta",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("meta")
    try:
        with boundary, capture:
            model(inputs).sum().backward()
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)
    return list(capture.build_nodes().values())


def test_deepseek_v4_save_ops_resolve_registered_dispatcher_ops():
    assert {str(op) for op in _resolve_save_ops()} == EXPECTED_SAVE_OPS


def test_selective_ac_extends_save_ops_only_during_model_wrapping():
    original_get_save_ops = activation_checkpoint._get_save_ops
    observed_save_ops = None

    def _capture_save_ops(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal observed_save_ops
        observed_save_ops = activation_checkpoint._get_save_ops()

    with patch.object(activation_checkpoint, "apply_ac", _capture_save_ops):
        apply_deepseek_v4_ac(
            nn.Linear(4, 4),
            SimpleNamespace(mode="selective"),
            model_compile_enabled=False,
            base_folder=".",
        )

    assert observed_save_ops is not None
    assert EXPECTED_SAVE_OPS <= {str(op) for op in observed_save_ops}
    assert activation_checkpoint._get_save_ops is original_get_save_ops


def test_full_ac_does_not_extend_upstream_save_ops():
    original_save_ops = {torch.ops.aten.relu.default}
    observed_save_ops = None

    def _capture_save_ops(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal observed_save_ops
        observed_save_ops = activation_checkpoint._get_save_ops()

    with (
        patch.object(
            activation_checkpoint,
            "_get_save_ops",
            return_value=original_save_ops,
        ) as get_save_ops,
        patch.object(activation_checkpoint, "apply_ac", _capture_save_ops),
    ):
        apply_deepseek_v4_ac(
            nn.Linear(4, 4),
            SimpleNamespace(mode="full"),
            model_compile_enabled=False,
            base_folder=".",
        )

    assert observed_save_ops == original_save_ops
    assert get_save_ops.call_count == 1


def test_selective_ac_caches_real_npu_quant_matmul_but_recomputes_quantization():
    selective_nodes = _capture_mxfp8_checkpoint("selective")
    full_nodes = _capture_mxfp8_checkpoint("full")

    def _recompute_ops(nodes: list) -> set[str]:
        return {
            node.annotations["raw_op_type"]
            for node in nodes
            if node.annotations["execution_kind"] == "recompute"
        }

    selective_recompute_ops = _recompute_ops(selective_nodes)
    full_recompute_ops = _recompute_ops(full_nodes)

    assert "npu.npu_dynamic_mx_quant.default" in selective_recompute_ops
    assert "npu.npu_quant_matmul.default" not in selective_recompute_ops
    assert "npu.npu_quant_matmul.default" in full_recompute_ops
