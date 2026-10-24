# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native precision choices must reach model construction and FSDP dispatch."""

import pytest
import torch
from torchtitan_simulator.cli import main
from torchtitan_simulator.config import SimulationConfig, build_meta_model, prepare_config
from torchtitan_simulator.runtime import meta_runtime


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_base_dtype_overrides_process_default_and_restores_it(dtype):
    prepared = prepare_config(SimulationConfig(training_dtype=dtype))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with meta_runtime() as runtime:
            runtime.configure_model(prepared)
            model = build_meta_model(prepared)
        assert model.tok_embeddings.weight.dtype == getattr(torch, dtype)
        assert model.layers["0"].attention.attn_sink.dtype == torch.float32
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("param_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "mxfp8"])
def test_native_precision_cli_reaches_fsdp_backward_and_optimizer(param_dtype, mode, quantized, tmp_path):
    arguments = [
        "--simulation.world-size",
        "4",
        "--parallelism.data-parallel-replicate-degree",
        "2",
        "--training.num-mtp-modules",
        "1",
        "--activation-checkpoint.mode",
        mode,
        "--training.dtype",
        "float32",
        "--training.mixed-precision-param",
        param_dtype,
        "--training.mixed-precision-reduce",
        "float32",
        "--simulation.output-dir",
        str(tmp_path / "unused"),
    ]
    if quantized:
        arguments.append("--extension.quantization.enable-quantized-training")
    workload = main(arguments)
    annotations = workload.iteration.schedule.annotations
    assert annotations["precision_configuration"] == {
        "dtype": "float32",
        "mixed_precision_param": param_dtype,
        "mixed_precision_reduce": "float32",
    }
    assert {item["dtype"] for item in annotations["gradient_validation"].values()} == {"torch.float32"}
    assert annotations["quantization_configuration"]["enabled"] == quantized
    events = annotations["memory_plan"].raw_events
    if quantized:
        assert any(event.raw_op_type == "npu.npu_quant_matmul.default" for event in events)
    if mode != "none":
        assert any(event.execution_kind == "recompute" for event in events)
    assert any(event.raw_op_type == "aten._fused_adamw_.default" for event in events)
    gathers = [event for event in events if "all_gather" in event.raw_op_type]
    assert gathers
    gather_dtypes = {ref.dtype for event in gathers for ref in event.inputs}
    if param_dtype == "float32":
        assert gather_dtypes == {"float32"}
    else:
        assert "bfloat16" in gather_dtypes
        # Native FSDP packs mixed BF16/FP32 groups as a byte view, without quantization.
        assert gather_dtypes <= {"bfloat16", "float32", "uint8"}
    reductions = [event for event in events if "reduce_scatter" in event.raw_op_type]
    assert reductions
    assert {ref.dtype for event in reductions for ref in event.inputs} == {"float32"}
    assert not (tmp_path / "unused").exists()


def test_base_bfloat16_fails_with_native_fsdp_reason_before_capture():
    with pytest.raises(ValueError, match="uniform original parameter dtype"):
        main(["--training.dtype", "bfloat16"])


@pytest.mark.parametrize(
    "values",
    [dict(training_dtype="float16"), dict(mixed_precision_param="16"), dict(mixed_precision_reduce="bfloat16")],
)
def test_precision_values_outside_native_contract_are_rejected(values):
    with pytest.raises(ValueError, match="must be one of"):
        SimulationConfig(**values).validate()
