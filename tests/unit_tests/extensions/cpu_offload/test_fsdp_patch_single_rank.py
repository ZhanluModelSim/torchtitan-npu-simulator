# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Single-rank equivalence test for the CPU-offload FSDP wrappers.

Runs a small model under fully_shard with CPUOffloadPolicy on one NPU rank
and two microbatches, then compares forward outputs and CPU-canonical
gradients against a non-FSDP reference. Instruments the wrapper internals to
confirm the copy-only unshard prefetch and the NPU gradient-accumulation
pipeline are actually exercised, and re-checks correctness with the hooks
disabled (pure delegation).
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy

from torchtitan_npu.extensions.distributed import grad_accum as fsdp_patch

import pytest

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason='NPU not available'
)


logging.basicConfig(level=logging.INFO, force=True)


def build_model() -> nn.Module:
    torch.manual_seed(1234)
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.GELU(),
        nn.Linear(128, 64),
    )


def make_inputs(device: torch.device) -> list[torch.Tensor]:
    torch.manual_seed(99)
    return [torch.randn(8, 64, device=device) for _ in range(2)]


def loss_for(out: torch.Tensor) -> torch.Tensor:
    weight = torch.arange(out.numel(), device=out.device, dtype=out.dtype).view_as(out)
    return (out * weight * 1e-3).sum()


def instrument(calls: dict) -> None:
    """Count invocations of the offload fast paths and their modes."""

    original_copy = fsdp_patch._copy_only_all_gather
    original_accumulate = fsdp_patch.accumulate_cpu_grad

    def counting_copy(*args, **kwargs):
        calls["copy_only"] += 1
        return original_copy(*args, **kwargs)

    def counting_accumulate(*args, **kwargs):
        calls["accumulate"] += 1
        calls["sync_accumulate"] += bool(kwargs.get("synchronous"))
        return original_accumulate(*args, **kwargs)

    fsdp_patch._copy_only_all_gather = counting_copy
    fsdp_patch.accumulate_cpu_grad = counting_accumulate


def run_case(enabled: bool, calls: dict) -> None:
    device = torch.device("npu", torch_npu.npu.current_device())
    mesh = init_device_mesh("npu", (1,))

    model_fsdp = build_model().to(device)
    for block in model_fsdp:
        fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
    fully_shard(model_fsdp, mesh=mesh, offload_policy=CPUOffloadPolicy())

    model_ref = build_model().to(device)
    for parameter in model_ref.parameters():
        parameter.grad = None

    inputs = make_inputs(device)

    outs = []
    for x in inputs:
        out = model_fsdp(x)
        outs.append(out.detach().clone())
        loss_for(out).backward()
    ref_outs = []
    for x in inputs:
        out = model_ref(x)
        ref_outs.append(out.detach().clone())
        loss_for(out).backward()
    torch_npu.npu.synchronize()

    label = "hooks ON" if enabled else "hooks OFF"
    for i, (out, ref) in enumerate(zip(outs, ref_outs, strict=False)):
        max_diff = (out - ref).abs().max().item()
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), f"{label}: forward output {i} differs (max {max_diff})"
        logging.info(f"[{label}] forward output {i}: max_diff={max_diff:.3e}")

    for (name, parameter), (_, ref_parameter) in zip(
        model_fsdp.named_parameters(), model_ref.named_parameters(), strict=False
    ):
        grad = parameter.grad
        assert grad is not None, f"{label}: missing grad for {name}"
        local = grad.to_local() if isinstance(grad, torch.distributed.tensor.DTensor) else grad
        assert local.device.type == "cpu", f"{label}: grad for {name} on {local.device}"
        ref_grad = ref_parameter.grad.detach().to("cpu")
        max_diff = (local - ref_grad).abs().max().item()
        assert torch.allclose(local, ref_grad, atol=1e-5, rtol=1e-5), f"{label}: grad {name} differs (max {max_diff})"
        if enabled:
            logging.info(f"[{label}] grad {name}: max_diff={max_diff:.3e} pinned={local.is_pinned()}")


def main() -> None:
    dist.init_process_group("hccl")
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    fsdp_patch.install()

    # Sanity: the wrappers must be installed and the originals reachable.
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group

    assert getattr(_collectives.foreach_reduce, "_torchtitan_npu_fsdp_grad_accum_v1", False)
    assert getattr(_param_group.FSDPParamGroup.unshard, "_torchtitan_npu_cpu_offload_prefetch_v2", False)
    assert getattr(_param_group.FSDPParamGroup.wait_for_unshard, "_torchtitan_npu_cpu_offload_prefetch_v2", False)
    assert _collectives.foreach_reduce.__wrapped__ is not None
    logging.info(f"wrappers installed: {_collectives.foreach_reduce.__name__}")

    calls = {"copy_only": 0, "accumulate": 0, "sync_accumulate": 0}
    instrument(calls)

    # Case 1: hooks disabled -> everything must delegate to upstream.
    run_case(enabled=False, calls=calls)
    assert calls["copy_only"] == 0, "copy-only path ran with hooks disabled"
    assert calls["accumulate"] == 0, "NPU accumulation ran with hooks disabled"
    logging.info(f"delegation case OK: {calls}")

    # Case 2: hooks enabled -> fast paths engaged, results still correct.
    fsdp_patch.register_cpu_offload_hooks()
    run_case(enabled=True, calls=calls)
    assert calls["copy_only"] > 0, "copy-only unshard prefetch never ran"
    assert calls["accumulate"] > 0, "NPU gradient accumulation never ran"
    assert calls["sync_accumulate"] == 0, "accumulation fell back to synchronous"
    logging.info(f"offload case OK: {calls}")

    fsdp_patch.clear()
    dist.destroy_process_group()
    logging.info("SINGLE-RANK EQUIVALENCE TEST PASSED")


if __name__ == "__main__":
    main()
