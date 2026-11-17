# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Regression test for the bounded-clip norm over stale slot tails (#808-12).

``_bounded_local_norm`` reuses fixed-size chunk buffers without zeroing.
The norm must only cover the elements actually written by the H2D copy;
summing the stale tail over-estimates the norm and over-clips gradients.
This test poisons both slot buffers with a large known value, then clips a
gradient whose last chunk is partial and compares against the exact norm.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import Shard

from torchtitan_npu.extensions.distributed import grad_clip

import pytest

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason='NPU not available'
)


# Shrink the internal chunk size so the tail test exercises partial chunks.
grad_clip._CHUNK_BYTES = 1024
CHUNK_NUMEL = 1024 // 4  # 256 elements


logging.basicConfig(level=logging.INFO, force=True)


def as_dtensor(values: torch.Tensor, mesh) -> DTensor:
    # Wrap the CPU local tensor directly (like FSDP's to_sharded_dtensor);
    # DTensor.from_local would move it to the NPU mesh device.
    local = values.to(torch.float32)
    spec = DTensorSpec(
        mesh,
        [Shard(0)],
        tensor_meta=TensorMeta(shape=tuple(local.shape), stride=local.stride(), dtype=local.dtype),
    )
    return DTensor(local, spec, requires_grad=local.requires_grad)


def main() -> None:
    dist.init_process_group("hccl")
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch_npu.npu.current_device())
    mesh = init_device_mesh("npu", (1,))

    # Prime both slot buffers with full chunks of a large value, so any
    # later partial chunk faces a known, heavily inflated stale tail.
    poison = [
        torch.full((CHUNK_NUMEL,), 100.0),
        torch.full((CHUNK_NUMEL,), 100.0),
    ]
    grad_clip._bounded_local_norm([as_dtensor(p, mesh) for p in poison], device, 2.0, None)

    # Victim: 300 elements -> chunks of 256 + 44. The 44-element chunk leaves
    # 212 stale 100.0 elements in the slot; the buggy norm would be ~1456
    # instead of sqrt(300) ~ 17.32.
    victim = torch.ones(CHUNK_NUMEL + 44)
    total, ready = grad_clip._bounded_local_norm([as_dtensor(victim, mesh)], device, 2.0, None)
    ready.synchronize()
    got = float(total.item())
    expected = float(torch.linalg.vector_norm(victim))
    assert abs(got - expected) < 1e-3, (
        f"bounded local norm polluted by stale slot tail: got {got:.4f}, expected {expected:.4f}"
    )
    logging.info(f"bounded local norm OK: {got:.4f} == {expected:.4f}")

    # inf-norm variant: a single large victim value must survive the poison.
    victim_inf = torch.cat([torch.full((CHUNK_NUMEL + 44,), 1.0), torch.full((5,), 3.0)])
    total_inf, ready_inf = grad_clip._bounded_local_norm([as_dtensor(victim_inf, mesh)], device, float("inf"), None)
    ready_inf.synchronize()
    got_inf = float(total_inf.item())
    assert abs(got_inf - 3.0) < 1e-5, f"bounded inf-norm polluted by stale slot tail: got {got_inf}, expected 3.0"
    logging.info(f"bounded inf-norm OK: {got_inf} == 3.0")

    dist.destroy_process_group()
    logging.info("BOUNDED CLIP NORM REGRESSION TEST PASSED")


if __name__ == "__main__":
    main()
