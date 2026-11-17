# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Lifecycle test: the container owns and releases every offload runtime.

Builds a real ``CpuOffloadOptimizersContainer`` over an FSDP-offloaded model,
verifies the resources it registers (hooks, clip consumer, pipelines), closes
it, and checks that everything is released: staging executor thread gone,
module pipelines cleared, hooks disabled symmetrically, close idempotent, and
a second build/close cycle leaves no extra threads behind.
"""

import logging
import os
import threading

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy
from torchtitan.components.optimizer import ParamGroupConfig

import torchtitan_npu.extensions.cpu_offload.runtime as clip_state
from torchtitan_npu.extensions.distributed import grad_accum, grad_clip
from torchtitan_npu.override.common.optimizer import CpuOffloadOptimizersContainer

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
        nn.Linear(128, 64),
    )


def staging_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("cpu-staging"))


def main() -> None:
    dist.init_process_group("hccl")
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch_npu.npu.current_device())
    mesh = init_device_mesh("npu", (1,))

    base_threads = staging_threads()

    for cycle in range(2):
        model = build_model().to(device)
        for block in model:
            fully_shard(block, mesh=mesh, offload_policy=CPUOffloadPolicy())
        fully_shard(model, mesh=mesh, offload_policy=CPUOffloadPolicy())

        x = torch.randn(8, 64, device=device)
        out = model(x)
        out.sum().backward()
        torch_npu.npu.synchronize()

        container = CpuOffloadOptimizersContainer(
            CpuOffloadOptimizersContainer.Config(
                param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 1e-2})]
            ),
            model_parts=[model],
        )

        # Force one pageable D2H so the staging executor thread exists.
        container._staging.submit_d2h(torch.ones(8, device=device), torch.ones(8), track=False)
        container._staging.wait()

        # Registered state: hooks on, consumer counted, container staging alive.
        assert grad_accum._cpu_offload_enabled(), "hooks not enabled after construction"
        active = clip_state.get_active_channel()
        assert active is not None and active.has_consumer(), "clip channel not active/registered"
        assert staging_threads() >= base_threads + 1, "staging executor thread missing after pageable D2H"

        container.close()

        # Released state: everything back to baseline.
        assert not grad_accum._cpu_offload_enabled(), "hooks not disabled after close"
        assert clip_state.get_active_channel() is None, "clip channel still active after close"
        assert not grad_accum._PIPELINES_BY_DEVICE, "FSDP pipelines not cleared after close"
        assert not grad_clip._PIPELINES, "clip pipelines not cleared after close"
        assert container._staging._executor is None, "staging executor not shut down"
        assert container._staging._pending == [], "staging transfers still pending"
        deadline = threading.Event()
        for _ in range(100):
            if staging_threads() == base_threads:
                break
            deadline.wait(0.1)
        assert staging_threads() == base_threads, (
            f"staging threads leaked after close: {staging_threads()} != {base_threads}"
        )
        container.close()  # idempotent
        logging.info(f"cycle {cycle}: build → register → close → release all OK")

    del container
    logging.info("LIFECYCLE TEST PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
