# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Upstream issue (feature request to torch_npu):
# https://gitcode.com/Ascend/pytorch/issues/4964

"""Initialize CPU DTensor parameters on their NPU mesh without CPU collectives.

Under CPU offload the model parameters materialize as DTensors whose local
shards live on CPU while the mesh is NPU. The upstream
``Module._init_param`` would run the initializer directly on such a
parameter, driving mesh collectives from CPU tensors. The installed wrapper
stages a same-shape NPU work DTensor, runs the initializer there, and copies
the result back to the CPU local shard.

Installation is explicit: ``TrainerEx`` calls :func:`install` before model
materialization when ``--training.enable-cpu-offload`` is set, because the
patched path is only reachable under CPU offload. Runs without CPU offload
never import this module and keep the upstream initializer unchanged.
"""

from __future__ import annotations

import functools
from typing import cast

import torch
import torch_npu
from torch.distributed.tensor import DTensor
from torchtitan.protocols.module import Module

_ORIGINAL_INIT_PARAM = Module._init_param
_PATCHED = "_torchtitan_npu_cpu_dtensor_init"


@functools.wraps(_ORIGINAL_INIT_PARAM)
def _init_param_on_mesh_device(
    self: Module,
    name: str,
    param: torch.nn.Parameter,
) -> None:
    if not (isinstance(param, DTensor) and param.device.type == "cpu" and param.device_mesh.device_type == "npu"):
        return _ORIGINAL_INIT_PARAM(self, name, param)

    cpu_parameter = cast("DTensor", param)
    local = cpu_parameter.to_local()
    local_work = torch.empty_strided(
        local.size(),
        local.stride(),
        dtype=local.dtype,
        device=torch.device("npu", torch_npu.npu.current_device()),
    )
    work = DTensor.from_local(
        local_work,
        cpu_parameter.device_mesh,
        cpu_parameter.placements,
        shape=cpu_parameter.size(),
        stride=cpu_parameter.stride(),
        run_check=False,
    )
    _ORIGINAL_INIT_PARAM(self, name, work)  # pyrefly: ignore [bad-argument-type]
    with torch.no_grad():
        local.copy_(local_work)
    torch.autograd.graph.increment_version(cpu_parameter)
    return None


def install() -> None:
    """Install the CPU-DTensor initializer wrapper once per process.

    Called by ``TrainerEx`` before ``super().__init__()`` when the user
    selects ``--training.enable-cpu-offload``: ``Module._init_param`` fires
    inside ``init_weights`` (after ``parallelize_fn`` applies the offload
    policy), which is earlier than the optimizer container exists. Because
    this module is only imported on that path, the captured
    ``_ORIGINAL_INIT_PARAM`` is always the pristine upstream method.
    """
    if getattr(Module, _PATCHED, False):
        return
    Module._init_param = _init_param_on_mesh_device
    setattr(Module, _PATCHED, True)
