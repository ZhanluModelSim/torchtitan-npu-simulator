# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Intercepts torch.distributed collective calls so they can run on
meta-device tensors under a FakeProcessGroup without touching the real
c10d dispatcher (see design doc §2 finding #4 and §5.2). Generalizes the
`is_fake_process_group` short-circuit pattern already used by
`torchtitan_npu.converters.kernels.moe_dispatch.NpuExpertParallel` to every
collective entry point (FSDP2 all-gather/reduce-scatter, TP all-reduce, DP
grad all-reduce, broadcast)."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from torchtitan_npu.distributed.process_group import is_fake_process_group
from torchtitan_npu.simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes

_event_counter = itertools.count()


@dataclass
class CommEvent:
    event_id: str
    comm_primitive: str
    group_name: str
    world_size: int
    tensor_shape: tuple[int, ...]
    dtype: str
    volume_bytes: int


class _NoOpWork:
    """Minimal stand-in for `torch.distributed.Work`, so callers that use
    the `async_op=True` idiom (call `.wait()` on the return value) do not
    crash when we skip the real collective."""

    def wait(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def is_completed(self) -> bool:
        return True


class CommEventRecorder:
    def __init__(self) -> None:
        self.events: list[CommEvent] = []

    def record(self, comm_primitive: str, group: object, tensor: torch.Tensor) -> None:
        dtype_str = dtype_to_str(tensor.dtype)
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1  # type: ignore[arg-type]
        self.events.append(
            CommEvent(
                event_id=f"comm_{next(_event_counter)}",
                comm_primitive=comm_primitive,
                group_name=_group_name(group),
                world_size=world_size,
                tensor_shape=tuple(int(d) for d in tensor.shape),
                dtype=dtype_str,
                volume_bytes=tensor_volume_bytes(tuple(tensor.shape), dtype_str),
            )
        )


def _group_name(group: object) -> str:
    if group is None:
        return "default"
    name = getattr(group, "group_name", None)
    return str(name) if name is not None else "default"


@contextmanager
def capture_fake_collectives() -> Iterator[CommEventRecorder]:
    """Monkeypatch the legacy (`torch.distributed.*`) and functional
    (`torch.distributed._functional_collectives.*`) collective APIs for the
    duration of the context.

    Legacy APIs receive a real `ProcessGroup` (or `None`) as their `group`
    argument, so they defensively check `is_fake_process_group(group)` and
    fall back to the real implementation when it is not fake (keeps this
    module safe to import even outside a simulation run).

    Functional-collective APIs (used internally by DTensor/FSDP2) accept a
    `ProcessGroup`, `DeviceMesh`, list of ranks, or group-name string as
    `group` -- resolving all of those reliably is fragile (see design doc
    §2 finding #4 discussion). Because this context manager is only ever
    active for the full duration of one simulated training step, and the
    simulator always runs entirely under a fake backend (never a mix of
    real and fake groups), the functional-collective patches always treat
    calls made while the context is active as fake, unconditionally.
    """
    recorder = CommEventRecorder()

    orig_all_reduce = dist.all_reduce
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_all_to_all_single = dist.all_to_all_single
    orig_broadcast = dist.broadcast
    orig_barrier = dist.barrier

    orig_funcol_all_reduce = funcol.all_reduce
    orig_funcol_all_gather_tensor = funcol.all_gather_tensor
    orig_funcol_reduce_scatter_tensor = funcol.reduce_scatter_tensor
    orig_funcol_all_to_all_single = funcol.all_to_all_single

    def patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
        recorder.record("allreduce", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)
        recorder.record("allgather", group, input_tensor)
        return _NoOpWork() if async_op else None

    def patched_reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)
        recorder.record("reduce_scatter", group, input)
        return _NoOpWork() if async_op else None

    def patched_all_to_all_single(  # noqa: ANN001
        output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
    ):
        if not is_fake_process_group(group):
            return orig_all_to_all_single(
                output, input, output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes, group=group, async_op=async_op,
            )
        recorder.record("all_to_all", group, input)
        return _NoOpWork() if async_op else None

    def patched_broadcast(tensor, src=0, group=None, async_op=False, group_src=None):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_broadcast(tensor, src=src, group=group, async_op=async_op, group_src=group_src)
        recorder.record("broadcast", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_barrier(group=None, async_op=False, device_ids=None):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)
        return _NoOpWork() if async_op else None

    def patched_funcol_all_reduce(self_tensor, reduceOp, group, tag=""):  # noqa: ANN001, N803
        recorder.record("allreduce", None, self_tensor)
        return self_tensor.clone()

    def patched_funcol_all_gather_tensor(self_tensor, gather_dim, group, tag=""):  # noqa: ANN001
        recorder.record("allgather", None, self_tensor)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        out_shape = list(self_tensor.shape)
        out_shape[gather_dim] *= world_size
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    def patched_funcol_reduce_scatter_tensor(self_tensor, reduceOp, scatter_dim, group, tag=""):  # noqa: ANN001, N803
        recorder.record("reduce_scatter", None, self_tensor)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        out_shape = list(self_tensor.shape)
        out_shape[scatter_dim] //= world_size
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    def patched_funcol_all_to_all_single(self_tensor, output_split_sizes, input_split_sizes, group, tag=""):  # noqa: ANN001
        recorder.record("all_to_all", None, self_tensor)
        out_shape = list(self_tensor.shape)
        if output_split_sizes:
            out_shape[0] = int(sum(output_split_sizes))
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    dist.all_reduce = patched_all_reduce
    dist.all_gather_into_tensor = patched_all_gather_into_tensor
    dist.reduce_scatter_tensor = patched_reduce_scatter_tensor
    dist.all_to_all_single = patched_all_to_all_single
    dist.broadcast = patched_broadcast
    dist.barrier = patched_barrier
    funcol.all_reduce = patched_funcol_all_reduce
    funcol.all_gather_tensor = patched_funcol_all_gather_tensor
    funcol.reduce_scatter_tensor = patched_funcol_reduce_scatter_tensor
    funcol.all_to_all_single = patched_funcol_all_to_all_single

    try:
        yield recorder
    finally:
        dist.all_reduce = orig_all_reduce
        dist.all_gather_into_tensor = orig_all_gather_into_tensor
        dist.reduce_scatter_tensor = orig_reduce_scatter_tensor
        dist.all_to_all_single = orig_all_to_all_single
        dist.broadcast = orig_broadcast
        dist.barrier = orig_barrier
        funcol.all_reduce = orig_funcol_all_reduce
        funcol.all_gather_tensor = orig_funcol_all_gather_tensor
        funcol.reduce_scatter_tensor = orig_funcol_reduce_scatter_tensor
        funcol.all_to_all_single = orig_funcol_all_to_all_single
