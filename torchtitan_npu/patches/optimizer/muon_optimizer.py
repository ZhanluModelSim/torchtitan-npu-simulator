# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan distributed_scion reference code,
# https://github.com/rakkit/torchtitan/tree/dist-scion/torchtitan/experiments/distributed_scion/
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import math
import re
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Shard
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    LRScheduler as LRSchedulerConfig,
    Optimizer as OptimizerConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.optimizer.virtual_allocator import (
    ADAMW_STATE_KEYS,
    MUON_STATE_KEYS,
    MuonVirtualAllocator,
)


COEFF_PRIMARY = (3.4445, -4.7750, 2.0315)
# DeepSeek-V4 hybrid Newton-Schulz: first 8 steps use primary coefficients
# for rapid convergence, last 2 steps switch to secondary coefficients
# to stabilize singular values precisely at 1.
COEFF_SECONDARY = (2.0, -1.5, 0.5)


def zeropower_via_newtonschulz5(grad, steps=10, eps=1e-7, hybrid_ns=False):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of grad.
    Quintic iteration whose coefficients are selected to maximize the slope at zero.
    When hybrid_ns=True, uses 8 primary steps + 2 secondary steps.
    """
    if steps >= 100:
        raise ValueError(
            f"Number of NS steps must be < 100 for computational efficiency, got {steps}"
        )
    if len(grad.shape) != 2 and len(grad.shape) != 3:
        raise ValueError(
            f"Gradients must be 2D or 3D tensors to use NS, got shape: {grad.shape}"
        )

    is_2d = len(grad.shape) == 2
    if is_2d:
        grad = grad.unsqueeze(0)

    if len(grad.shape) != 3:
        raise ValueError(
            f"Gradients must be 2D or 3D tensors after unsqueeze, got shape: {grad.shape}"
        )

    original_dtype = grad.dtype
    x = grad.bfloat16()

    rows, cols = x.shape[-2], x.shape[-1]
    transposed = False
    if rows > cols:
        transposed = True
        x = x.transpose(1, 2)

    norm = torch.linalg.norm(x, dim=(-2, -1), keepdim=True)
    x = x / (norm + eps)

    a, b, c = COEFF_PRIMARY
    for i in range(steps):
        if hybrid_ns and i >= 8:
            a, b, c = COEFF_SECONDARY
        gram = torch.bmm(x, x.transpose(1, 2))
        poly = b * gram + c * torch.bmm(gram, gram)
        x = a * x + torch.bmm(poly, x)

    if transposed:
        x = x.transpose(1, 2)

    if is_2d:
        x = x.squeeze(0)

    return x.to(original_dtype)


_EXPERT_KEYWORDS = ("experts", "expert")


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2


def get_param_type(p, param_name, fsdp_enabled, expert_enabled):
    """Classify parameter for distributed Muon routing.

    Uses dimension + name heuristics for robust Expert detection.
    Falls back to FSDP/DDP based on parallelism config.
    """
    is_expert_param = (
        p.ndim == 3
        and expert_enabled
        and any(kw in param_name for kw in _EXPERT_KEYWORDS)
    )
    if is_expert_param:
        return ParamType.Expert
    if fsdp_enabled:
        return ParamType.FSDP
    return ParamType.DDP


def tp_axis(placements, tp_enabled=False):
    """Return the index in placements that belongs to tensor-parallel (TP).

    Heuristics (PyTorch-TP default layouts):
      1. Row-parallel weights -> _StridedShard -> that axis is TP.
      2. Col-parallel weights -> Shard(dim != 0) -> that axis is TP
         (FSDP shards dim-0, so a non-zero dim means TP).
    """
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            return i
    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim != 0:
            return i
    if tp_enabled and len(placements) == 1:
        if isinstance(placements[0], Shard):
            return 0
    return None


def gather_tp_shard(tensor, tp_group, tp_world_size, original_placements):
    tp_mesh_dim = tp_axis(original_placements, True)
    if tp_mesh_dim is None:
        raise RuntimeError("TP mesh dimension not found")
    shard_dim = original_placements[tp_mesh_dim].dim
    output_tensors = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    dist.all_gather(output_tensors, tensor, group=tp_group)
    return torch.cat(output_tensors, dim=shard_dim)


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    splits = torch.arange(full).chunk(world_size)
    if rank >= len(splits):
        dim0 = 0
    else:
        dim0 = len(splits[rank])
    return (dim0, *shape[1:])


@dataclass
class CommContext:
    """Communication context for distributed optimizer operations."""

    rank: int
    world_size: int
    cast_dtype: torch.dtype
    device: torch.device
    tp_group: object = None
    tp_world_size: int = 1
    dp_group: object = None
    dp_replicate_group: object = None
    fsdp_group: object = None


@dataclass
class InitInfo:
    """Initialization parameters for DistributedMuon."""

    lr: float
    momentum: float
    nesterov: bool
    weight_decay: float
    eps: float
    ns_steps: int
    adjust_lr_fn: str
    hybrid_ns: bool
    communication_dtype: torch.dtype
    extra_reduce_for_hsdp: bool
    experts_weights_layout: str


@dataclass
class BucketSpec:
    """Bucket specification for all-gather/scatter operations."""

    start_idx: int
    end_idx: int
    local_updates: dict | None = None
    global_updates: list | None = None
    recv_shapes: list | None = None
    send_shapes: list | None = None


class DistributedMuon(Optimizer):
    """Distributed Muon optimizer.

    Supports:
    - FSDP: all_to_all communication (step_fsdp)
    - DDP: all_gather communication (step_ddp)
    - Expert/MoE: local LMO (step_experts)

    Core components:
    - lmo(): zeropower + normalise_grad
    - Distributed communication (all_to_all, all_gather)
    - Momentum buffer management
    """

    def __init__(
        self,
        params,
        param_names,
        parallel_dims,
        lr=1e-3,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        eps=1e-7,
        communication_dtype=torch.bfloat16,
        extra_reduce_for_hsdp=False,
        experts_weights_layout="G-D_out-D_in",
        adjust_lr_fn="original",
        hybrid_ns=False,
    ):
        self.is_light = False
        self.communication_dtype = communication_dtype
        self.extra_reduce_for_hsdp = extra_reduce_for_hsdp
        self.log_parameters_types = True
        self.adjust_lr_fn = adjust_lr_fn
        self.hybrid_ns = hybrid_ns
        _init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        defaults = self._build_defaults(**_init_kwargs)
        self._init_parallel_info(parallel_dims)
        self._validate_and_set_expert_layout(experts_weights_layout)
        self._log_init_info(self._build_init_info(**_init_kwargs))
        super().__init__(params, defaults)
        self.groups_info = {}
        self.parameters_to_groups = {}
        self._refresh_groups_info()
        self.ddp_params, self.ddp_param_names = [], []
        self.fsdp_params, self.fsdp_param_names = [], []
        self.expert_params, self.expert_param_names = [], []
        self._build_param_lists(param_names)

    @torch.no_grad()
    def get_momentum_or_grad(self, p, momentum, nesterov, gather_to_local=False):
        """Retrieve effective gradient for a parameter.

        Assumes momentum buffer has already been updated in pre-pass.
        """
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            state = self.state.get(p, None)
            if state is None or "momentum_buffer" not in state:
                raise ValueError(
                    "Momentum buffer missing; ensure pre-pass ran before "
                    "calling get_momentum_or_grad."
                )
            buf = state["momentum_buffer"]
            g = buf if not nesterov else g.add(buf, alpha=momentum)

        if gather_to_local and isinstance(g, DTensor):
            from torch.distributed.tensor.placement_types import Replicate

            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()

        return g

    @torch.no_grad()
    def prepare_gradients_and_momentum(self):
        """Fused pre-pass: update momentum buffers for all params with grads.

        buf <- m * buf + (1 - m) * g
        Uses foreach kernels per (device, dtype, momentum).
        """
        buckets = defaultdict(lambda: {"bufs": [], "grads": [], "m": 0.0})

        for group in self.param_groups:
            m = float(group["momentum"])
            use_momentum = (not self.is_light) and (0.0 < m < 1.0)
            if not use_momentum:
                continue

            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is None or not p.requires_grad:
                    continue

                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                key = (g.dtype, m)
                bucket = buckets[key]
                bucket["bufs"].append(buf)  # pyrefly: ignore[missing-attribute]
                bucket["grads"].append(g)  # pyrefly: ignore[missing-attribute]
                bucket["m"] = m

        for (_, m), bucket_data in buckets.items():
            if not bucket_data["bufs"]:
                continue
            torch._foreach_lerp_(  # pyrefly: ignore[no-matching-overload]
                bucket_data["bufs"], bucket_data["grads"], 1.0 - m
            )

    @torch.no_grad()
    def step(self, closure=None):  # pyrefly: ignore[bad-override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Refresh schedulables each step
        self._refresh_groups_info()

        self.prepare_gradients_and_momentum()

        self.step_experts(self.expert_params, self.expert_param_names)
        self.step_ddp(self.ddp_params, self.ddp_param_names)
        self.step_fsdp(self.fsdp_params, self.fsdp_param_names)

        return loss

    def step_ddp(self, ddp_params, ddp_param_names, skip_update=False):
        if len(ddp_params) == 0:
            return

        ctx = self._build_ddp_context(ddp_params)

        bucket_size = ctx.world_size
        total_buckets = (
            math.ceil(len(ddp_params) / bucket_size)
            if ctx.world_size > 1
            else len(ddp_params)
        )

        local_updates = self._precompute_ddp_local_updates(ddp_params, ctx)

        global_updates = [None] * len(ddp_params)

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, len(ddp_params))

            if skip_update:
                continue

            bucket = BucketSpec(
                start_idx=start_idx,
                end_idx=end_idx,
                local_updates=local_updates,
                global_updates=global_updates,
            )
            gathered = self._ddp_allgather_bucket(ddp_params, bucket, ctx)

            for i in range(end_idx - start_idx):
                global_updates[  # pyrefly: ignore[unsupported-operation]
                    start_idx + i
                ] = gathered[i]

        if not skip_update:
            self.update_bucket_params(
                ddp_params, global_updates, 0, len(ddp_params), tp_group=ctx.tp_group
            )

    def step_experts(self, expert_params, expert_param_names, skip_update=False):
        if len(expert_params) == 0:
            return

        transpose = self.experts_need_transpose
        for _, p in enumerate(expert_params):
            lr, nesterov, momentum, wd, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)
            if g is None:
                continue
            u = self.lmo(
                g,
                **param_kwargs,
                transpose_experts=transpose,
                adjust_lr_fn=self.adjust_lr_fn,
                hybrid_ns=self.hybrid_ns,
            )

            if not skip_update:
                self.update_bucket_params([p], [u], 0, 1)

    def step_fsdp(self, fsdp_params, fsdp_param_names, skip_update=False):
        if not fsdp_params:
            return

        ctx = self._build_fsdp_context(fsdp_params)
        if ctx is None:
            return

        global_updates = [None] * len(fsdp_params)

        total_buckets = math.ceil(len(fsdp_params) / ctx.world_size)

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * ctx.world_size
            end_idx = min(start_idx + ctx.world_size, len(fsdp_params))

            u, recv_shapes, send_shapes = self._fsdp_alltoall_and_lmo(
                fsdp_params,
                start_idx,
                end_idx,
                ctx,
            )

            if not skip_update:
                bucket = BucketSpec(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    global_updates=global_updates,
                    recv_shapes=recv_shapes,
                    send_shapes=send_shapes,
                )
                self._fsdp_scatter_updates(u, bucket, ctx)

        if not skip_update:
            self.update_bucket_params(
                fsdp_params,
                global_updates,
                0,
                len(fsdp_params),
                tp_group=ctx.tp_group,
            )

    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_group=None):
        """Apply parameter updates with weight decay and optional TP slicing."""
        slice_params = params[start_idx:end_idx]
        slice_updates = updates[: (end_idx - start_idx)]

        prepared = self._prepare_updates(slice_params, slice_updates, tp_group)

        buckets = defaultdict(
            lambda: {
                "orig": [],
                "locals": [],
                "updates": [],
                "lr": None,
                "wd": None,
                "m": None,
            }
        )

        for p, u in prepared:
            if u is None:
                continue
            lr, _, momentum, wd, _ = self.groups_info[self.parameters_to_groups[id(p)]]
            p_local = p.to_local() if isinstance(p, DTensor) else p

            if p_local.shape != u.shape:
                raise ValueError(
                    f"Shape mismatch: param shard {p_local.shape} vs "
                    f"update slice {u.shape}"
                )

            if u.dtype is not p_local.dtype:
                u = u.to(p_local.dtype)

            key = (p_local.device, p_local.dtype, float(lr), float(wd), float(momentum))
            b = buckets[key]
            b["orig"].append(p)  # pyrefly: ignore[missing-attribute]
            b["locals"].append(p_local)  # pyrefly: ignore[missing-attribute]
            b["updates"].append(u)  # pyrefly: ignore[missing-attribute]
            b["lr"], b["wd"], b["m"] = lr, wd, momentum

        for (_, _, lr, wd, m), data in buckets.items():
            if not data["locals"]:
                continue
            if wd != 0.0:
                torch._foreach_mul_(data["locals"], 1.0 - wd * lr)
            torch._foreach_add_(data["locals"], data["updates"], alpha=-lr)

    def _build_defaults(self, **kwargs):
        return dict(
            lr=kwargs["lr"],
            momentum=kwargs["momentum"],
            weight_decay=kwargs["weight_decay"],
            nesterov=kwargs["nesterov"],
            eps=kwargs["eps"],
            backend_steps=kwargs["ns_steps"],
        )

    def _build_init_info(self, **kwargs):
        return InitInfo(
            lr=kwargs["lr"],
            momentum=kwargs["momentum"],
            nesterov=kwargs["nesterov"],
            weight_decay=kwargs["weight_decay"],
            eps=kwargs["eps"],
            ns_steps=kwargs["ns_steps"],
            adjust_lr_fn=kwargs["adjust_lr_fn"],
            hybrid_ns=kwargs["hybrid_ns"],
            communication_dtype=kwargs["communication_dtype"],
            extra_reduce_for_hsdp=kwargs["extra_reduce_for_hsdp"],
            experts_weights_layout=kwargs["experts_weights_layout"],
        )

    def _build_ddp_context(self, ddp_params):
        dp_group = (
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group()
            if self.dp_replicate_enabled
            else None
        )
        world_size = dp_group.size() if dp_group is not None else 1
        rank = dp_group.rank() if dp_group is not None else 0

        tp_group = (
            self.parallel_dims.get_optional_mesh("tp").get_group()
            if self.tp_enabled
            else None
        )
        tp_world_size = (
            dist.get_world_size(group=tp_group) if tp_group is not None else 1
        )

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype

        return CommContext(
            rank=rank,
            world_size=world_size,
            cast_dtype=cast_dtype,
            device=device,
            tp_group=tp_group,
            tp_world_size=tp_world_size,
            dp_group=dp_group,
        )

    def _build_fsdp_context(self, fsdp_params):
        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        if fsdp_mesh is None:
            logger.warning("step_fsdp called but fsdp mesh is None, skipping")
            return None
        fsdp_group = fsdp_mesh.get_group()
        world_size = dist.get_world_size(fsdp_group)
        rank = dist.get_rank(fsdp_group)
        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype

        tp_group = (
            self.parallel_dims.get_optional_mesh("tp").get_group()
            if self.tp_enabled
            else None
        )
        tp_world_size = dist.get_world_size(group=tp_group) if tp_group else 1
        dp_replicate_group = (
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group()
            if self.dp_replicate_enabled
            else None
        )

        return CommContext(
            rank=rank,
            world_size=world_size,
            cast_dtype=cast_dtype,
            device=device,
            tp_group=tp_group,
            tp_world_size=tp_world_size,
            fsdp_group=fsdp_group,
            dp_replicate_group=dp_replicate_group,
        )

    def _build_param_lists(self, param_names):
        """Classify parameters into FSDP/DDP/Expert buckets."""
        self.ddp_params.clear()
        self.ddp_param_names.clear()
        self.fsdp_params.clear()
        self.fsdp_param_names.clear()
        self.expert_params.clear()
        self.expert_param_names.clear()

        named_params = self._resolve_named_params(param_names, self.param_groups)

        for p_name, p in named_params:
            if not p.requires_grad:
                continue
            ptype = get_param_type(p, p_name, self.fsdp_enabled, self.expert_enabled)
            if ptype == ParamType.DDP:
                self.ddp_params.append(p)
                self.ddp_param_names.append(p_name)
            elif ptype == ParamType.FSDP:
                self.fsdp_params.append(p)
                self.fsdp_param_names.append(p_name)
            elif ptype == ParamType.Expert:
                self.expert_params.append(p)
                self.expert_param_names.append(p_name)

        # Sort FSDP params by numel (big first) to reduce padding
        fsdp_pairs = list(zip(self.fsdp_params, self.fsdp_param_names))
        self.fsdp_params, self.fsdp_param_names = self._sort_pairs_by_numel(
            fsdp_pairs  # pyrefly: ignore[bad-assignment]
        )

        # Sort expert params
        expert_pairs = list(zip(self.expert_params, self.expert_param_names))
        self.expert_params, self.expert_param_names = self._sort_pairs_by_numel(
            expert_pairs,
            key_fn=lambda x: (
                x[0].numel(),
                x[0].shape[1],
            ),  # pyrefly: ignore[bad-assignment]
        )

        self._finalize_ddp_param_lists()

    def _collect_bucket_grads(
        self,
        fsdp_params,
        start_idx,
        end_idx,
        ctx: CommContext,
    ):
        grads_send_list, send_shapes = [], []
        target_shape, param_kwargs_me = None, None

        for i in range(ctx.world_size):
            p_idx = start_idx + i
            if p_idx < end_idx:
                p = fsdp_params[p_idx]
                _, nesterov, momentum, _, param_kwargs = self.groups_info[
                    self.parameters_to_groups[id(p)]
                ]
                g = self.get_momentum_or_grad(p, momentum, nesterov)

                if g is None:
                    ref_p = fsdp_params[p_idx]
                    dummy = torch.zeros(
                        ref_p.to_local().shape, dtype=ctx.cast_dtype, device=ctx.device
                    )
                    grads_send_list.append(dummy)
                    send_shapes.append(dummy.shape)
                    continue

                g_local = self._gather_grad_local(g, ctx.tp_group, ctx.tp_world_size)
                grads_send_list.append(g_local.to(dtype=ctx.cast_dtype))
                send_shapes.append(g_local.shape)

                if i == ctx.rank:
                    target_shape = p.shape
                    param_kwargs_me = param_kwargs
            else:
                ref_p = fsdp_params[end_idx - 1]
                dummy = torch.zeros(
                    ref_p.to_local().shape, dtype=ctx.cast_dtype, device=ctx.device
                )
                grads_send_list.append(dummy)
                send_shapes.append(dummy.shape)

        return grads_send_list, send_shapes, target_shape, param_kwargs_me

    def _ddp_allgather_bucket(self, ddp_params, bucket: BucketSpec, ctx: CommContext):
        if bucket.local_updates is None:
            raise ValueError("bucket.local_updates must not be None")
        if bucket.global_updates is None:
            raise ValueError("bucket.global_updates must not be None")
        my_idx = bucket.start_idx + ctx.rank
        if my_idx < bucket.end_idx:
            send_u = bucket.local_updates.get(my_idx)
            if send_u is None:
                ref = ddp_params[my_idx]
                send_u = torch.zeros(ref.shape, dtype=ctx.cast_dtype, device=ctx.device)
        else:
            ref = ddp_params[bucket.end_idx - 1]
            send_u = torch.zeros(ref.shape, dtype=ctx.cast_dtype, device=ctx.device)

        if ctx.dp_group is not None and ctx.world_size > 1:
            gathered = []
            pad_buffer = None
            for i in range(ctx.world_size):  # pyrefly: ignore[bad-assignment]
                param_idx = bucket.start_idx + i
                if param_idx < len(ddp_params):
                    ref = ddp_params[param_idx]
                    if bucket.global_updates[param_idx] is None:
                        bucket.global_updates[  # pyrefly: ignore[bad-assignment]
                            param_idx
                        ] = torch.empty(
                            ref.shape, dtype=ctx.cast_dtype, device=ctx.device
                        )
                    recv = bucket.global_updates[param_idx]
                else:
                    ref = ddp_params[bucket.end_idx - 1]
                    if pad_buffer is None or pad_buffer.shape != ref.shape:
                        pad_buffer = torch.empty(
                            ref.shape, dtype=ctx.cast_dtype, device=ctx.device
                        )
                    recv = pad_buffer
                gathered.append(recv)
            dist.all_gather(gathered, send_u, group=ctx.dp_group)
            return gathered

        return [send_u]

    def _finalize_ddp_param_lists(self):
        ddp_pairs = list(zip(self.ddp_params, self.ddp_param_names))
        ddp_pairs.sort(key=lambda x: x[0].numel(), reverse=True)
        dp_group = (
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group()
            if self.dp_replicate_enabled
            else None
        )
        w = dp_group.size() if dp_group is not None else 1
        ddp_pairs = self._snake_interleave(ddp_pairs, w)
        if ddp_pairs:
            (
                self.ddp_params,  # pyrefly: ignore[bad-assignment]
                self.ddp_param_names,  # pyrefly: ignore[bad-assignment]
            ) = list(zip(*ddp_pairs))
        else:
            self.ddp_params = []
            self.ddp_param_names = []

        if self.log_parameters_types:
            logger.info(
                f"DistributedMuon param counts: "
                f"fsdp={len(self.fsdp_params)} | expert={len(self.expert_params)} | "
                f"ddp={len(self.ddp_params)}"
            )
            self.log_parameters_types = False

    def _fsdp_alltoall_and_lmo(
        self,
        fsdp_params,
        start_idx,
        end_idx,
        ctx: CommContext,
    ):
        (
            grads_send_list,
            send_shapes,
            target_shape,
            param_kwargs_me,
        ) = self._collect_bucket_grads(
            fsdp_params,
            start_idx,
            end_idx,
            ctx,
        )

        if target_shape is None:
            ref_idx = end_idx - 1
            target_shape = fsdp_params[ref_idx].shape
            param_kwargs_me = self.groups_info[
                self.parameters_to_groups[id(fsdp_params[ref_idx])]
            ][-1]

        recv_shapes = [
            calculate_shard_shape(target_shape, r, ctx.world_size)
            for r in range(ctx.world_size)
        ]
        recv_list_grads = [
            torch.empty(s, dtype=ctx.cast_dtype, device=ctx.device) for s in recv_shapes
        ]

        dist.all_to_all(recv_list_grads, grads_send_list, group=ctx.fsdp_group)

        full_g = torch.cat(recv_list_grads, dim=0)
        u = self.lmo(  # pyrefly: ignore[missing-argument]
            full_g,
            **param_kwargs_me,  # pyrefly: ignore[bad-unpacking]
            adjust_lr_fn=self.adjust_lr_fn,
            hybrid_ns=self.hybrid_ns,
        )

        if ctx.dp_replicate_group and self.extra_reduce_for_hsdp:
            dist.all_reduce(u, group=ctx.dp_replicate_group, op=dist.ReduceOp.AVG)

        return u, recv_shapes, send_shapes

    def _fsdp_scatter_updates(self, u, bucket: BucketSpec, ctx: CommContext):
        if bucket.recv_shapes is None:
            raise ValueError("bucket.recv_shapes must not be None")
        if bucket.send_shapes is None:
            raise ValueError("bucket.send_shapes must not be None")
        if bucket.global_updates is None:
            raise ValueError("bucket.global_updates must not be None")
        split_rows = [s[0] for s in bucket.recv_shapes]
        updates_send_list = list(torch.split(u, split_rows, dim=0))
        recv_list_updates = [
            torch.empty(s, dtype=ctx.cast_dtype, device=ctx.device)
            for s in bucket.send_shapes
        ]

        dist.all_to_all(recv_list_updates, updates_send_list, group=ctx.fsdp_group)

        for i in range(bucket.end_idx - bucket.start_idx):
            bucket.global_updates[bucket.start_idx + i] = recv_list_updates[i]

    def _gather_grad_local(self, g, tp_group, tp_world_size):
        if not isinstance(g, DTensor):
            return g
        original_placements = g.placements
        tp_mesh_dim = tp_axis(original_placements)
        if tp_group and tp_mesh_dim is not None:
            return gather_tp_shard(
                g.to_local(),
                tp_group,
                tp_world_size,
                original_placements,
            )
        return g.to_local()

    def _init_parallel_info(self, parallel_dims):
        self.world_mesh = parallel_dims.world_mesh
        self.parallel_dims = parallel_dims
        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled

        if self.dp_replicate_enabled or self.fsdp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                self.is_dp_rank_0 = dist.get_rank(loss_mesh.get_group()) == 0
            else:
                self.is_dp_rank_0 = dist.get_rank() == 0
        else:
            self.is_dp_rank_0 = dist.get_rank() == 0

    def _log_init_info(self, info: InitInfo):
        logger.info(
            f"DistributedMuon optimizer "
            f"is enabled with world_mesh={self.world_mesh} | "
            f"fsdp_enabled={self.fsdp_enabled} | "
            f"EP={self.expert_enabled} | TP={self.tp_enabled} | "
            f"DP={self.dp_replicate_enabled}"
        )
        logger.info(
            f"DistributedMuon hyperparams: "
            f"lr={info.lr} | momentum={info.momentum} | nesterov={info.nesterov} | "
            f"weight_decay={info.weight_decay} | eps={info.eps} | "
            f"ns_steps={info.ns_steps} | "
            f"adjust_lr_fn={info.adjust_lr_fn} | hybrid_ns={info.hybrid_ns} | "
            f"communication_dtype={info.communication_dtype} | "
            f"extra_reduce_for_hsdp={info.extra_reduce_for_hsdp} | "
            f"experts_weights_layout={info.experts_weights_layout}"
        )

    def _precompute_ddp_local_updates(self, ddp_params, ctx: CommContext):
        local_updates = {}
        for i in range(ctx.rank, len(ddp_params), ctx.world_size):
            p = ddp_params[i]
            _, nesterov, momentum, _, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(p)]
            ]
            g = self.get_momentum_or_grad(p, momentum, nesterov)
            if g is None:
                local_updates[i] = None
                continue
            if isinstance(g, DTensor) and ctx.tp_group is not None:
                g = gather_tp_shard(
                    g.to_local(), ctx.tp_group, ctx.tp_world_size, g.placements
                )
            else:
                g = g.to_local() if isinstance(g, DTensor) else g
            u = self.lmo(
                g.to(dtype=ctx.cast_dtype),
                **param_kwargs,
                adjust_lr_fn=self.adjust_lr_fn,
                hybrid_ns=self.hybrid_ns,
            )
            local_updates[i] = u
        return local_updates

    def _prepare_updates(self, slice_params, slice_updates, tp_group):
        if tp_group is None:
            return list(zip(slice_params, slice_updates))
        prepared = []
        for p, u in zip(slice_params, slice_updates):
            if u is None:
                prepared.append((p, None))
                continue
            u = self._slice_update_for_tp(p, u, tp_group)
            prepared.append((p, u))
        return prepared

    def _refresh_groups_info(self):
        for group_idx, group in enumerate(self.param_groups):
            param_kwargs = {
                "eps": group["eps"],
                "backend_steps": group["backend_steps"],
            }
            self.groups_info[group_idx] = [
                group["lr"],
                group["nesterov"],
                group["momentum"],
                group["weight_decay"],
                param_kwargs,
            ]
            for param in group["params"]:
                self.parameters_to_groups[id(param)] = group_idx

    @torch.no_grad()
    def _slice_update_for_tp(self, p, u, tp_group):
        if not isinstance(p, DTensor):
            return u
        p_local = p.to_local()
        placements = p.placements
        tp_mesh_dim = tp_axis(placements, tp_enabled=(u.shape == p.shape))
        if tp_mesh_dim is None:
            return u
        shard_dim = placements[tp_mesh_dim].dim  # pyrefly: ignore[missing-attribute]
        if u.shape == p_local.shape:
            return u
        chunk_size = p_local.shape[shard_dim]
        start = tp_group.rank() * chunk_size
        slicer = [slice(None)] * u.dim()
        slicer[shard_dim] = slice(start, start + chunk_size)
        return u[tuple(slicer)]

    def _validate_and_set_expert_layout(self, experts_weights_layout):
        if experts_weights_layout not in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ]:
            raise ValueError(
                f"Unknown experts weights layout: {experts_weights_layout}"
            )
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"

    @staticmethod
    @torch.no_grad()
    def lmo(
        g,
        eps,
        backend_steps,
        transpose_experts=False,
        adjust_lr_fn="original",
        hybrid_ns=False,
    ):
        """LMO: Low-orthogonal Matrix Operation (zeropower + normalise).

        Supports: 2D (linear), 3D (MoE expert).
        """
        g = g.to_local() if isinstance(g, DTensor) else g

        def _orth_and_norm(x):
            x = zeropower_via_newtonschulz5(
                x, steps=backend_steps, eps=eps, hybrid_ns=hybrid_ns
            )
            x = DistributedMuon.normalise_grad(x, eps=eps, adjust_lr_fn=adjust_lr_fn)
            return x

        if g.ndim == 2:
            return _orth_and_norm(g)
        elif g.ndim == 3:
            if g.shape[0] > 0:
                g = g.transpose(1, 2) if transpose_experts else g
                g = _orth_and_norm(g)
                g = g.transpose(1, 2) if transpose_experts else g
            return g
        else:
            raise ValueError(f"lmo expects 2D or 3D grad, got shape: {g.shape}")

    @staticmethod
    @torch.no_grad()
    def normalise_grad(g, eps, adjust_lr_fn="original"):
        """Normalise gradient tensor with spectral norm factor."""
        a, b = g.size(-2), g.size(-1)
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            g = g * math.sqrt(max(1, a / b))
        elif adjust_lr_fn == "match_rms_adamw":
            g = g * 0.18 * math.sqrt(max(a, b))  # deepseekV4 use 0.18
        return g

    @staticmethod
    def _resolve_named_params(param_names, param_groups):
        """Build (name, param) pairs from param_names and param_groups."""
        all_params = []
        for group in param_groups:
            all_params.extend(group["params"])

        if len(param_names) == len(all_params):
            return list(zip(param_names, all_params))

        named_params = []
        for group in param_groups:
            group_pnames = group.get("param_names", None)
            if group_pnames is not None:
                named_params.extend(zip(group_pnames, group["params"]))
            else:
                for p in group["params"]:
                    named_params.append((f"param_{id(p)}", p))
        return named_params

    @staticmethod
    def _snake_interleave(pairs, w):
        """Snake-interleave pairs across w DP replicas."""
        if w <= 1:
            return pairs
        # fmt: off
        blocks = [pairs[i:i + w] for i in range(0, len(pairs), w)]
        # fmt: on
        for b, blk in enumerate(blocks):
            if b % 2 == 1:
                blk.reverse()
        return [p for blk in blocks for p in blk]

    @staticmethod
    def _sort_pairs_by_numel(pairs, key_fn=None):
        """Sort (param, name) pairs and unzip back into two lists."""
        if not pairs:
            return [], []
        sort_key = key_fn if key_fn else lambda x: x[0].numel()
        pairs.sort(key=sort_key, reverse=True)
        params, names = list(zip(*pairs))
        return list(params), list(names)


_MUON_EXCLUDED_KEYWORDS = ("embed", "lm_head", "output")


def _classify_param(name, p):
    if not p.requires_grad:
        return None
    if p.ndim == 2:
        if any(kw in name for kw in _MUON_EXCLUDED_KEYWORDS):
            return "adamw"
        return "muon"
    if p.ndim == 3:
        return "muon"
    return "adamw"


def _split_parameters_for_muon(
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims | None = None,
) -> tuple[list[nn.Parameter], list[str], list[nn.Parameter], list[str]]:
    """Split parameters into Muon (2D/3D) and AdamW (others) groups.

    Dimensionality routing (per design docs):
    - 2D params → Muon (except embeddings, lm_head, and output layers → AdamW)
    - 3D params → Muon (expert weights)
    - others → AdamW
    """
    muon_params = []
    muon_param_names = []
    adamw_params = []
    adamw_param_names = []

    for model in model_parts:
        for name, p in model.named_parameters():
            category = _classify_param(name, p)
            if category == "muon":
                muon_params.append(p)
                muon_param_names.append(name)
            elif category == "adamw":
                adamw_params.append(p)
                adamw_param_names.append(name)

    return muon_params, muon_param_names, adamw_params, adamw_param_names


def _get_muon_lr_config(
    optimizer_config: OptimizerConfig,
    base_lr: float,
) -> tuple[float, str | None]:
    """Calculate Muon's effective learning rate and adjustment mode.

    Returns:
        Tuple of (muon_lr, muon_adjust_lr_fn)
    """
    muon_adjust_lr_fn = (
        optimizer_config.muon_adjust_lr_fn  # pyrefly: ignore[missing-attribute]
    )
    muon_lr = getattr(optimizer_config, "muon_lr", None)

    if muon_adjust_lr_fn == "original" and muon_lr is not None:
        return float(muon_lr), muon_adjust_lr_fn

    if muon_adjust_lr_fn == "match_rms_adamw" and muon_lr is not None:
        logger.warning(
            f"[Muon] muon_lr={muon_lr} is ignored when "
            f"muon_adjust_lr_fn='match_rms_adamw'. Using base lr={base_lr} instead."
        )
    return base_lr, muon_adjust_lr_fn


def _build_muon_kwargs(
    muon_lr: float,
    weight_decay: float,
    optimizer_config: OptimizerConfig,
    muon_adjust_lr_fn: str | None,
) -> dict[str, Any]:
    """Build kwargs for DistributedMuon constructor."""
    muon_kwargs = {
        "lr": muon_lr,
        "weight_decay": weight_decay,
        "momentum": optimizer_config.muon_momentum,  # pyrefly: ignore[missing-attribute]
        "nesterov": optimizer_config.muon_enable_nesterov,  # pyrefly: ignore[missing-attribute]
        "ns_steps": optimizer_config.muon_ns_steps,  # pyrefly: ignore[missing-attribute]
        "eps": optimizer_config.eps,
        "adjust_lr_fn": muon_adjust_lr_fn,
        "hybrid_ns": optimizer_config.muon_hybrid_ns,  # pyrefly: ignore[missing-attribute]
    }
    return muon_kwargs


def _build_adamw_kwargs(
    lr: float,
    weight_decay: float,
    optimizer_config: OptimizerConfig,
) -> dict[str, Any]:
    """Build kwargs for torch.optim.AdamW constructor."""
    optim_implementation = optimizer_config.implementation
    if optim_implementation not in ["fused", "foreach", "for-loop"]:
        raise ValueError(
            f"Invalid implementation '{optim_implementation}'. "
            f"Must be one of: 'fused', 'foreach', 'for-loop'"
        )
    return {
        "lr": lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }


def build_muon_hybrid_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
    virtual_allocator: bool = False,
) -> OptimizersContainer:
    """Build Muon hybrid optimizer: DistributedMuon (2D/3D) + AdamW (non-2D/3D)."""
    lr = optimizer_config.lr
    weight_decay = optimizer_config.weight_decay

    muon_lr, muon_adjust_lr_fn = _get_muon_lr_config(optimizer_config, lr)

    (
        muon_params,
        muon_param_names,
        adamw_params,
        adamw_param_names,
    ) = _split_parameters_for_muon(model_parts, parallel_dims)

    logger.info(
        f"[MuonAdamW] Muon optimizer parameters ({len(muon_param_names)}): {muon_param_names}"
    )
    logger.info(
        f"[MuonAdamW] AdamW optimizer parameters ({len(adamw_param_names)}): {adamw_param_names}"
    )

    muon_kwargs = _build_muon_kwargs(
        muon_lr, weight_decay, optimizer_config, muon_adjust_lr_fn
    )
    adamw_kwargs = _build_adamw_kwargs(lr, weight_decay, optimizer_config)

    extra_rules = getattr(optimizer_config, "extra_param_group_split_rules", None)

    param_groups = _build_muon_param_groups(
        muon_params, muon_param_names, muon_kwargs, extra_rules
    )
    muon = DistributedMuon(
        param_groups,
        muon_param_names,
        parallel_dims,
        **muon_kwargs,
    )
    adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)

    if virtual_allocator:
        va = MuonVirtualAllocator(
            pp_rank=getattr(parallel_dims, "pp_rank", 0),
            pp_size=getattr(parallel_dims, "pp_size", 1),
        )

        logger.info("[VirtualAllocator] Enabled virtual allocator with swap memory.")

        return VirtualMuonHybridOptimizersContainer(
            model_parts,
            [muon, adamw],
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            virtual_allocator=va,
        )

    return MuonHybridOptimizersContainer(
        model_parts, [muon, adamw], muon_adjust_lr_fn=muon_adjust_lr_fn
    )


def _build_muon_param_groups(
    params: list[nn.Parameter],
    param_names: list[str],
    default_kwargs: dict[str, Any],
    extra_rules: list[dict] | None,
) -> list[dict]:
    """Build parameter groups for DistributedMuon with extra_param_group_split_rules.

    Matches parameters by name against regex rules, creating separate param groups
    with overridden settings (lr, backend, etc.).
    """
    default_config = {
        "lr": default_kwargs.get("lr"),
        "weight_decay": default_kwargs.get("weight_decay"),
        "momentum": default_kwargs.get("momentum"),
        "nesterov": default_kwargs.get("nesterov"),
        "eps": default_kwargs.get("eps", 1e-7),
        "backend_steps": default_kwargs.get("ns_steps", 5),
    }

    # Build rule configs
    rule_configs = []
    for rule in extra_rules or []:
        rc = default_config.copy()
        rc.update(rule)
        if "str_match" not in rc:
            logger.warning(
                "extra_param_group_split_rules entry missing 'str_match', skipping"
            )
            continue
        rc["param_str_match"] = rc.pop("str_match")
        rule_configs.append(rc)

    # Classify params
    param_dict = OrderedDict(zip(param_names, params))
    groups = []

    for rc in rule_configs:
        str_match = rc.pop("param_str_match")
        filter_fn = functools.partial(re.search, str_match)
        matched_names = [n for n in param_dict.keys() if filter_fn(n)]
        if not matched_names:
            logger.warning(f'No parameters found for str_match "{str_match}"')
            continue
        group = {
            "params": [param_dict.pop(n) for n in matched_names],
            "param_names": matched_names,
        }
        group.update(rc)  # pyrefly: ignore[no-matching-overload]
        groups.append(group)

    # Remaining params go to default group
    remaining_names = list(param_dict.keys())
    if remaining_names:
        default_group = {
            "params": [param_dict[n] for n in remaining_names],
            "param_names": remaining_names,
        }
        default_group.update(default_config)  # pyrefly: ignore[no-matching-overload]
        groups.insert(0, default_group)

    return groups


class MuonHybridOptimizersContainer(OptimizersContainer):
    """Container for Muon + AdamW hybrid optimizers.

    Key difference from upstream OptimizersContainer:
    - Upstream: model_parts[i] <-> optimizers[i] (1:1 pairing)
    - This class: each optimizer manages a subset of params from all model_parts

    When muon_adjust_lr_fn == "original":
    - Muon and AdamW use different base_lr
    - Must be used with MuonLRSchedulersContainer

    When muon_adjust_lr_fn == "match_rms_adamw":
    - Muon and AdamW use the same base_lr
    - Can use standard LRSchedulersContainer

    state_dict/load_state_dict use double loop over each optimizer and model_part.
    DCP APIs automatically filter to only process params managed by each optimizer.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: list[Optimizer],
        muon_adjust_lr_fn: str | None = None,
    ) -> None:
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.muon_adjust_lr_fn = muon_adjust_lr_fn
        all_params = []
        for model in model_parts:
            all_params.extend(p for p in model.parameters() if p.requires_grad)
        Optimizer.__init__(self, all_params, {})

    def __iter__(self) -> Iterator[Optimizer]:
        """Return iterator over sub-optimizers for MuonLRSchedulersContainer."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers (Muon + AdamW = 2)."""
        return len(self.optimizers)

    @property
    def muon_optimizer(self) -> Optimizer:
        """Get the Muon optimizer."""
        return self.optimizers[0]

    @property
    def adamw_optimizer(self) -> Optimizer | None:
        """Get the AdamW optimizer, or None if no AdamW params exist."""
        return self.optimizers[1] if len(self.optimizers) > 1 else None

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Save state for all optimizers using double loop over optimizer x model_part."""
        merged = {}
        for opt in self.optimizers:
            for model in self.model_parts:
                sd = get_optimizer_state_dict(
                    model,
                    opt,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
                merged.update(sd)
        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for all optimizers using double loop over optimizer x model_part."""
        for opt in self.optimizers:
            for model in self.model_parts:
                set_optimizer_state_dict(
                    model,
                    opt,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )


class VirtualMuonHybridOptimizersContainer(MuonHybridOptimizersContainer):
    """Virtual optimizer container for Muon + AdamW hybrid.

    Allocates optimizer states in swap memory (CPU) and handles
    state swap during step() to reduce GPU memory usage.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: list[Optimizer],
        muon_adjust_lr_fn: str | None = None,
        virtual_allocator: MuonVirtualAllocator | None = None,
    ):
        super().__init__(model_parts, optimizers, muon_adjust_lr_fn)
        self.virtual_allocator = virtual_allocator
        self._step_count = 0

    def step(self, *args, **kwargs) -> None:
        self._swap_states_to_device()
        super().step(*args, **kwargs)
        self._swap_states_to_host()
        self._step_count += 1
        if self._step_count == 1 and self.virtual_allocator is not None:
            self.virtual_allocator.compute_theoretical_swap_size(
                self.optimizers, MUON_STATE_KEYS, ADAMW_STATE_KEYS
            )
            self.virtual_allocator.print_swap_summary()

    def state_dict(self) -> dict[str, Any]:
        self._swap_states_to_device()
        sd = super().state_dict()
        self._swap_states_to_host()
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._swap_states_to_device()
        super().load_state_dict(state_dict)
        self._swap_states_to_host()

    def _swap_param_states_to_device(self, state, p, keys):
        for key in keys:
            if key not in state:
                continue
            swap_tensor = state[key]
            if swap_tensor is not None and swap_tensor.device.type == "cpu":
                global_shape = p.shape if isinstance(p, DTensor) else None
                state[
                    key
                ] = self.virtual_allocator.copy2device(  # pyrefly: ignore[missing-attribute]
                    swap_tensor,
                    ref_tensor=p,
                    global_shape=global_shape,  # pyrefly: ignore[bad-argument-type]
                )

    def _swap_optim_states_to_device(self, optim, keys):
        for group in optim.param_groups:
            for p in group["params"]:
                state = optim.state.get(p, {})
                if state:
                    self._swap_param_states_to_device(state, p, keys)

    def _for_each_optim_state(self, fn):
        if self.virtual_allocator is None:
            return
        for optim_idx, optim in enumerate(self.optimizers):
            keys = MUON_STATE_KEYS if optim_idx == 0 else ADAMW_STATE_KEYS
            fn(optim, keys)

    def _swap_states_to_device(self):
        self._for_each_optim_state(self._swap_optim_states_to_device)

    def _swap_param_states_to_host(self, state, keys):
        for key in keys:
            if key not in state:
                continue
            device_tensor = state[key]
            if device_tensor is not None and device_tensor.device.type != "cpu":
                state[
                    key
                ] = self.virtual_allocator.copy2swap(  # pyrefly: ignore[missing-attribute]
                    device_tensor
                )

    def _swap_optim_states_to_host(self, optim, keys):
        for group in optim.param_groups:
            for p in group["params"]:
                state = optim.state.get(p, {})
                if state:
                    self._swap_param_states_to_host(state, keys)

    def _swap_states_to_host(self):
        self.virtual_allocator.actually_swap_size = (  # pyrefly: ignore[missing-attribute]
            0
        )
        self._for_each_optim_state(self._swap_optim_states_to_host)


class MuonLRSchedulersContainer:
    """LR Scheduler container for Muon hybrid optimizers.

    Creates independent LambdaLR schedulers for Muon and AdamW,
    ensuring each maintains its own base_lr.

    Key difference from upstream LRSchedulersContainer:
    - Upstream: assumes all optimizers use the same base_lr
    - This class: allows Muon and AdamW to have different base_lr

    Note: state_dict only saves the first scheduler's state (last_epoch),
    consistent with upstream behavior since Muon and AdamW share the same
    lr curve, only differing in base_lr.
    """

    def __init__(
        self,
        optimizers: MuonHybridOptimizersContainer,
        lr_lambda: Callable,
    ) -> None:
        if len(optimizers) != 2:
            raise ValueError(
                f"MuonHybridOptimizersContainer must have 2 optimizers, got {len(optimizers)}"
            )

        # Create independent LambdaLR for Muon and AdamW
        self.schedulers = [
            LambdaLR(optimizers.muon_optimizer, lr_lambda),
            LambdaLR(
                optimizers.adamw_optimizer,  # pyrefly: ignore[bad-argument-type]
                lr_lambda,
            ),
        ]

        logger.info("[MuonLRSchedulersContainer] Created 2 schedulers")
        logger.info(f"  Muon scheduler base_lrs: {self.schedulers[0].base_lrs}")
        logger.info(f"  AdamW scheduler base_lrs: {self.schedulers[1].base_lrs}")
        logger.info(
            f"  Muon param_groups lr: {[pg['lr'] for pg in optimizers.muon_optimizer.param_groups]}"
        )
        logger.info(
            f"  AdamW param_groups lr: "
            f"{[pg['lr'] for pg in optimizers.adamw_optimizer.param_groups]}"  # pyrefly: ignore[missing-attribute]
        )

    def __iter__(self):
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Step all schedulers synchronously."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Save only the first scheduler's state (last_epoch is shared)."""
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load last_epoch for all schedulers without overwriting base_lrs.

        Critical: Only load last_epoch, do not overwrite base_lrs.
        base_lrs are set from each optimizer during LambdaLR construction.
        Muon and AdamW have different base_lrs that must remain independent.

        PyTorch LambdaLR.load_state_dict would overwrite base_lrs, so we
        manually set last_epoch instead of calling load_state_dict.
        """
        last_epoch = state_dict["last_epoch"]
        for scheduler in self.schedulers:
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = [
                scheduler.base_lrs[i] * scheduler.lr_lambdas[i](last_epoch)
                for i in range(len(scheduler.base_lrs))
            ]


def _build_lr_lambda_from_config(
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> Callable:
    """Build lr_lambda function from scheduler config.

    Extracted from build_muon_lr_schedulers to avoid code duplication.
    """
    warmup_steps = int(lr_scheduler_config.warmup_steps)

    if warmup_steps > training_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) exceed total training steps ({training_steps}). "
            f"Adjusting warmup steps to {training_steps}."
        )
        warmup_steps = training_steps

    if lr_scheduler_config.decay_ratio is not None:
        decay_steps = round(training_steps * lr_scheduler_config.decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps

    stable_steps = training_steps + 1 - warmup_steps - decay_steps
    lr_decay_type = lr_scheduler_config.decay_type
    min_lr_factor = lr_scheduler_config.min_lr_factor

    def linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_decay_type: str,
        min_lr_factor: float,
    ):
        warmup_stable_steps = warmup_steps + stable_steps
        if current_step < warmup_steps:
            current_step += 1
            curr_adjustment = float(current_step / warmup_steps)
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        else:
            current_step += 1
            progress = float(current_step - warmup_stable_steps) / decay_steps
            if lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif lr_decay_type == "cosine":
                curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
            curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment
        return curr_adjustment

    return functools.partial(
        linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_decay_type,
        min_lr_factor=min_lr_factor,
    )


def build_muon_lr_schedulers(
    optimizers: MuonHybridOptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> MuonLRSchedulersContainer | Any:
    """Build LR scheduler for MuonHybridOptimizersContainer.

    Routes to different scheduler types based on muon_adjust_lr_fn:
    - "original": MuonLRSchedulersContainer (different base_lr for Muon and AdamW)
    - Other: Standard LRSchedulersContainer (same base_lr for both)

    Args:
        optimizers: MuonHybridOptimizersContainer instance
        lr_scheduler_config: LR scheduler configuration
        training_steps: Total training steps

    Returns:
        MuonLRSchedulersContainer or LRSchedulersContainer
    """
    lr_lambda = _build_lr_lambda_from_config(lr_scheduler_config, training_steps)

    if optimizers.muon_adjust_lr_fn == "original":
        return MuonLRSchedulersContainer(optimizers, lr_lambda)
    else:
        return LRSchedulersContainer(optimizers, lr_lambda)
