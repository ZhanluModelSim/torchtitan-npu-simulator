# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan distributed_scion reference code,
# https://github.com/rakkit/torchtitan/tree/dist-scion/torchtitan/experiments/distributed_scion/
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import logging
import math
import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, _StridedShard
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer

logger = logging.getLogger("torchtitan")


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
        raise ValueError(f"Number of NS steps must be < 100 for computational efficiency, got {steps}")
    if len(grad.shape) != 2 and len(grad.shape) != 3:
        raise ValueError(f"Gradients must be 2D or 3D tensors to use NS, got shape: {grad.shape}")

    is_2d = len(grad.shape) == 2
    if is_2d:
        grad = grad.unsqueeze(0)

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
    is_expert_param = p.ndim == 3 and expert_enabled and any(kw in param_name for kw in _EXPERT_KEYWORDS)
    if is_expert_param:
        return ParamType.Expert
    if fsdp_enabled:
        return ParamType.FSDP
    return ParamType.DDP


def validate_expert_shards_on_dim0(p, param_name):
    if not isinstance(p, DTensor):
        return

    for placement in p.placements:
        if isinstance(placement, (Shard, _StridedShard)) and placement.dim != 0:
            raise RuntimeError(
                f"DistributedMuon expert route only supports expert DTensor shards on dim 0, "
                f"but parameter {param_name} has placement {placement} in placements {p.placements}. "
                "This usually means expert weights are sharded inside each expert matrix; "
                "step_experts would run local Muon on incomplete matrices."
            )


def tp_axis(placements, tp_enabled=False):
    """Return the index in placements that belongs to tensor-parallel (TP).

    Heuristics (PyTorch-TP default layouts):
      1. Row-parallel weights -> _StridedShard -> that axis is TP.
      2. Col-parallel weights -> Shard(dim != 0) -> that axis is TP
         (FSDP shards dim-0, so a non-zero dim means TP).
      3. Pure-TP (no FSDP): any single Shard on a TP mesh is TP.
      4. Replicate placements are not TP-sharded -> return None.
    """
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            return i
    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim != 0:
            return i
    if tp_enabled and len(placements) == 1 and isinstance(placements[0], Shard):
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
    dim0 = 0 if rank >= len(splits) else len(splits[rank])
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
class SwapMergeContext:
    """Swap-related context for FSDP merge group processing."""

    merge_buckets: int
    use_swap: bool
    to_device_stream: Any = None
    to_host_stream: Any = None


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
        self._param_type_map: dict[int, ParamType] = {}
        self._expert_param_ids: set[int] = set()
        self._fsdp_param_ids: set[int] = set()
        self._ddp_param_ids: set[int] = set()
        self.ddp_params, self.ddp_param_names = [], []
        self.fsdp_params, self.fsdp_param_names = [], []
        self.expert_params, self.expert_param_names = [], []
        self._build_param_lists(param_names)

        self._swap_enabled: bool = False
        self._swap_container: Any = None
        self._swap_merge_buckets: int = 1

        self._device_module: Any = None
        self._swap_to_device_stream: Any = None
        self._swap_to_host_stream: Any = None

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
            x = zeropower_via_newtonschulz5(x, steps=backend_steps, eps=eps, hybrid_ns=hybrid_ns)
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
            return list(zip(param_names, all_params, strict=True))

        named_params = []
        for group in param_groups:
            group_pnames = group.get("param_names", None)
            if group_pnames is not None:
                named_params.extend(zip(group_pnames, group["params"], strict=True))
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
        params, names = list(zip(*pairs, strict=True))
        return list(params), list(names)

    @torch.no_grad()
    def get_momentum_or_grad(self, p, momentum, nesterov):
        """Retrieve effective gradient for a parameter.

        Assumes momentum buffer has already been updated in pre-pass.
        """
        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            state = self.state.get(p, None)
            if state is None or state.get("momentum_buffer") is None:
                raise ValueError("Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad.")
            buf = state["momentum_buffer"]
            g = buf if not nesterov else g.add(buf, alpha=momentum)

        return g

    @torch.no_grad()
    def prepare_gradients_and_momentum(self, skip_param_types=None):
        """Fused pre-pass: update momentum buffers for all params with grads.

        buf <- m * buf + (1 - m) * g
        Uses foreach kernels per (device, dtype, momentum).
        When skip_param_types is provided, params of those types are skipped
        (their momentum update happens later in step_experts/step_fsdp).
        """
        if skip_param_types is None:
            skip_param_types = set()

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

                ptype = self._get_param_type(p)
                if skip_param_types and ptype in skip_param_types:
                    continue

                state = self.state.setdefault(p, {})
                if "momentum_buffer" not in state or state["momentum_buffer"] is None:
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

        self._refresh_groups_info()

        skip_param_types = None
        if self._swap_enabled and self._swap_container is not None:
            skip_param_types = {ParamType.Expert, ParamType.FSDP}
        self.prepare_gradients_and_momentum(skip_param_types=skip_param_types)

        self.step_experts(self.expert_params, self.expert_param_names)
        self.step_ddp(self.ddp_params, self.ddp_param_names)
        self.step_fsdp(self.fsdp_params, self.fsdp_param_names)

        return loss

    def step_ddp(self, ddp_params, ddp_param_names):
        if len(ddp_params) == 0:
            return

        ctx = self._build_ddp_context(ddp_params)

        bucket_size = ctx.world_size
        total_buckets = math.ceil(len(ddp_params) / bucket_size) if ctx.world_size > 1 else len(ddp_params)

        local_updates = self._precompute_ddp_local_updates(ddp_params, ctx)

        swap_merge_buckets = getattr(self, "_swap_merge_buckets", 1)
        num_merge_groups = math.ceil(total_buckets / swap_merge_buckets)

        for merge_idx in range(num_merge_groups):
            merge_start_bucket = merge_idx * swap_merge_buckets
            merge_end_bucket = min(merge_start_bucket + swap_merge_buckets, total_buckets)
            merge_start_idx = merge_start_bucket * bucket_size
            merge_end_idx = min(merge_end_bucket * bucket_size, len(ddp_params))
            merge_params = ddp_params[merge_start_idx:merge_end_idx]
            merge_updates: list[Any] = [None] * len(merge_params)

            for bucket_idx in range(merge_start_bucket, merge_end_bucket):
                start_idx = bucket_idx * bucket_size
                end_idx = min(start_idx + bucket_size, len(ddp_params))

                gathered = self._ddp_allgather_bucket(
                    ddp_params,
                    start_idx,
                    end_idx,
                    local_updates,
                    ctx,
                )
                for i in range(end_idx - start_idx):
                    merge_updates[start_idx - merge_start_idx + i] = gathered[i]
                del gathered

            self.update_bucket_params(
                merge_params,
                merge_updates,
                tp_group=ctx.tp_group,
            )
            del merge_updates

    def step_experts(self, expert_params, expert_param_names):
        if len(expert_params) == 0:
            return

        transpose = self.experts_need_transpose
        use_swap = self._swap_enabled and self._swap_container is not None

        swap_to_device_stream: Any = None
        swap_to_host_stream: Any = None
        swap_state: Any = None

        if use_swap:
            swap_to_device_stream = self._swap_to_device_stream
            swap_to_host_stream = self._swap_to_host_stream

        for p, p_name in zip(expert_params, expert_param_names, strict=True):
            validate_expert_shards_on_dim0(p, p_name)
            _, nesterov, momentum, _, param_kwargs = self.groups_info[self.parameters_to_groups[id(p)]]

            if use_swap:
                swap_state = self._swap_h2d_single(p, swap_to_device_stream, swap_to_host_stream)
                self._update_momentum_single(p, momentum)

            g = self.get_momentum_or_grad(p, momentum, nesterov)
            if g is None:
                if use_swap:
                    self._swap_d2h_single(swap_state, swap_to_host_stream)
                continue

            u = self.lmo(
                g,
                **param_kwargs,
                transpose_experts=transpose,
                adjust_lr_fn=self.adjust_lr_fn,
                hybrid_ns=self.hybrid_ns,
            )

            self.update_bucket_params([p], [u])

            if use_swap:
                self._swap_d2h_single(swap_state, swap_to_host_stream)

        if use_swap:
            self._device_module.current_stream().wait_stream(
                swap_to_host_stream
            )  # pyrefly: ignore[missing-attribute,unbound-name]

    def step_fsdp(self, fsdp_params, fsdp_param_names):
        if not fsdp_params:
            return

        ctx = self._build_fsdp_context(fsdp_params)
        if ctx is None:
            return

        total_buckets = math.ceil(len(fsdp_params) / ctx.world_size)
        use_swap = self._swap_enabled and self._swap_container is not None

        swap_merge_buckets = getattr(self, "_swap_merge_buckets", 1)
        num_merge_groups = math.ceil(total_buckets / swap_merge_buckets)

        swap_ctx = SwapMergeContext(
            merge_buckets=swap_merge_buckets,
            use_swap=use_swap,
            to_device_stream=self._swap_to_device_stream if use_swap else None,
            to_host_stream=self._swap_to_host_stream if use_swap else None,
        )

        for merge_idx in range(num_merge_groups):
            self._process_fsdp_merge_group(
                merge_idx,
                fsdp_params,
                ctx,
                swap_ctx,
                total_buckets,
            )

        if use_swap:
            self._device_module.current_stream().wait_stream(
                swap_ctx.to_host_stream
            )  # pyrefly: ignore[missing-attribute,unbound-name]

    def update_bucket_params(self, params, updates, tp_group=None):
        """Apply parameter updates with weight decay and optional TP slicing."""
        prepared = self._prepare_updates(params, updates, tp_group)

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
            if u is None or getattr(p, "grad", None) is None:
                continue
            lr, _, momentum, wd, _ = self.groups_info[self.parameters_to_groups[id(p)]]
            p_local = p.to_local() if isinstance(p, DTensor) else p

            if p_local.shape != u.shape:
                raise ValueError(f"Shape mismatch: param shard {p_local.shape} vs update slice {u.shape}")

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
            torch._foreach_add_(  # pyrefly: ignore [no-matching-overload]
                data["locals"], data["updates"], alpha=-lr
            )

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

    @torch.no_grad()
    def _update_momentum_single(self, p, momentum):
        m = float(momentum)
        if not (0.0 < m < 1.0):
            return
        g = getattr(p, "grad", None)
        if g is None or not p.requires_grad:
            return
        state = self.state.get(p, {})
        buf = state.get("momentum_buffer", None)
        if buf is None:
            raise RuntimeError(
                f"momentum_buffer is None for param {p.shape} after swap_to_device. Check swap state initialization."
            )
        buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
        g_local = g.to_local() if isinstance(g, DTensor) else g
        buf_local.lerp_(g_local, 1.0 - m)

    def _get_param_type(self, p):
        ptype = self._param_type_map.get(id(p), None)
        if ptype is not None:
            return ptype
        pid = id(p)
        if pid in self._expert_param_ids:
            ptype = ParamType.Expert
        elif pid in self._fsdp_param_ids:
            ptype = ParamType.FSDP
        elif pid in self._ddp_param_ids:
            ptype = ParamType.DDP
        else:
            ptype = ParamType.DDP
        self._param_type_map[pid] = ptype
        return ptype

    def _swap_h2d_single(self, param, swap_to_device_stream, swap_to_host_stream):
        dev = self._device_module  # pyrefly: ignore[missing-attribute]
        swap_state = self._swap_container.get_swap_state(id(param))  # pyrefly: ignore[missing-attribute]
        with dev.stream(swap_to_device_stream):  # pyrefly: ignore[missing-attribute]
            swap_to_device_stream.wait_stream(swap_to_host_stream)
            swap_state.swap_to_device(stream=swap_to_device_stream)
        swap_state.wait_swap()
        return swap_state

    def _swap_d2h_single(self, swap_state, swap_to_host_stream):
        dev = self._device_module  # pyrefly: ignore[missing-attribute]
        compute_done = dev.current_stream().record_event()  # pyrefly: ignore[missing-attribute]
        with dev.stream(swap_to_host_stream):  # pyrefly: ignore[missing-attribute]
            swap_to_host_stream.wait_event(compute_done)
            swap_state.swap_to_host(stream=swap_to_host_stream)

    def _swap_h2d_merge_group(self, params, swap_to_device_stream, swap_to_host_stream):
        dev = self._device_module  # pyrefly: ignore[missing-attribute]
        with dev.stream(swap_to_device_stream):  # pyrefly: ignore[missing-attribute]
            swap_to_device_stream.wait_stream(swap_to_host_stream)
            for p in params:
                swap_state = self._swap_container.get_swap_state(id(p))  # pyrefly: ignore[missing-attribute]
                swap_state.swap_to_device(stream=swap_to_device_stream)
        h2d_done_event = swap_to_device_stream.record_event()
        dev.current_stream().wait_event(h2d_done_event)  # pyrefly: ignore[missing-attribute]
        for p in params:
            swap_state = self._swap_container.get_swap_state(id(p))  # pyrefly: ignore[missing-attribute]
            swap_state.wait_swap()

    def _swap_momentum_update(self, params):
        for p in params:
            momentum = self.groups_info[self.parameters_to_groups[id(p)]][2]
            self._update_momentum_single(p, momentum)

    def _swap_d2h_merge_group(self, params, swap_to_host_stream):
        dev = self._device_module  # pyrefly: ignore[missing-attribute]
        compute_done = dev.current_stream().record_event()  # pyrefly: ignore[missing-attribute]
        with dev.stream(swap_to_host_stream):  # pyrefly: ignore[missing-attribute]
            swap_to_host_stream.wait_event(compute_done)
            for p in params:
                swap_state = self._swap_container.get_swap_state(id(p))  # pyrefly: ignore[missing-attribute]
                swap_state.swap_to_host(stream=swap_to_host_stream)

    def _process_fsdp_merge_group(
        self,
        merge_idx,
        fsdp_params,
        ctx,
        swap_ctx: SwapMergeContext,
        total_buckets,
    ):
        merge_start_bucket = merge_idx * swap_ctx.merge_buckets
        merge_end_bucket = min(merge_start_bucket + swap_ctx.merge_buckets, total_buckets)
        merge_start_idx = merge_start_bucket * ctx.world_size
        merge_end_idx = min(merge_end_bucket * ctx.world_size, len(fsdp_params))
        merge_params = fsdp_params[merge_start_idx:merge_end_idx]
        merge_updates: list[Any] = [None] * len(merge_params)

        if swap_ctx.use_swap:
            self._swap_h2d_merge_group(merge_params, swap_ctx.to_device_stream, swap_ctx.to_host_stream)

        for bucket_idx in range(merge_start_bucket, merge_end_bucket):
            start_idx = bucket_idx * ctx.world_size
            end_idx = min(start_idx + ctx.world_size, len(fsdp_params))

            if swap_ctx.use_swap:
                self._swap_momentum_update(fsdp_params[start_idx:end_idx])

            u, recv_shapes, send_shapes = self._fsdp_alltoall_and_lmo(
                fsdp_params,
                start_idx,
                end_idx,
                ctx,
            )
            recv_list_updates = self._fsdp_scatter_updates(
                u,
                recv_shapes,
                send_shapes,
                ctx,
            )
            for i in range(end_idx - start_idx):
                merge_updates[start_idx - merge_start_idx + i] = recv_list_updates[i]
            del recv_list_updates

        self.update_bucket_params(merge_params, merge_updates, tp_group=ctx.tp_group)
        del merge_updates

        if swap_ctx.use_swap:
            self._swap_d2h_merge_group(merge_params, swap_ctx.to_host_stream)

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
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group() if self.dp_replicate_enabled else None
        )
        world_size = dp_group.size() if dp_group is not None else 1
        rank = dp_group.rank() if dp_group is not None else 0

        tp_group = self.parallel_dims.get_optional_mesh("tp").get_group() if self.tp_enabled else None
        tp_world_size = dist.get_world_size(group=tp_group) if tp_group is not None else 1

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

        tp_group = self.parallel_dims.get_optional_mesh("tp").get_group() if self.tp_enabled else None
        tp_world_size = dist.get_world_size(group=tp_group) if tp_group else 1
        dp_replicate_group = (
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group() if self.dp_replicate_enabled else None
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
        fsdp_pairs = list(zip(self.fsdp_params, self.fsdp_param_names, strict=True))
        self.fsdp_params, self.fsdp_param_names = self._sort_pairs_by_numel(
            fsdp_pairs  # pyrefly: ignore[bad-assignment]
        )

        # Sort expert params
        expert_pairs = list(zip(self.expert_params, self.expert_param_names, strict=True))
        self.expert_params, self.expert_param_names = self._sort_pairs_by_numel(
            expert_pairs,
            key_fn=lambda x: (
                x[0].numel(),
                x[0].shape[1],
            ),  # pyrefly: ignore[bad-assignment]
        )

        self._finalize_ddp_param_lists()

        self._expert_param_ids = {id(p) for p in self.expert_params}
        self._fsdp_param_ids = {id(p) for p in self.fsdp_params}
        self._ddp_param_ids = {id(p) for p in self.ddp_params}

    def _collect_bucket_grads(
        self,
        fsdp_params,
        start_idx,
        end_idx,
        ctx: CommContext,
    ) -> tuple[list[Any], list[Any], Any | None, dict[str, Any] | None]:
        grads_send_list, send_shapes = [], []
        target_shape, param_kwargs_me = None, None

        for i in range(ctx.world_size):
            p_idx = start_idx + i
            if p_idx < end_idx:
                p = fsdp_params[p_idx]
                _, nesterov, momentum, _, param_kwargs = self.groups_info[self.parameters_to_groups[id(p)]]
                g = self.get_momentum_or_grad(p, momentum, nesterov)

                if i == ctx.rank:
                    target_shape = p.shape
                    param_kwargs_me = param_kwargs

                if g is None:
                    ref_p = fsdp_params[p_idx]
                    dummy = torch.zeros(ref_p.to_local().shape, dtype=ctx.cast_dtype, device=ctx.device)
                    grads_send_list.append(dummy)
                    send_shapes.append(dummy.shape)
                    continue

                g_local = self._gather_grad_local(g, ctx.tp_group, ctx.tp_world_size)
                grads_send_list.append(g_local.to(dtype=ctx.cast_dtype))
                send_shapes.append(g_local.shape)
            else:
                ref_p = fsdp_params[end_idx - 1]
                dummy = torch.zeros(ref_p.to_local().shape, dtype=ctx.cast_dtype, device=ctx.device)
                grads_send_list.append(dummy)
                send_shapes.append(dummy.shape)

        return grads_send_list, send_shapes, target_shape, param_kwargs_me

    def _ddp_allgather_bucket(self, ddp_params, start_idx, end_idx, local_updates, ctx: CommContext):
        send_u = self._ddp_bucket_send_update(ddp_params, start_idx, end_idx, local_updates, ctx)

        if ctx.dp_group is not None and ctx.world_size > 1:
            gathered = self._ddp_bucket_gather_buffers(ddp_params, start_idx, end_idx, ctx)
            dist.all_gather(gathered, send_u, group=ctx.dp_group)
            return gathered

        return [send_u]

    def _ddp_bucket_send_update(self, ddp_params, start_idx, end_idx, local_updates, ctx: CommContext):
        my_idx = start_idx + ctx.rank
        if my_idx < end_idx:
            send_u = local_updates.get(my_idx)
            if send_u is not None:
                return send_u
            ref = ddp_params[my_idx]
        else:
            ref = ddp_params[end_idx - 1]
        return torch.zeros(ref.shape, dtype=ctx.cast_dtype, device=ctx.device)

    def _ddp_bucket_gather_buffers(self, ddp_params, start_idx, end_idx, ctx: CommContext):
        gathered: list[Any] = []
        pad_buffer: Any = None
        for i in range(ctx.world_size):
            param_idx = start_idx + i
            buffer, pad_buffer = self._ddp_bucket_gather_buffer(ddp_params, param_idx, end_idx, ctx, pad_buffer)
            gathered.append(buffer)
        return gathered

    def _ddp_bucket_gather_buffer(self, ddp_params, param_idx, end_idx, ctx: CommContext, pad_buffer):
        if param_idx < len(ddp_params):
            ref = ddp_params[param_idx]
            return torch.empty(ref.shape, dtype=ctx.cast_dtype, device=ctx.device), pad_buffer

        ref = ddp_params[end_idx - 1]
        if pad_buffer is None or pad_buffer.shape != ref.shape:
            pad_buffer = torch.empty(ref.shape, dtype=ctx.cast_dtype, device=ctx.device)
        return pad_buffer, pad_buffer

    def _finalize_ddp_param_lists(self):
        ddp_pairs = list(zip(self.ddp_params, self.ddp_param_names, strict=True))
        ddp_pairs.sort(key=lambda x: x[0].numel(), reverse=True)
        dp_group = (
            self.parallel_dims.get_optional_mesh("dp_replicate").get_group() if self.dp_replicate_enabled else None
        )
        w = dp_group.size() if dp_group is not None else 1
        ddp_pairs = self._snake_interleave(ddp_pairs, w)
        if ddp_pairs:
            (
                self.ddp_params,  # pyrefly: ignore[bad-assignment]
                self.ddp_param_names,  # pyrefly: ignore[bad-assignment]
            ) = list(zip(*ddp_pairs, strict=True))
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
            param_kwargs_me = self.groups_info[self.parameters_to_groups[id(fsdp_params[ref_idx])]][-1]
        recv_shapes = [calculate_shard_shape(target_shape, r, ctx.world_size) for r in range(ctx.world_size)]
        recv_list_grads = [torch.empty(s, dtype=ctx.cast_dtype, device=ctx.device) for s in recv_shapes]
        dist.all_to_all(recv_list_grads, grads_send_list, group=ctx.fsdp_group)
        full_g = torch.cat(recv_list_grads, dim=0)
        if param_kwargs_me is None:
            raise RuntimeError("Failed to resolve Muon parameter kwargs for the FSDP bucket.")
        u = self.lmo(
            full_g,
            eps=param_kwargs_me["eps"],
            backend_steps=param_kwargs_me["backend_steps"],
            adjust_lr_fn=self.adjust_lr_fn,
            hybrid_ns=self.hybrid_ns,
        )
        if ctx.dp_replicate_group and self.extra_reduce_for_hsdp:
            dist.all_reduce(u, group=ctx.dp_replicate_group, op=dist.ReduceOp.AVG)
        return u, recv_shapes, send_shapes

    def _fsdp_scatter_updates(self, u, recv_shapes, send_shapes, ctx: CommContext):
        split_rows = [s[0] for s in recv_shapes]
        updates_send_list = list(torch.split(u, split_rows, dim=0))
        recv_list_updates = [torch.empty(s, dtype=ctx.cast_dtype, device=ctx.device) for s in send_shapes]

        dist.all_to_all(recv_list_updates, updates_send_list, group=ctx.fsdp_group)
        return recv_list_updates

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
            _, nesterov, momentum, _, param_kwargs = self.groups_info[self.parameters_to_groups[id(p)]]
            g = self.get_momentum_or_grad(p, momentum, nesterov)
            if g is None:
                local_updates[i] = None
                continue
            if isinstance(g, DTensor) and ctx.tp_group is not None:
                tp_dim = tp_axis(g.placements, tp_enabled=True)
                if tp_dim is not None:
                    g = gather_tp_shard(g.to_local(), ctx.tp_group, ctx.tp_world_size, g.placements)
                else:
                    g = g.to_local()
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
            return list(zip(slice_params, slice_updates, strict=True))
        prepared = []
        for p, u in zip(slice_params, slice_updates, strict=True):
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

    def _validate_and_set_expert_layout(self, experts_weights_layout):
        if experts_weights_layout not in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ]:
            raise ValueError(f"Unknown experts weights layout: {experts_weights_layout}")
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"


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
    optimizer_config: Any,
    base_lr: float,
) -> tuple[float, str | None]:
    """Calculate Muon's effective learning rate and adjustment mode.

    Returns:
        Tuple of (muon_lr, muon_adjust_lr_fn)
    """
    muon_adjust_lr_fn = optimizer_config.muon_adjust_lr_fn
    muon_lr = optimizer_config.muon_lr

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
    optimizer_config: Any,
    muon_adjust_lr_fn: str | None,
) -> dict[str, Any]:
    """Build kwargs for DistributedMuon constructor."""
    muon_kwargs = {
        "lr": muon_lr,
        "weight_decay": weight_decay,
        "momentum": optimizer_config.muon_momentum,
        "nesterov": optimizer_config.muon_enable_nesterov,
        "ns_steps": optimizer_config.muon_ns_steps,
        "eps": optimizer_config.eps,
        "adjust_lr_fn": muon_adjust_lr_fn,
        "hybrid_ns": optimizer_config.muon_hybrid_ns,
    }
    return muon_kwargs


def _build_adamw_kwargs(
    lr: float,
    weight_decay: float,
    optimizer_config: Any,
) -> dict[str, Any]:
    """Build kwargs for torch.optim.AdamW constructor."""
    optim_implementation = optimizer_config.implementation
    if optim_implementation not in ["fused", "foreach", "for-loop"]:
        raise ValueError(
            f"Invalid implementation '{optim_implementation}'. Must be one of: 'fused', 'foreach', 'for-loop'"
        )
    return {
        "lr": lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }


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

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        _owner: ClassVar[type | None] = None
        muon_lr: float | None = None
        muon_momentum: float = 0.95
        muon_enable_nesterov: bool = True
        muon_ns_steps: int = 5
        muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = "match_rms_adamw"
        muon_hybrid_ns: bool = False
        extra_param_group_split_rules: list[dict] | None = None

        def build(self, **kwargs):
            model_parts = kwargs["model_parts"]
            parallel_dims = kwargs.get("parallel_dims")
            if parallel_dims is None:
                from torchtitan_npu.patches.torchtitan._trainer_config_stash import get_active_parallel_dims

                parallel_dims = get_active_parallel_dims()
            ft_manager = kwargs.get("ft_manager")

            muon_lr_val, muon_adjust_lr_fn = _get_muon_lr_config(self, self.lr)
            (muon_params, muon_param_names, adamw_params, adamw_param_names) = _split_parameters_for_muon(model_parts)
            logger.info(f"[MuonAdamW] Muon optimizer parameters ({len(muon_param_names)}): {muon_param_names}")
            logger.info(f"[MuonAdamW] AdamW optimizer parameters ({len(adamw_param_names)}): {adamw_param_names}")

            muon_kwargs = _build_muon_kwargs(muon_lr_val, self.weight_decay, self, muon_adjust_lr_fn)
            adamw_kwargs = _build_adamw_kwargs(self.lr, self.weight_decay, self)
            param_groups = _build_muon_param_groups(
                muon_params, muon_param_names, muon_kwargs, self.extra_param_group_split_rules
            )
            muon = DistributedMuon(param_groups, muon_param_names, parallel_dims, **muon_kwargs)
            adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)
            owner = self._owner
            if owner is None:
                raise RuntimeError("MuonHybridOptimizersContainer.Config owner is not initialized.")
            return owner(model_parts, [muon, adamw], muon_adjust_lr_fn=muon_adjust_lr_fn)

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
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @property
    def muon_optimizer(self) -> Optimizer:
        return self.optimizers[0]

    @property
    def adamw_optimizer(self) -> Optimizer | None:
        return self.optimizers[1] if len(self.optimizers) > 1 else None

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
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
        for opt in self.optimizers:
            for model in self.model_parts:
                set_optimizer_state_dict(
                    model,
                    opt,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
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
            logger.warning("extra_param_group_split_rules entry missing 'str_match', skipping")
            continue
        rc["param_str_match"] = rc.pop("str_match")
        rule_configs.append(rc)

    # Classify params
    param_dict = OrderedDict(zip(param_names, params, strict=True))
    groups = []

    for rc in rule_configs:
        str_match = rc.pop("param_str_match")
        filter_fn = functools.partial(re.search, str_match)
        matched_names = [n for n in param_dict if filter_fn(n)]
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


class MuonLRSchedulersContainer(Stateful):
    """LR Scheduler container for Muon hybrid optimizers.

    Creates independent LambdaLR schedulers for Muon and AdamW,
    ensuring each maintains its own base_lr.

    Key difference from upstream LRSchedulersContainer:
    - Upstream: assumes all optimizers use the same base_lr
    - This class: allows Muon and AdamW to have different base_lr

    Inherits from Stateful so DCP correctly saves/loads lr_scheduler state.
    Without this, checkpoint resume would not restore last_epoch, causing
    wrong LR values after step 6+.

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
            raise ValueError(f"MuonHybridOptimizersContainer must have 2 optimizers, got {len(optimizers)}")

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
        logger.info(f"  Muon param_groups lr: {[pg['lr'] for pg in optimizers.muon_optimizer.param_groups]}")
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
                scheduler.base_lrs[i] * scheduler.lr_lambdas[i](last_epoch) for i in range(len(scheduler.base_lrs))
            ]


def _build_lr_lambda_from_config(
    lr_scheduler_config: LRSchedulersContainer.Config,
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
    lr_scheduler_config: LRSchedulersContainer.Config,
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


_ORIG_LR_SCHEDULER_BUILD = LRSchedulersContainer.Config.build


def _patched_lr_scheduler_build(self, *, optimizers, training_steps):
    if isinstance(optimizers, MuonHybridOptimizersContainer) and optimizers.muon_adjust_lr_fn == "original":
        total_steps = self.total_steps if self.total_steps is not None else training_steps
        logger.info(
            "[lr_scheduler patch] MuonHybridOptimizersContainer with "
            "muon_adjust_lr_fn=original, using MuonLRSchedulersContainer"
        )
        return build_muon_lr_schedulers(optimizers, self, total_steps)
    return _ORIG_LR_SCHEDULER_BUILD(self, optimizers=optimizers, training_steps=training_steps)


LRSchedulersContainer.Config.build = _patched_lr_scheduler_build  # pyrefly: ignore [bad-assignment]
