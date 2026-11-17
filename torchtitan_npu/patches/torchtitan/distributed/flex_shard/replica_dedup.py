# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# TODO: Delete this patch once https://github.com/pytorch/torchtitan/pull/4411
# ([DistMuon] Deduplicate HSDP replica compute) is merged upstream.

"""HSDP replica-state deduplication for TorchTitan ``DistMuon``.

The implementation deliberately leaves FlexShard's storage-to-compute runtime
unchanged.  A complete parameter storage tensor is assigned to one
``dp_replicate`` coordinate, and that replica domain rebuilds ordinary native
bucket plans for its owned tensor subset.  Once all owners have completed the
native update, the updated local parameter shards are broadcast over the real
``dp_replicate`` process group.

The owner-only update is followed by a canonical asynchronous broadcast as part
of the deduplicated step. It requires the TorchTitan 0.3 FlexShard interfaces
used below.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
import torchtitan.components.optimizer.optimizer as optimizer_module
import torchtitan.distributed.flex_shard as flex_shard_module
import torchtitan.distributed.flex_shard.dist_muon as dist_muon_module
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch.distributed.device_mesh import DeviceMesh


_DP_REPLICATE_AXIS = "dp_replicate"
_ORIGINAL_BUILD_DIST_MUON = dist_muon_module.build_dist_muon
_ORIGINAL_STEP = dist_muon_module.DistMuon.step


def _owner_coordinate(fqn: str, replica_degree: int) -> int:
    """Return a process-independent owner coordinate for one complete tensor."""
    if replica_degree < 1:
        raise ValueError("dp_replicate degree must be positive")
    return int.from_bytes(hashlib.sha256(fqn.encode("utf-8")).digest()[:8], "big") % replica_degree


def _mesh_axis(mesh: DeviceMesh, axis_name: str) -> int:
    axis_names = mesh.mesh_dim_names
    if axis_names is None or axis_name not in axis_names:
        raise ValueError(f"HSDP Muon dedup requires mesh axis {axis_name!r}")
    return axis_names.index(axis_name)


def _mesh_coordinate(mesh: DeviceMesh) -> tuple[int, ...]:
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("current rank is not a member of the parameter device mesh")
    return tuple(coordinate)


def _owner_rank(mesh: DeviceMesh, *, axis: int, owner_coordinate: int) -> int:
    coordinate = list(_mesh_coordinate(mesh))
    coordinate[axis] = owner_coordinate
    return int(mesh.mesh[tuple(coordinate)].item())


def _replica_group_ranks(mesh: DeviceMesh, *, axis: int) -> tuple[int, ...]:
    """Return this tensor's dp_replicate group as a stable global-rank key."""
    coordinate = list(_mesh_coordinate(mesh))
    ranks = []
    for replica_coordinate in range(mesh.size(axis)):
        coordinate[axis] = replica_coordinate
        ranks.append(int(mesh.mesh[tuple(coordinate)].item()))
    return tuple(ranks)


def _validate_replica_dedup_layout(compute_layout: Any) -> tuple[int, int, int]:
    """Resolve and validate one tensor's HSDP ownership coordinate."""
    fqn = compute_layout.fqn
    param = compute_layout.param
    if not isinstance(param, DTensor):
        raise TypeError(f"HSDP Muon dedup requires DTensor parameter {fqn!r}")
    mesh = param.device_mesh
    axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
    if type(param.placements[axis]) is not Replicate:
        raise ValueError(
            f"HSDP Muon dedup requires replicated storage on {_DP_REPLICATE_AXIS!r} "
            f"for {fqn!r}; got {param.placements[axis]!r}"
        )
    degree = mesh.size(axis)
    return axis, degree, _owner_coordinate(fqn, degree)


def _canonical_replica_dedup_layouts(muon: Any) -> tuple[Any, ...]:
    """Return the immutable full layout order used by replica collectives.

    Native DistMuon load hooks rebuild plans from the current
    ``_parameter_compute_layouts``.  Under replica dedup that is intentionally
    the owner-only subset, so it must never be used to replace the full
    broadcast/checkpoint order after construction.
    """
    canonical_layouts = getattr(muon, "_replica_dedup_all_layouts", None)
    if canonical_layouts is None:
        canonical_layouts = tuple(muon._parameter_compute_layouts)
        canonical_fqns = tuple(layout.fqn for layout in canonical_layouts)
        if len(set(canonical_fqns)) != len(canonical_fqns):
            raise ValueError("HSDP Muon dedup canonical layouts contain duplicate FQNs")
        muon._replica_dedup_all_layouts = canonical_layouts
    return canonical_layouts


def _canonical_layout_fingerprint(layouts: Sequence[Any], owner_by_fqn: Mapping[str, int]) -> int:
    """Hash the ordered transport contract without serializing tensor metadata."""
    digest = hashlib.sha256()
    for layout in layouts:
        fqn = layout.fqn.encode("utf-8")
        digest.update(len(fqn).to_bytes(4, "big"))
        digest.update(fqn)
        digest.update(owner_by_fqn[layout.fqn].to_bytes(4, "big"))
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


def _validate_canonical_layouts_across_replicas(layouts: Sequence[Any], owner_by_fqn: Mapping[str, int]) -> None:
    """Fail before training if replica peers would issue different broadcasts."""
    if not layouts:
        return
    param = layouts[0].param
    mesh = param.device_mesh
    group = mesh[_DP_REPLICATE_AXIS].get_group()
    local = torch.tensor(
        [len(layouts), _canonical_layout_fingerprint(layouts, owner_by_fqn)],
        device=param.to_local().device,
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    local_contract = tuple(local.cpu().tolist())
    gathered_contracts = [tuple(item.cpu().tolist()) for item in gathered]
    if any(contract != local_contract for contract in gathered_contracts):
        raise RuntimeError(
            "HSDP Muon dedup canonical FQN/owner sequence differs across "
            f"dp_replicate peers: rank={dist.get_rank()}, "
            f"local={local_contract}, gathered={gathered_contracts}"
        )


def _configure_replica_dedup(muon: Any) -> None:
    """Install owner-only native plans and retain canonical broadcast order."""
    all_layouts = _canonical_replica_dedup_layouts(muon)
    owner_by_fqn: dict[str, int] = {}
    local_owned_layouts = []
    for layout in all_layouts:
        axis, degree, owner = _validate_replica_dedup_layout(layout)
        del axis, degree
        owner_by_fqn[layout.fqn] = owner
        mesh = layout.param.device_mesh
        replica_axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
        if _mesh_coordinate(mesh)[replica_axis] == owner:
            local_owned_layouts.append(layout)

    _validate_canonical_layouts_across_replicas(all_layouts, owner_by_fqn)

    total_expert_tensors = sum(".moe.routed_experts." in layout.fqn for layout in all_layouts)
    local_expert_tensors = sum(".moe.routed_experts." in layout.fqn for layout in local_owned_layouts)
    logger.info(
        "HSDP DistMuon replica dedup: rank=%d owns %d/%d tensors (%d/%d routed-expert tensors)",
        dist.get_rank(),
        len(local_owned_layouts),
        len(all_layouts),
        local_expert_tensors,
        total_expert_tensors,
    )

    muon._initialize_plan(tuple(local_owned_layouts))
    muon._replica_dedup_owner_by_fqn = owner_by_fqn
    muon._validate_plan_across_ranks()
    muon._redistribution_runtime.reserve_buffers(
        muon._bucket_plans,
        local_tensor_spec=muon._local_tensor_spec,
    )

    muon._replica_broadcast_pipeline = _ReplicaBroadcastPipeline(muon)


@dataclass(frozen=True, slots=True)
class _BroadcastEntry:
    """One canonical parameter broadcast over its real dp_replicate group."""

    fqn: str
    param: DTensor
    local: torch.Tensor
    group: Any
    group_key: tuple[int, ...]
    source_rank: int
    is_local_owner: bool


@dataclass(slots=True)
class _InflightBroadcast:
    entry: _BroadcastEntry
    work: Any


@dataclass(slots=True)
class _GroupProgress:
    entries: tuple[_BroadcastEntry, ...]
    cursor: int = 0


class _ReplicaBroadcastPipeline:
    """Issue canonical async broadcasts when locally owned storage is ready.

    Each dp_replicate group has an independent cursor. Non-owners may prepost
    the next receive, while an owner cursor stops until its native
    ``_apply_update`` has recorded an event. This preserves a shared collective
    sequence without serializing the owner-only FlexShard runtimes.
    """

    def __init__(self, muon: Any) -> None:
        all_layouts = _canonical_replica_dedup_layouts(muon)
        entries_by_group: dict[tuple[int, ...], list[_BroadcastEntry]] = {}
        entries_by_fqn: dict[str, _BroadcastEntry] = {}
        for layout in all_layouts:
            param = layout.param
            mesh = param.device_mesh
            axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
            group_key = _replica_group_ranks(mesh, axis=axis)
            owner_coordinate = muon._replica_dedup_owner_by_fqn[layout.fqn]
            entry = _BroadcastEntry(
                fqn=layout.fqn,
                param=param,
                local=param.to_local(),
                group=mesh[_DP_REPLICATE_AXIS].get_group(),
                group_key=group_key,
                source_rank=_owner_rank(
                    mesh,
                    axis=axis,
                    owner_coordinate=owner_coordinate,
                ),
                is_local_owner=_mesh_coordinate(mesh)[axis] == owner_coordinate,
            )
            entries_by_group.setdefault(group_key, []).append(entry)
            entries_by_fqn[entry.fqn] = entry

        if not entries_by_fqn:
            raise RuntimeError("HSDP async replica broadcast requires Muon layouts")
        self._entries_by_fqn = entries_by_fqn
        self._groups = {group_key: _GroupProgress(tuple(entries)) for group_key, entries in entries_by_group.items()}
        device: torch.device = next(iter(entries_by_fqn.values())).local.device
        self._device_handle = torch.get_device_module(device)
        self._broadcast_stream = self._device_handle.Stream(device=device, priority=0)
        self._ready_events = {fqn: self._device_handle.Event() for fqn in entries_by_fqn}
        self._ready_fqns: set[str] = set()
        self._inflight: list[_InflightBroadcast] = []
        self._caller_stream: Any | None = None
        self._state: Literal["idle", "active", "failed"] = "idle"
        self._validate_contract()

    def _broadcast_device(self) -> torch.device:
        return next(iter(self._entries_by_fqn.values())).local.device

    def _validate_contract(self) -> None:
        """Fail before a step if peers disagree on any transport collective."""
        for group_key, progress in self._groups.items():
            entries = progress.entries
            digest = hashlib.sha256()
            for entry in entries:
                fqn = entry.fqn.encode("utf-8")
                digest.update(len(fqn).to_bytes(4, "big"))
                digest.update(fqn)
                digest.update(entry.source_rank.to_bytes(8, "big", signed=True))
                digest.update(str(entry.local.dtype).encode("utf-8"))
                digest.update(len(entry.local.shape).to_bytes(4, "big"))
                for dimension in entry.local.shape:
                    digest.update(int(dimension).to_bytes(8, "big", signed=True))
            fingerprint = int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)
            local = torch.tensor(
                [len(entries), fingerprint],
                device=self._broadcast_device(),
                dtype=torch.int64,
            )
            gathered = [torch.empty_like(local) for _ in group_key]
            dist.all_gather(gathered, local, group=entries[0].group)
            contracts = [tuple(item.cpu().tolist()) for item in gathered]
            if any(contract != contracts[0] for contract in contracts[1:]):
                raise RuntimeError(
                    f"HSDP async replica broadcast contract differs across group={group_key}: contracts={contracts}"
                )

    def ensure_idle(self) -> None:
        if self._state != "idle":
            raise RuntimeError(
                f"HSDP async replica broadcast is not idle at a checkpoint boundary: state={self._state}"
            )

    def begin_step(self) -> None:
        self.ensure_idle()
        self._state = "active"
        self._ready_fqns.clear()
        self._inflight.clear()
        self._caller_stream = self._device_handle.current_stream(self._broadcast_device())
        for progress in self._groups.values():
            progress.cursor = 0
        for group_key in sorted(self._groups):
            self._progress(group_key)

    def mark_updated(self, compute_layout: Any) -> None:
        if self._state != "active":
            raise RuntimeError("HSDP async replica broadcast received an update while inactive")
        entry = self._entries_by_fqn.get(compute_layout.fqn)
        if entry is None:
            raise RuntimeError(f"HSDP async replica broadcast has no entry for {compute_layout.fqn!r}")
        if not entry.is_local_owner:
            raise RuntimeError(f"HSDP async replica broadcast observed a non-owner update for {entry.fqn!r}")
        if entry.fqn in self._ready_fqns:
            raise RuntimeError(f"HSDP async replica broadcast observed duplicate update for {entry.fqn!r}")
        self._ready_events[entry.fqn].record(self._device_handle.current_stream(self._broadcast_device()))
        self._ready_fqns.add(entry.fqn)
        self._progress(entry.group_key)

    def _progress(self, group_key: tuple[int, ...]) -> None:
        progress = self._groups[group_key]
        while progress.cursor < len(progress.entries):
            entry = progress.entries[progress.cursor]
            if entry.is_local_owner and entry.fqn not in self._ready_fqns:
                return
            with self._device_handle.stream(self._broadcast_stream):
                if entry.is_local_owner:
                    self._broadcast_stream.wait_event(self._ready_events[entry.fqn])
                work = dist.broadcast(
                    entry.local,
                    src=entry.source_rank,
                    group=entry.group,
                    async_op=True,
                )
            self._inflight.append(_InflightBroadcast(entry=entry, work=work))
            progress.cursor += 1

    def finish_step(self) -> None:
        if self._state != "active":
            raise RuntimeError("HSDP async replica broadcast finish while inactive")
        missing = [
            entry.fqn
            for entry in self._entries_by_fqn.values()
            if entry.is_local_owner and entry.fqn not in self._ready_fqns
        ]
        if missing:
            raise RuntimeError(
                f"HSDP async replica broadcast missed owner updates: {missing[:3]} (total={len(missing)})"
            )
        for group_key, progress in self._groups.items():
            self._progress(group_key)
            if progress.cursor != len(progress.entries):
                entry = progress.entries[progress.cursor]
                raise RuntimeError(
                    f"HSDP async replica broadcast could not issue canonical entry {entry.fqn!r} in group={group_key}"
                )
        for inflight in self._inflight:
            inflight.work.wait()
            if not inflight.entry.is_local_owner:
                torch.autograd.graph.increment_version(inflight.entry.param)
        assert self._caller_stream is not None
        self._caller_stream.wait_stream(self._broadcast_stream)
        self._state = "idle"
        self._caller_stream = None

    def mark_failed(self) -> None:
        self._state = "failed"


def _prune_nonowner_state(muon: Any) -> None:
    """Return a loaded standard-layout state dict to owner-only training state."""
    for layout in muon._replica_dedup_all_layouts:
        mesh = layout.param.device_mesh
        axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
        if _mesh_coordinate(mesh)[axis] != muon._replica_dedup_owner_by_fqn[layout.fqn]:
            muon.state.pop(layout.param, None)


def _validate_owner_momentum_state(muon: Any, *, phase: str) -> None:
    """Check that loaded persistent momentum exists only on its owner replica."""
    for layout in muon._replica_dedup_all_layouts:
        mesh = layout.param.device_mesh
        axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
        is_owner = _mesh_coordinate(mesh)[axis] == muon._replica_dedup_owner_by_fqn[layout.fqn]
        has_momentum = "momentum_buffer" in muon.state.get(layout.param, {})
        if is_owner != has_momentum:
            raise RuntimeError(
                "HSDP Muon dedup momentum ownership mismatch after "
                f"{phase}: rank={dist.get_rank()}, fqn={layout.fqn!r}, "
                f"owner={muon._replica_dedup_owner_by_fqn[layout.fqn]}, "
                f"has_momentum={has_momentum}"
            )


def _after_replica_dedup_load_state_dict(optimizer: torch.optim.Optimizer) -> None:
    """Restore owner plans after the native post-load hook rebuilt full plans."""
    _configure_replica_dedup(optimizer)
    _prune_nonowner_state(optimizer)
    _validate_owner_momentum_state(optimizer, phase="DCP load")


def _ensure_storage_layout_momentum(muon: Any, layout: Any) -> DTensor:
    """Create a standard storage-layout momentum state on one non-owner rank."""
    state = muon.state[layout.param]
    momentum = state.get("momentum_buffer")
    if momentum is None:
        momentum = torch.zeros_like(layout.param, memory_format=torch.preserve_format)
        state["momentum_buffer"] = momentum
    if not isinstance(momentum, DTensor):
        raise RuntimeError(f"expected DTensor momentum for {layout.fqn!r}")
    return momentum


@contextmanager
def standard_state_dict_layout(muon: Any):
    """Temporarily materialize owner momentum on every replica for native DCP."""
    pipeline = getattr(muon, "_replica_broadcast_pipeline", None)
    if pipeline is not None:
        pipeline.ensure_idle()
    temporary_params = []
    try:
        for layout in muon._replica_dedup_all_layouts:
            mesh = layout.param.device_mesh
            axis = _mesh_axis(mesh, _DP_REPLICATE_AXIS)
            owner = muon._replica_dedup_owner_by_fqn[layout.fqn]
            if _mesh_coordinate(mesh)[axis] != owner:
                _ensure_storage_layout_momentum(muon, layout)
                temporary_params.append(layout.param)
            momentum = _ensure_storage_layout_momentum(muon, layout)
            group = mesh[_DP_REPLICATE_AXIS].get_group()
            src = _owner_rank(mesh, axis=axis, owner_coordinate=owner)
            dist.broadcast(momentum.to_local(), src=src, group=group)
        yield
    finally:
        for param in temporary_params:
            muon.state.pop(param, None)


@contextmanager
def standard_load_layout(muon: Any):
    """Expose every state template before native DCP unflattens its flat dict."""
    pipeline = getattr(muon, "_replica_broadcast_pipeline", None)
    if pipeline is not None:
        pipeline.ensure_idle()
    try:
        for layout in muon._replica_dedup_all_layouts:
            _ensure_storage_layout_momentum(muon, layout)
        yield
    finally:
        _prune_nonowner_state(muon)


@torch.no_grad()
def _replica_dedup_step(muon: Any, closure: Any = None) -> float | None:
    """Run native owner plans and overlap canonical parameter broadcasts."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    pipeline = muon._replica_broadcast_pipeline
    original_apply_update = muon._apply_update

    def apply_and_mark(compute_layout: Any, direction: torch.Tensor) -> None:
        original_apply_update(compute_layout, direction)
        pipeline.mark_updated(compute_layout)

    pipeline.begin_step()
    muon._apply_update = apply_and_mark
    try:
        _ORIGINAL_STEP(muon, None)
        pipeline.finish_step()
    except Exception:
        pipeline.mark_failed()
        raise
    finally:
        muon._apply_update = original_apply_update
    return loss


def build_dist_muon(
    params: Iterable[dict[str, Any]],
    *,
    compute_sharding_by_fqn: Mapping[str, Any],
    bucket_configs: Sequence[Any],
    enable_hsdp_replica_dedup: bool = False,
    **kwargs: Any,
) -> Any:
    """Build native DistMuon, optionally replacing its plans with owner plans."""
    muon = _ORIGINAL_BUILD_DIST_MUON(
        params,
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
        **kwargs,
    )
    if not enable_hsdp_replica_dedup:
        return muon
    _configure_replica_dedup(muon)
    muon.register_load_state_dict_post_hook(_after_replica_dedup_load_state_dict)
    muon.step = _replica_dedup_step.__get__(muon, type(muon))
    return muon


def apply() -> None:
    """Route every upstream factory binding through the opt-in wrapper."""
    dist_muon_module.build_dist_muon = build_dist_muon
    flex_shard_module.build_dist_muon = build_dist_muon
    optimizer_module.build_dist_muon = build_dist_muon


apply()
