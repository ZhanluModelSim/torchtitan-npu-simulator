# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Adapt upstream healing callbacks to CPU transport and live NPU shards."""

import copy
from datetime import timedelta
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchft.manager import Manager

if TYPE_CHECKING:
    from torchtitan_npu.extensions.torchtitan.experiments.torchft.process_group import ProcessGroupHCCLEx


def _snapshot_to_cpu(state):
    leaves, spec = tree_flatten(state)
    copied = []
    for value in leaves:
        if isinstance(value, DTensor):
            value = value.to_local()
        copied.append(
            value.detach().to(device="cpu", copy=True) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        )
    return tree_unflatten(copied, spec)


@torch.no_grad()
def _restore_tensor_leaves(snapshot, live_state):
    """Copy local shards into existing tensors while preserving DTensor placement."""
    saved_leaves, saved_spec = tree_flatten(snapshot)
    live_leaves, live_spec = tree_flatten(live_state)
    if saved_spec != live_spec:
        raise ValueError("TorchFT recovery state structure does not match the current model")
    # Validate every leaf before changing any live state.
    for saved, live in zip(saved_leaves, live_leaves, strict=True):
        if isinstance(saved, torch.Tensor) != isinstance(live, torch.Tensor):
            raise ValueError("TorchFT recovery tensor type mismatch")
        if isinstance(live, torch.Tensor):
            target = live.to_local() if isinstance(live, DTensor) else live
            if saved.shape != target.shape or saved.dtype != target.dtype:
                raise ValueError("TorchFT recovery tensor shape or dtype mismatch")
    restored = []
    for saved, live in zip(saved_leaves, live_leaves, strict=True):
        if isinstance(live, torch.Tensor):
            if isinstance(live, DTensor) or live.device.type != "cpu":
                target = live.to_local() if isinstance(live, DTensor) else live
                target.copy_(saved)
                restored.append(live)
            else:
                # Optimizers can take ownership of CPU step counters on load.
                restored.append(saved.clone())
        else:
            restored.append(copy.deepcopy(saved))
    return tree_unflatten(restored, live_spec)


class TorchFTCommitTimeoutError(RuntimeError):
    """A commit result is unknown; torchrun must restart the whole replica."""


class ManagerEx(Manager):
    _pg: "ProcessGroupHCCLEx"
    _user_state_dicts: dict[str, Any]
    _quorum_future: Any
    _pending_state_dict: Any
    _recovery_event: Any
    _quorum_id: int
    _group_world_size: int
    _checkpoint_transport: Any
    _quorum_timeout: timedelta

    if TYPE_CHECKING:
        # TorchFT is optional in the base CI environment, so its inherited
        # members are not visible to the type checker there.
        def start_quorum(
            self,
            allow_heal: bool = True,
            shrink_only: bool = False,
            timeout: timedelta | None = None,
        ) -> None: ...

        def report_error(self, e: Exception) -> None: ...

        def state_dict(self) -> dict[str, int]: ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commit_authorized = False
        self._initial_quorum_barrier_pending = True
        self._healing_snapshot: dict[str, Any] | None = None
        self._snapshot_requested = Event()
        self._snapshot_ready = Event()
        self._snapshot_error: BaseException | None = None

    def _begin_step(self):
        self.commit_authorized = False
        if self._quorum_future is not None and not self._quorum_future.done():
            # A late healing request must finish before the snapshot events reset.
            self.wait_quorum()
        if self._quorum_id != -1 and cast("ProcessGroupHCCLEx", self._pg)._get_process_group() is None:
            # A failed configure may have recorded the quorum ID without creating a PG.
            self._quorum_id = -1
        self._healing_snapshot = None
        self._snapshot_error = None
        self._snapshot_requested.clear()
        self._snapshot_ready.clear()
        if self._initial_quorum_barrier_pending:
            if self._group_world_size > 1:
                dist.barrier()
            self._initial_quorum_barrier_pending = False
        self.start_quorum()

    def wait_quorum(self):
        assert self._quorum_future is not None
        while not self._quorum_future.done():
            if self._snapshot_requested.wait(0.01) and not self._snapshot_ready.is_set():
                try:
                    self._healing_snapshot = self._export_snapshot()
                except BaseException as error:
                    self._snapshot_error = error
                finally:
                    self._snapshot_ready.set()
        return super().wait_quorum()

    def should_commit(self, timeout=None):
        try:
            accepted = super().should_commit(timeout)
        except TimeoutError as error:
            self.report_error(error)
            self._checkpoint_transport.disallow_checkpoint()
            raise TorchFTCommitTimeoutError("TorchFT commit RPC timed out; restart the entire replica") from error
        self.commit_authorized = accepted
        return accepted

    def discard_interrupted_step(self, error):
        if self.commit_authorized:
            raise RuntimeError("Cannot discard an authorized optimizer update; restart the replica") from error
        # Device restart invalidates the old recovery stream event. Quorum was
        # applied synchronously before forward; no pending healing is lost here.
        self._recovery_event = None
        self.report_error(error)
        try:
            if self.should_commit():
                raise RuntimeError("An interrupted TorchFT attempt was unexpectedly authorized") from error
        finally:
            # Rebuild a destroyed communicator even if membership is unchanged.
            self._quorum_id = -1

    def _export_snapshot(self):
        return {
            "user": _snapshot_to_cpu({key: fn() for key, fn in self._user_state_dicts.items()}),
            "torchft": self.state_dict(),
        }

    def _apply_pending_state_dict(self):
        if self._recovery_event is not None:
            self._recovery_event.synchronize()
            self._recovery_event = None
        super()._apply_pending_state_dict()

    def register_state_dict_fn(self, key, load_state_dict, state_dict):
        def load_from_cpu(snapshot):
            load_state_dict(_restore_tensor_leaves(snapshot, state_dict()))

        super().register_state_dict_fn(key, load_from_cpu, state_dict)

    def _manager_state_dict(self):
        self._snapshot_requested.set()
        if not self._snapshot_ready.wait(self._quorum_timeout.total_seconds()):
            raise TimeoutError("Training thread did not prepare the TorchFT healing state")
        if self._snapshot_error is not None:
            raise RuntimeError("TorchFT healing state export failed") from self._snapshot_error
        if self._healing_snapshot is None:
            raise RuntimeError("TorchFT healing state was not prepared")
        return self._healing_snapshot
