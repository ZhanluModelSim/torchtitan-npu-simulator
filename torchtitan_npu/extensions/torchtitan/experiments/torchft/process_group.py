# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HCCL process-group construction and lifecycle for the TorchFT experiment."""

from datetime import timedelta
from typing import Any, cast

import torch
from torch.distributed import ProcessGroup, Store
from torchft.process_group import ProcessGroupWrapper

from torchtitan_npu.extensions.torchtitan.experiments.torchft.recovery.hccl import HcclRecovery


class ProcessGroupHCCLEx(ProcessGroupWrapper):
    # Inherited from the pinned ProcessGroupWrapper; the dependency is optional
    # in the static-only CodeCheck environment.
    _timeout: timedelta
    _quorum_id: int | None
    _group_rank: int | None
    _global_ranks: list[int] | None

    def __init__(
        self,
        timeout=timedelta(seconds=60),
        *,
        recovery_timeout: timedelta,
        warmup_timeout: timedelta | None = None,
    ):
        super().__init__(timeout)
        self.recovery = HcclRecovery(
            timeout,
            recovery_timeout=recovery_timeout,
            warmup_timeout=warmup_timeout,
            get_process_group=self._get_process_group,
            clear_process_group=self._clear_process_group,
        )
        self._used_store_addresses: set[str] = set()

    def configure(
        self,
        store_addr: str,
        replica_id: str,
        rank: int,
        world_size: int,
        quorum_id: int | None = None,
        group_rank: int | None = None,
        group_world_size: int | None = None,
        global_ranks: list[int] | None = None,
    ) -> None:
        # Lighthouse advances the quorum ID after a rejected commit, which gives
        # every rank a new prefix. Never rebuild HCCL with stale rendezvous keys.
        if store_addr in self._used_store_addresses:
            raise RuntimeError(f"TorchFT HCCL requires a fresh store prefix for reconfiguration: {store_addr}")
        self._used_store_addresses.add(store_addr)
        super().configure(
            store_addr, replica_id, rank, world_size, quorum_id, group_rank, group_world_size, global_ranks
        )

    def _get_process_group(self):
        return self._pg

    def _clear_process_group(self, pg):
        assert self._pg is pg
        self._pg = None

    def errored(self):
        return self.recovery.errored()

    def _run_context(self):
        return self.recovery.track_collective()

    def _wrap_work(self, work, opts):
        return self.recovery.wrap_work(self, work, opts)

    def wait_for_completion(self, work) -> None:
        self.recovery.wait_for_completion(work)

    def getBackendName(self) -> str:  # noqa: N802
        return "torchft-hccl"

    def _create_pg(self, store: Store, rank: int, world_size: int) -> ProcessGroup:
        from torch_npu._C._distributed_c10d import ProcessGroupHCCL  # pyrefly: ignore [missing-import]

        self.recovery.prepare_group(torch.accelerator.current_device_index())
        options = ProcessGroupHCCL.Options()
        options._timeout = self.recovery.watchdog_timeout
        options.group_id = f"torchft_quorum_{self._quorum_id}_rank_{self._group_rank}"
        if self._global_ranks:
            options.global_ranks_in_group = self._global_ranks
        backend = ProcessGroupHCCL(store, rank, world_size, options)
        backend._set_sequence_number_for_group()
        pg = ProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.CUSTOM)
        pg._register_backend(torch.device("npu"), ProcessGroup.BackendType.CUSTOM, backend)
        self.recovery.group_created()
        return pg

    def abort(self, errored: bool = True) -> None:
        with self.recovery.release_group(errored=errored) as pg:
            if pg is not None:
                backend = cast("Any", pg._get_backend(torch.device("npu")))
                backend.abort()
                backend.shutdown()
                backend.clear_workmeta_list()

    def shutdown(self) -> None:
        with self.recovery.release_group(errored=False) as pg:
            if pg is not None:
                backend = cast("Any", pg._get_backend(torch.device("npu")))
                backend.shutdown()
