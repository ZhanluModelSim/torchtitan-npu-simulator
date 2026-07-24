# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rank identities used by multi-process meta simulation.

The real process group contains one worker per PP rank, while DeviceMesh
describes the full simulated world. A capture worker therefore has two rank
identities that must not be interchanged:

* ``capture_process_rank`` is the real Gloo rank (and PP control-plane rank).
* ``logical_global_rank`` is the representative rank in the full mesh.

The representative keeps every non-PP coordinate at zero. TorchTitan builds
the PP axis first, so its flattened rank is ``pp_rank * ranks_per_pp_stage``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationRankContext:
    capture_process_rank: int
    capture_world_size: int
    logical_global_rank: int
    logical_world_size: int
    pp_degree: int

    @classmethod
    def resolve(
        cls,
        *,
        logical_world_size: int,
        pp_degree: int,
    ) -> "SimulationRankContext":
        if logical_world_size < 1 or pp_degree < 1:
            raise ValueError(
                "logical_world_size and pp_degree must be positive, got "
                f"{logical_world_size} and {pp_degree}"
            )
        if logical_world_size % pp_degree:
            raise ValueError(
                f"logical world size {logical_world_size} is not divisible by "
                f"PP degree {pp_degree}"
            )

        env_rank = os.environ.get("RANK")
        env_world_size = os.environ.get("WORLD_SIZE")
        if (env_rank is None) != (env_world_size is None):
            raise RuntimeError(
                "RANK and WORLD_SIZE must either both be set or both be absent"
            )
        capture_rank = int(env_rank) if env_rank is not None else 0
        capture_world_size = (
            int(env_world_size) if env_world_size is not None else (pp_degree if pp_degree > 1 else 1)
        )
        distributed_initialized = False

        try:
            import torch.distributed as dist

            if dist.is_initialized():
                distributed_initialized = True
                distributed_rank = dist.get_rank()
                distributed_world_size = dist.get_world_size()
                if env_rank is not None and distributed_rank != capture_rank:
                    raise RuntimeError(
                        f"RANK={capture_rank} disagrees with distributed rank "
                        f"{distributed_rank}"
                    )
                if env_world_size is not None and distributed_world_size != capture_world_size:
                    raise RuntimeError(
                        f"WORLD_SIZE={capture_world_size} disagrees with distributed "
                        f"world size {distributed_world_size}"
                    )
                capture_rank = distributed_rank
                capture_world_size = distributed_world_size
        except ImportError:
            pass

        if (
            pp_degree > 1
            and env_rank is None
            and not distributed_initialized
        ):
            raise RuntimeError(
                "PP capture requires one initialized worker per pipeline rank; "
                "RANK/WORLD_SIZE are absent and torch.distributed is not initialized"
            )
        if pp_degree > 1 and capture_world_size != pp_degree:
            raise RuntimeError(
                "multi_proc_meta requires one real worker per PP rank: "
                f"capture world size={capture_world_size}, PP degree={pp_degree}"
            )
        if not 0 <= capture_rank < capture_world_size:
            raise ValueError(
                f"capture rank {capture_rank} is outside world size {capture_world_size}"
            )

        ranks_per_pp_stage = logical_world_size // pp_degree
        return cls(
            capture_process_rank=capture_rank,
            capture_world_size=capture_world_size,
            logical_global_rank=capture_rank * ranks_per_pp_stage,
            logical_world_size=logical_world_size,
            pp_degree=pp_degree,
        )


_active_rank_context: SimulationRankContext | None = None


def configure_simulation_rank_context(
    *,
    logical_world_size: int,
    pp_degree: int,
) -> SimulationRankContext:
    global _active_rank_context
    context = SimulationRankContext.resolve(
        logical_world_size=logical_world_size,
        pp_degree=pp_degree,
    )
    _active_rank_context = context
    return context


def get_simulation_rank_context() -> SimulationRankContext:
    if _active_rank_context is None:
        raise RuntimeError("simulation rank context has not been configured")
    return _active_rank_context


def clear_simulation_rank_context() -> None:
    global _active_rank_context
    _active_rank_context = None
