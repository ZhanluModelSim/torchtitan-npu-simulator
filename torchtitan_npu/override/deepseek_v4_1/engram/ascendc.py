# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN Engram lookups with authoritative Host weights and sparse gradients.

The EP-local CPU table is excluded from FSDP. Its address-stable storage is
registered with ElasticBuffer, and only fetched rows enter NPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.profiler import record_function

from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

from .compile import compiled_lookup

_ENGRAM_ALIGNMENT = 128
_INT32_MAX = torch.iinfo(torch.int32).max


def _dedicated_engram_group(ep_group: dist.ProcessGroup) -> dist.ProcessGroup:
    """Give each table its own registered storage and communication context."""
    # The operator pool is keyed by communicator and retains the first storage
    # address. Rebuilding a model must not reuse a previous table's context.
    ranks = dist.get_process_group_ranks(ep_group)
    group = dist.new_group(ranks=ranks, backend="hccl", use_local_synchronization=True)
    assert isinstance(group, dist.ProcessGroup)
    return group


class _EngramFetchSparseOffload(torch.autograd.Function):
    """Accumulate owner-local gradients without exposing CPU weight.grad."""

    @staticmethod
    def forward(ctx, weight, row_ids, table, keepalive):  # pyrefly: ignore [bad-override]
        fetched, fetch = table._fetch_rows(row_ids)
        ctx.fetch = fetch
        ctx.table = table
        ctx.anchor_device = keepalive.device
        ctx.anchor_dtype = keepalive.dtype
        return fetched

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        ctx.table._accumulate_fetch_gradient(grad_output, ctx.fetch)
        return None, None, None, torch.zeros((), device=ctx.anchor_device, dtype=ctx.anchor_dtype)


class HostOffloadEngramTable(HostEngramTable):
    """Replace the Torch Host lookup with CANN Fetch/FetchGrad."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostEngramTable.Config):
        num_max_tokens_per_rank: int
        pin_memory: bool = True

    def __init__(self, config: Config):
        super().__init__(config)
        if config.num_max_tokens_per_rank <= 0:
            raise ValueError(f"num_max_tokens_per_rank must be positive, got {config.num_max_tokens_per_rank}.")
        if self.embedding_dim % _ENGRAM_ALIGNMENT != 0:
            # EngramFetch serves rows in 128-element units. Padding the shard to
            # meet that used to hide the requirement behind a per-forward copy;
            # declaring it keeps every flavor on the zero-copy path instead.
            raise ValueError(
                f"EngramFetch requires a row width that is a multiple of {_ENGRAM_ALIGNMENT}, "
                f"but the table declares embedding_dim={self.embedding_dim}."
            )
        if self.num_embeddings > _INT32_MAX:
            # EngramFetch takes int32 row IDs, so the table's row count is the
            # real constraint. Checking it here fails at build time instead of
            # waiting for a batch that happens to hash a row above the range.
            raise ValueError(
                f"EngramFetch only accepts int32 row IDs, but the table declares {self.num_embeddings} rows."
            )
        if not config.pin_memory:
            raise ValueError("AscendC Host Engram requires pin_memory=True for direct storage registration.")
        self.num_max_tokens_per_rank = config.num_max_tokens_per_rank
        self.pin_memory = config.pin_memory
        self._registered_storage: torch.Tensor | None = None
        self._registered_scale: torch.Tensor | None = None
        self._elastic_buffer = None
        self._elastic_buffer_spec: tuple[int, int, torch.dtype, int] | None = None

    def _fetch_storage_dtype(self, *, param_dtype: torch.dtype) -> torch.dtype:
        return self.storage_dtype(param_dtype=param_dtype)

    def _fetch_storage(self) -> torch.Tensor:
        return self.weight

    def _fetch_scale(self) -> torch.Tensor | None:
        return None

    def _initialize_fetch_storage(self) -> None:
        """Build any storage derived from the authoritative FP32 table."""

    def _submit_engram_storage(
        self,
        elastic_buffer,
        storage: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> None:
        """Register once; in-place optimizer and checkpoint updates stay visible."""
        storage = storage.detach()
        if scale is None:
            elastic_buffer.engram_write(storage)
        else:
            scale = scale.detach()
            elastic_buffer.engram_write(storage, scale)
        self._registered_storage = storage
        self._registered_scale = scale

    @staticmethod
    def _elastic_buffer_type():
        # Imported lazily so CPU/golden runs do not require the operator pack.
        from cann_ops_transformer import ElasticBuffer

        return ElasticBuffer

    def init_elastic_buffer(self, *, param_dtype: torch.dtype) -> None:
        """Create this table's communicator and ElasticBuffer.

        Called once from ``_shard_engram_tables``, which mirrors how
        ``BaseEPTokenDispatcher.wire_meshes`` calls ``init_buffer``: the EP mesh
        is known, no batch has been seen yet, and every rank reaches the
        collective in the same order. Doing it here rather than on the first
        lookup gives every rank the same communicator construction order.

        Only the shard geometry is needed, all of which follows from the config,
        so this runs before ``to_empty`` materializes any storage.
        """
        if self._elastic_buffer is not None:
            return
        if self.lookup_mesh is None or self.lookup_mesh.size() == 1:
            raise ValueError("AscendC Host Engram requires expert_parallel_degree > 1.")
        ep_size = self.lookup_mesh.size()
        if self.num_embeddings % ep_size != 0:
            raise ValueError(
                f"Engram table has {self.num_embeddings} physical rows, which is not divisible by EP degree {ep_size}."
            )
        local_rows = self.num_embeddings // ep_size
        if self.storage_dtype(param_dtype=param_dtype) != torch.float32:
            raise ValueError("AscendC Host Engram currently requires FP32 master weights.")
        dtype = self._fetch_storage_dtype(param_dtype=param_dtype)
        capacity = self.num_max_tokens_per_rank

        elastic_buffer_type = self._elastic_buffer_type()
        num_cpu_bytes = elastic_buffer_type.get_engram_storage_size_hint(local_rows, self.embedding_dim, dtype)
        group = _dedicated_engram_group(self.lookup_mesh.get_group())
        self._elastic_buffer = elastic_buffer_type(
            group,
            num_cpu_bytes=num_cpu_bytes,
            num_max_tokens_per_rank=capacity,
            with_grad=True,
        )
        self._elastic_buffer_spec = (local_rows, self.embedding_dim, dtype, capacity)

    def _require_elastic_buffer(
        self,
        storage: torch.Tensor,
        request_count: int,
        scale: torch.Tensor | None = None,
    ):
        """Return the buffer built for this shard, checking it still fits."""
        if self._elastic_buffer is None:
            raise RuntimeError(
                "The Engram ElasticBuffer has not been created. It is built by "
                "parallelize_deepseek_v4_1; a table used outside that path must call "
                "init_elastic_buffer() first."
            )
        capacity = self.num_max_tokens_per_rank
        if request_count > capacity:
            raise ValueError(
                "EngramFetch request count exceeds its fixed per-rank capacity: "
                f"got {request_count}, capacity {capacity}. Set "
                "num_max_tokens_per_rank to the largest "
                "batch * sequence * ngram-head count used by the run."
            )
        spec = (storage.shape[0], storage.shape[1], storage.dtype, capacity)
        if self._elastic_buffer_spec != spec:
            raise RuntimeError(
                "The Engram shard does not match the ElasticBuffer built for it: "
                f"{self._elastic_buffer_spec} -> {spec}."
            )
        if self._registered_storage is None:
            raise RuntimeError("Engram storage is not registered; initialize model weights before lookup.")
        if storage.data_ptr() != self._registered_storage.data_ptr():
            raise RuntimeError("Engram registered storage was replaced; rebuild the model with a new communicator.")
        if (scale is None) != (self._registered_scale is None):
            raise RuntimeError("Engram registered scale metadata changed; rebuild the model with a new communicator.")
        if (
            scale is not None
            and self._registered_scale is not None
            and scale.data_ptr() != self._registered_scale.data_ptr()
        ):
            raise RuntimeError("Engram registered scale was replaced; rebuild the model with a new communicator.")
        return self._elastic_buffer

    def parallelize(self, parallel_dims) -> None:
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None or ep_mesh.size() <= 1:
            raise ValueError("AscendC Host Engram requires expert_parallel_degree > 1.")
        super().parallelize(parallel_dims)

    def _init_self_parameters(self) -> None:
        super()._init_self_parameters()
        self._initialize_fetch_storage()
        if self._elastic_buffer is not None:
            self._submit_engram_storage(
                self._elastic_buffer,
                self._fetch_storage(),
                self._fetch_scale(),
            )

    def _lookup_dtype(self) -> torch.dtype:
        return torch.float32

    def _fetch_rows(self, row_ids: torch.Tensor):
        buffer = self._require_elastic_buffer(self._fetch_storage(), row_ids.numel(), self._fetch_scale())
        with record_function("engram::fetch"):
            indices = row_ids.reshape(-1).to(dtype=torch.int32).contiguous()
            return buffer.engram_fetch(indices)()

    def _accumulate_fetch_gradient(self, grad_output: torch.Tensor, fetch) -> None:
        buffer = self._elastic_buffer
        assert buffer is not None
        grad_fetched = grad_output.reshape(-1, self.embedding_dim).float().contiguous()
        with record_function("engram::fetch_grad"):
            grad_unique, unique_local_entry = buffer.engram_fetch_grad(grad_fetched, fetch)
        with record_function("engram::sparse_accum"):
            self.accumulate_sparse_gradient(unique_local_entry, grad_unique)

    def _distributed_lookup(self, row_ids_N: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return compiled_lookup(self, row_ids_N)
        flat_ids = row_ids_N.reshape(-1)
        return _EngramFetchSparseOffload.apply(self.weight, flat_ids, self, self._grad_keepalive).view(
            *row_ids_N.shape, self.embedding_dim
        )


__all__ = ["HostOffloadEngramTable"]
