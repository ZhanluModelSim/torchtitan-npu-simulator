# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Upstream-PR: none - repository-specific document-chunk load balancing.

"""Document-chunk load balancing for the typed context-parallel input API."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor.experimental._context_parallel._load_balancer import _LoadBalancer
from torchtitan.distributed.context_parallel import ContextParallelLoadBalancer
from torchtitan.distributed.spmd_types import spmd_mesh_group

from torchtitan_npu.models.common.restore import doc_chunk_permutation, document_starts


class DocChunkLoadBalancer(_LoadBalancer):
    """Place each rank's chunk of every document in its sequence shard."""

    def __init__(self, cu_seqlens: torch.Tensor, row_len: int, cp_size: int) -> None:
        if row_len % cp_size:
            raise ValueError(f"the row ({row_len} tokens) does not split into {cp_size} slabs.")
        self.cu_seqlens = cu_seqlens
        self.slab_len = row_len // cp_size
        self.cp_size = cp_size

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        permutation = doc_chunk_permutation(self.cu_seqlens, self.cp_size, self.slab_len)
        return torch.argsort(permutation, dim=-1) if restore else permutation


class DocChunkCPLoadBalancer(ContextParallelLoadBalancer):
    """Build document chunks from the global positions in the metadata context."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancer.Config):
        pass

    def __init__(self, config: Config, *, seq_len: int, attention_metadata: Any) -> None:
        del config, attention_metadata
        from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_shard_context

        positions = cp_shard_context().positions
        cp_group = spmd_mesh_group("cp")
        if positions is None or cp_group is None:
            raise RuntimeError("Document-chunk CP requires global positions and an active CP mesh.")
        self.balancer = DocChunkLoadBalancer(document_starts(positions), seq_len, cp_group.size())

    def generate_permutation(self) -> torch.Tensor:
        return self.balancer._generate_indices()
