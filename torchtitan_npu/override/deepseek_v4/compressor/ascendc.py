# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CANN compressor using the operator's native autograd implementation.

The fused kernel compresses one local stream, addressed by that stream's own ``cu_seqlens``: the
ratio-4 overlap's carry is not its business (``CarryExchange`` prepends the
predecessor block, passes the augmented boundaries and trims the extra output block), so this
module has no process group and no CP branch.  The kernel's zero tiling table is an operator
input built here, from the stream's own shape symbols, never from a host maximum that a traced
graph would bake in; the model-level compressor interface does not know it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.models.deepseek_v4.compressor import Compressor as UpstreamCompressor

from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation
from torchtitan_npu.ops.ascendc.compressor import load_compressor


def _state_inputs(
    cu_seqlens: torch.Tensor,
    x: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
):
    cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
    # Two slices, not ``torch.diff``: the document count is an unbacked symbol here and
    # ``diff``'s length check would guard on it, which the FX tracer cannot evaluate.
    seqused = cu_seqlens[1:] - cu_seqlens[:-1]
    block_size = 8 if ratio == 4 else 16
    coff = 2 if ratio == 4 else 1
    # CANN tiling requires state_block_table despite its optional schema.
    # In cache_mode=1, block ID 0 skips cache writes for full-document training.  Neither rows
    # nor width may be a traced host constant: a table captured with one batch's shape would be
    # reused for the next batch's documents.  Both are shape symbols here -- the segment count
    # from the boundaries, the width bounded by the stream itself (a document is no longer than
    # the stream it sits in).
    width = torch.sym_max((x.shape[1] + block_size - 1) // block_size, 1)
    state_block_table = torch.zeros((seqused.numel(), width), dtype=torch.int32, device=x.device)
    state_cache = torch.zeros((1, block_size, 2 * coff * head_dim), dtype=torch.float32, device=x.device)
    return state_block_table, state_cache, cu_seqlens, seqused


class AscCompressor(CompressorImplementation):
    @dataclass(kw_only=True, slots=True)
    class Config(CompressorImplementation.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._compressor_fn = load_compressor()

    def forward(
        self,
        compressor: Compressor,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ratio = compressor.compress_ratio
        if cu_seqlens is None:
            raise ValueError("the CANN compressor requires the local stream's cu_seqlens")
        state_block_table, state_cache, cu_seqlens, seqused = _state_inputs(
            cu_seqlens,
            x,
            ratio=ratio,
            head_dim=compressor.head_dim,
        )
        pooled = self._compressor_fn(
            x.reshape(-1, x.shape[-1]).contiguous(),
            compressor.wkv.weight,
            compressor.wgate.weight,
            state_cache,
            # CANN also allocates saved softmax/KV outputs with APE's dtype.
            compressor.ape.float(),
            cmp_ratio=ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            coff=1 + int(compressor.overlap),
            cache_mode=1,
        )
        # The kernel may allocate capacity beyond this stream's blocks; the kept blocks are
        # this stream's own count, expressed with its shape symbol rather than a host value.
        pooled = compressor.norm(pooled[: x.shape[1] // ratio].to(x.dtype))
        return compressor.rope(pooled.unsqueeze(1), positions=positions.reshape(-1)[::ratio]).squeeze(1).unsqueeze(0)


class AscNativeCompressor(UpstreamCompressor):
    """Fuse complete packed groups while retaining the upstream compressor layout.

    CANN emits partial groups, whereas upstream packs only complete groups and
    keeps unused pool slots. Partial or empty documents therefore use upstream
    arithmetic. Checking document alignment synchronizes in eager execution;
    this dispatch is not a compilation path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(UpstreamCompressor.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._compressor_fn = None

    def forward(self, x, positions, cu_seqlens=None):
        ratio = self.compress_ratio
        seqlen = x.shape[0]
        if seqlen == 0:
            return super().forward(x, positions, cu_seqlens)
        if cu_seqlens is None:
            if seqlen % ratio:
                return super().forward(x, positions, cu_seqlens)
            cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=x.device)
        else:
            native_cu_seqlens = cu_seqlens
            # Native metadata reserves unused document slots by repeating the
            # final boundary. Keep the first end marker and every internal
            # boundary, including real empty documents, for backend dispatch.
            end_index = torch.searchsorted(cu_seqlens, cu_seqlens[-1], right=False)
            indices = torch.arange(cu_seqlens.numel(), device=cu_seqlens.device)
            cu_seqlens = cu_seqlens[indices <= end_index]
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            if not bool(((lengths > 0) & (lengths % ratio == 0)).all()):
                return super().forward(x, positions, native_cu_seqlens)

        state_block_table, state_cache, cu_seqlens, seqused = _state_inputs(
            cu_seqlens,
            x.unsqueeze(0),
            ratio=ratio,
            head_dim=self.head_dim,
        )
        if self._compressor_fn is None:
            self._compressor_fn = load_compressor()
        pooled = self._compressor_fn(
            x.contiguous(),
            self.wkv.weight,
            self.wgate.weight,
            state_cache,
            self.ape.float(),
            cmp_ratio=ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            coff=1 + int(self.overlap),
            cache_mode=1,
        )
        pooled = self.norm(pooled[: seqlen // ratio].to(x.dtype))
        nope, rope = pooled.split([self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1)
        comp_positions = (
            positions[::ratio] if positions is not None else torch.arange(0, seqlen, ratio, device=x.device)
        )
        rope = self.rope(rope.unsqueeze(1), positions=comp_positions).squeeze(1)
        return torch.cat((nope, rope), dim=-1)
