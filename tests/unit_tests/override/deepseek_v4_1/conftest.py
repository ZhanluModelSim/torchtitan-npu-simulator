# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared builders for the V4.1 sparse-attention override tests."""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata
from torchtitan_npu.override.deepseek_v4_1.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
    AscV41Metadata,
    build_fused_metadata,
)


def packed_positions(bounds: list[int]) -> torch.Tensor:
    """One packed row whose documents have these boundaries: an ``arange`` per document."""
    return torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)


def reference_row(positions: torch.Tensor, *, ratio: int) -> DeepSeekV41Metadata:
    """The row a fused entry starts from, built the way the reference backend builds it.

    ``cu_seq_k`` is the backend's own ratio applied to the row's boundaries, so the row handed
    to :func:`build_fused_metadata` has to be built at the ratio of the entry it feeds.  A
    reference config carries no kernel geometry at all -- that is ``test_fused_geometry``'s pin --
    which is why the window stated here is incidental to the row.
    """
    config = CompressedSparseInnerAttention2.Config(window_size=128, softmax_scale=0.5, compress_ratio=ratio)
    return config.build_attention_metadata(positions)


@pytest.fixture
def v41_metadata():
    """Build one geometry's fused metadata for a packed row of these document bounds.

    The test states the bounds it wants to exercise; the positions that produce them are
    rebuilt here, so the metadata comes from the same builders the attention backends call
    rather than being hand-rolled.  One ``arange`` per document, concatenated, is what a packed
    loader emits, and the entry is the row plus one ratio's frames and metadata -- what the
    AscendC attention layer is handed.

    The metadata is sized by the geometry and every build produces it -- the port has no inline
    operator path -- so an entry is always built *for a layer*: state that layer's geometry, or
    hand over its ``port`` config and let it build the entry, which is what a run does.  One of
    the two is required precisely because a default would silently size the metadata for the
    released shape, which no test here runs.
    """

    def _build(
        bounds: list[int],
        *,
        ratio: int = 2,
        geometry: AicpuGeometry | None = None,
        port=None,
    ) -> AscV41Metadata:
        if (geometry is None) == (port is None):
            raise ValueError("state either the geometry this entry is for or the port config that builds it")
        positions = packed_positions(bounds)
        if port is not None:
            return port.build_attention_metadata(positions)
        return build_fused_metadata(reference_row(positions, ratio=ratio), ratio=ratio, geometry=geometry)

    return _build
