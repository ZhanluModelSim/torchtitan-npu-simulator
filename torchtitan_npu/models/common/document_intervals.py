# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The per-document intervals of a packed row's compressed pool.

Two readers need them: the attention core, which derives a query's causally reachable slot
range for the layers that carry no selector, and the selector's second pass, which turns a
document-local selection back into the row-wide entries the keys live in.  They live here
because the selector cannot import the attention module (which imports it).
"""

import torch


def document_intervals(
    cu_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor, *, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(pool_start_L1, pool_end_L1, local_L1)``: each query's own document's compressed
    slots, and the query's place in it.

    ``cu_seqlens`` are the document boundaries and ``cu_seqlens_k`` the compressed pool's.
    ``pool_start`` is at once the first slot a query may reach and the shift between a
    selection's per-document numbering and the row-wide one the pooled keys are stored in.
    """
    positions = torch.arange(num_tokens, device=cu_seqlens.device)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    pool_start_L1 = cu_seqlens_k.index_select(0, documents).unsqueeze(-1)
    pool_end_L1 = cu_seqlens_k.index_select(0, documents + 1).unsqueeze(-1)
    local_L1 = (positions - cu_seqlens.index_select(0, documents)).unsqueeze(-1)
    return pool_start_L1, pool_end_L1, local_L1
