# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture support for system3's host-offload Engram hash tables.

The production table is CPU-resident, shards itself over EP and routes
deduplicated row requests through all-to-all exchanges whose split sizes
are data-dependent (``torch.unique``/``bincount``), which Meta capture
cannot execute. This bridge replaces ``System3HostHashTable._lookup_rows``
on meta tensors with a shape-visible decomposition: the two host
transfers of the official local path stay visible as ``_to_copy`` nodes,
and at EP > 1 the three exchanges run as native functional all-to-alls
with balanced (uniform) split geometry, mirroring the EP token dispatcher
bridge in ``compat/routing.py``. Real-device execution is untouched.
"""

import torch
from torch.distributed._functional_collectives import all_to_all_single


def install_system3_engram_meta_support(scope):
    """Give the host hash table a Meta-visible official lookup."""
    from torchtitan_npu.models.system3.model import System3HostHashTable

    original_lookup = System3HostHashTable._lookup_rows

    def _meta_lookup_rows(self, flat_ids):
        if flat_ids.device.type != "meta":
            return original_lookup(self, flat_ids)
        b, s, nh = flat_ids.shape
        head_dim = self.weight.shape[1]
        device = flat_ids.device
        if self._ep_size <= 1:
            # Official local path: host gather, then one rows H2D transfer.
            # The ids D2H (b*s*nh int64, a few hundred bytes) cannot be
            # materialized from meta and is too small to model; the captured
            # graph keeps the dominant rows transfer visible with its real
            # byte count.
            rows_host = torch.empty((flat_ids.numel(), head_dim), dtype=self.weight.dtype, device="cpu")
            return rows_host.to(device=device).view(b, s, nh, head_dim)
        # Official routed path with balanced split geometry: the count
        # exchange, the row-id exchange and the row exchange each appear as
        # native functional all-to-alls; the owner-local gather contributes
        # the H2D transfer of the received rows.
        ep_size = self._ep_size
        mesh = self._ep_mesh
        assert mesh is not None, "EP-sharded engram lookup has no mesh"
        counts = torch.zeros(ep_size, dtype=torch.int64, device=device)
        all_to_all_single(counts, None, None, mesh)
        total = b * s * nh
        quotient, remainder = divmod(total, ep_size)
        splits = [quotient + int(rank < remainder) for rank in range(ep_size)]
        received_ids = all_to_all_single(flat_ids.reshape(-1), splits, splits, mesh)
        rows_owner_host = torch.empty(
            (received_ids.shape[0], head_dim),
            dtype=self.weight.dtype,
            device="cpu",
        )
        rows_owner = rows_owner_host.to(device=device)
        rows = all_to_all_single(rows_owner, splits, splits, mesh)
        return rows.view(b, s, nh, head_dim)

    scope.replace(System3HostHashTable, "_lookup_rows", _meta_lookup_rows)
    return "host-table lookup decomposition: visible transfers (EP=1) / balanced functional a2a (EP>1)"


__all__ = ["install_system3_engram_meta_support"]
