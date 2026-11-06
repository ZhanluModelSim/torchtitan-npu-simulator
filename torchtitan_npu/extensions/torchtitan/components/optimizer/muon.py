# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed DistMuon configuration for optional NPU replica deduplication."""

from dataclasses import dataclass

from torchtitan.components.optim import DistMuon as _DistMuon


class DistMuon(_DistMuon):
    """Build DistMuon, then install the requested replica ownership plan."""

    @dataclass(kw_only=True, slots=True)
    class Config(_DistMuon.Config):
        enable_hsdp_replica_dedup: bool = False

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if config.enable_hsdp_replica_dedup:
            from torchtitan_npu.patches.torchtitan.distributed.flex_shard.replica_dedup import configure_dist_muon

            configure_dist_muon(self)
