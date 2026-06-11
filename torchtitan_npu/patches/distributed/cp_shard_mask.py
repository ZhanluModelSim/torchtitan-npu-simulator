# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch torchtitan.distributed.context_parallel.cp_shard to dispatch custom
# attention-mask types to the CP mask-handler registry.
#
# Upstream ``cp_shard`` sequence-shards ``attention_masks`` (expects a
# BlockMask/dict). Backends whose CP metadata is a custom object (e.g.
# DeepSeek-V4 SMLA) must not be sharded — each rank needs its own per-rank
# shapes. If a handler owns the mask, this wrapper shards only the buffers and
# returns the handler's per-rank result; otherwise it passes through. Stays
# backend-agnostic — backend logic lives in the registered handler.

from __future__ import annotations

from functools import wraps

import torchtitan.distributed.context_parallel as titan_cp
from torchtitan.tools.logging import logger

from torchtitan_npu.distributed.context_parallel import adjust_cp_mask


def _patch_cp_shard_mask_dispatch() -> None:
    original = titan_cp.cp_shard

    @wraps(original)
    def wrapper(
        cp_mesh,
        inputs,
        attention_masks,
        load_balancer_type="headtail",
        input_seq_dim=1,
    ):
        handled, adjusted = adjust_cp_mask(attention_masks, cp_mesh)
        if handled:
            # A mask handler's per-rank metadata (e.g. DeepSeek-V4 SMLA) is
            # computed for CONTIGUOUS sharding. A head-tail / ptrr load balancer
            # reorders tokens, which the metadata does not account for, so reject
            # it loudly instead of silently corrupting training.
            if load_balancer_type is not None:
                raise ValueError(
                    "Context-Parallel attention masks taken over by a mask "
                    "handler (e.g. DeepSeek-V4 SMLA) require contiguous sequence "
                    "sharding; set parallelism.context_parallel_load_balancer="
                    f"None (got {load_balancer_type!r})."
                )
            sharded_inputs, _ = original(cp_mesh, inputs, None, None, input_seq_dim)
            return sharded_inputs, adjusted

        return original(
            cp_mesh, inputs, attention_masks, load_balancer_type, input_seq_dim
        )

    titan_cp.cp_shard = wrapper  # pyrefly: ignore [bad-assignment]
    logger.info("[Patch] Registered Context-Parallel cp_shard mask-handler dispatch.")


_patch_cp_shard_mask_dispatch()
