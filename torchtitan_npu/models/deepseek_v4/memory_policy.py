# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4-specific GraphTrainer memory policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _find_fsdp_unshard_save_nodes,
    _make_full_memory_policy,
)
from torchtitan.experiments.graph_trainer.registry import (
    register_memory_policy,
)

from torchtitan_npu.patches.torchtitan.graph_trainer import memory_policy

if TYPE_CHECKING:
    import torch


@register_memory_policy("dsv4-mhc")
def _dsv4_mhc_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Apply full recomputation with MHC/MoE activation save overrides."""
    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=config.parallelism.pipeline_parallel_degree > 1,
    )
    force_save_nodes = _find_fsdp_unshard_save_nodes(gm) if not fsdp_reshard_after_forward else None

    policy_fn = memory_policy.make_node_override_memory_policy(
        base_policy=_make_full_memory_policy(),
        overrides={
            memory_policy.NodePolicyKey(
                target="aten.matmul.default",
                module_fqn="layers.*.attention.wo_b",
                occurrence=(1,),
            ): CheckpointPolicy.MUST_SAVE,
            memory_policy.NodePolicyKey(
                target="aten.add.Tensor",
                module_fqn="layers.*.moe",
                occurrence=(1,),
            ): CheckpointPolicy.MUST_SAVE,
        },
    )
    memory_policy.tag_sac_policy(
        gm,
        policy_fn=policy_fn,
        force_save_nodes=force_save_nodes,
        save_input_every_n_layers=4,
    )
    return gm
