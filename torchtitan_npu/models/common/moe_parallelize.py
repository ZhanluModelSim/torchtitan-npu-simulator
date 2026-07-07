# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.distributed.expert_parallel import (
    DeepEPExpertParallel,
    ExpertParallel,
    TensorParallel,
    TorchAOExpertParallel,
)

logger = logging.getLogger(__name__)


def apply_sequence_sharded_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    comm_backend: str = "standard",
    hybridep_non_blocking_expert_capacity_factor: float | None = None,
    pad_multiple: int | None = None,
):
    assert tp_mesh is not None or ep_mesh is not None, f"""
        At least one of Tensor Parallel mesh (tp_mesh) or Expert Parallel mesh (ep_mesh) must be provided.
        Current status: tp_mesh={tp_mesh}, ep_mesh={ep_mesh}
        """

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Shard(1),),
                    use_local_input=True,
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": SequenceParallel(sequence_dim=0, use_local_output=True),
            }
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts": PrepareModuleInput(
                            input_layouts=(Shard(0),),
                            desired_input_layouts=(Replicate(),),
                        ),
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(output_layouts=Shard(0)),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            experts_mesh = ep_mesh
            if comm_backend in ("deepep", "hybridep"):
                if comm_backend == "deepep" and pad_multiple is not None:
                    raise ValueError(
                        "DeepEP does not support pad_multiple. Use hybridep or standard comm backend instead."
                    )
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts
                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                    comm_backend=comm_backend,
                    hybridep_non_blocking_expert_capacity_factor=hybridep_non_blocking_expert_capacity_factor,
                    pad_multiple=pad_multiple,
                )
                logger.info(f"Applying {comm_backend.upper()} to MoE layer")
            elif pad_multiple is not None:
                experts_plan = TorchAOExpertParallel(pad_multiple)
            elif comm_backend == "standard":
                experts_plan = ExpertParallel()
            else:
                raise ValueError(f"Unsupported MoE communication backend: {comm_backend!r}")
        else:
            raise NotImplementedError("ETP is not supported currently")

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )
