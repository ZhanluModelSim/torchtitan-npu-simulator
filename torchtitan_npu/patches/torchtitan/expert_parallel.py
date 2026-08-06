# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/distributed/expert_parallel.py

The NPU GMM converter fuses w1+w3 into w13 and sets w1/w3 to None.
The upstream TensorParallel._partition_fn and ExpertTensorParallel._partition_fn
unconditionally access module.w1 / module.w3, which crashes when those are None.
This patch adds None-guards and w13 sharding support.

It also adds a fake-backend path for ExpertParallel token dispatch. PyTorch's
FakeProcessGroup returns tensors with valid metadata but arbitrary values, while
MoE expert parallelism uses all-to-all exchanged token counts as split sizes.
"""

import torch
import torch.nn as nn
import torchtitan.distributed.expert_parallel as ep_module
from torch.distributed import _functional_collectives as funcol
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, distribute_tensor

from torchtitan_npu.distributed.process_group import is_fake_process_group


def _distribute_w13_interleaved(w13_param, device_mesh):
    """Distribute w13=[w1|w3] so each rank gets [w1_shard|w3_shard].

    Shard(1) on w13 would give contiguous chunks that break the w1|w3
    boundary (e.g. rank 0 gets only w1 data, rank N-1 gets only w3 data).
    npu_swiglu expects [w1_out|w3_out] on every rank, so we must split,
    shard separately, then re-concatenate local shards.
    """
    if isinstance(w13_param, DTensor):
        w13_full = w13_param.full_tensor()
    else:
        w13_full = w13_param.data if isinstance(w13_param, nn.Parameter) else w13_param

    w1_full, w3_full = torch.chunk(w13_full, 2, dim=1)
    w1_dt = distribute_tensor(w1_full, device_mesh, [Shard(1)])
    w3_dt = distribute_tensor(w3_full, device_mesh, [Shard(1)])
    w1_local = w1_dt.to_local()
    w3_local = w3_dt.to_local()
    w13_local = torch.cat([w1_local, w3_local], dim=1)
    return DTensor.from_local(w13_local, device_mesh, [Shard(1)], run_check=False)


def _tp_partition_fn(self, name, module, device_mesh):
    if not hasattr(module, "w2"):
        return

    # w1 has shape (experts, out_dim, in_dim)
    if module.w1 is not None:
        module.register_parameter("w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)])))

    # w2 has shape (experts, in_dim, out_dim)
    module.register_parameter(
        "w2",
        nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
    )

    # w3 has shape (experts, out_dim, in_dim)
    if module.w3 is not None:
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )

    # w13 has shape (experts, out_dim*2, in_dim), fused w1 and w3 for GMM
    if getattr(module, "w13", None) is not None:
        w13_dt = _distribute_w13_interleaved(module.w13, device_mesh)
        module.register_parameter("w13", nn.Parameter(w13_dt))


def _distribute_w13_interleaved_etp(w13_param, device_mesh):
    """Distribute w13=[w1|w3] under ETP with [Shard(0), Shard(1)] placements.

    Same w1|w3 boundary issue as _distribute_w13_interleaved but for 2D mesh.
    """
    if isinstance(w13_param, DTensor):
        w13_full = w13_param.full_tensor()
    else:
        w13_full = w13_param.data if isinstance(w13_param, nn.Parameter) else w13_param

    w1_full, w3_full = torch.chunk(w13_full, 2, dim=1)
    w1_dt = distribute_tensor(w1_full, device_mesh, [Shard(0), Shard(1)])
    w3_dt = distribute_tensor(w3_full, device_mesh, [Shard(0), Shard(1)])
    w1_local = w1_dt.to_local()
    w3_local = w3_dt.to_local()
    w13_local = torch.cat([w1_local, w3_local], dim=1)
    return DTensor.from_local(w13_local, device_mesh, [Shard(0), Shard(1)], run_check=False)


def _etp_partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
    if not hasattr(mod, "w2"):
        return

    if mod.w1 is not None:
        mod.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(
                    mod.w1,  # pyrefly: ignore [bad-argument-type]
                    device_mesh,
                    [Shard(0), Shard(1)],
                )
            ),
        )

    mod.register_parameter(
        "w2",
        nn.Parameter(
            distribute_tensor(
                mod.w2,  # pyrefly: ignore [bad-argument-type]
                device_mesh,
                [Shard(0), Shard(2)],
            )
        ),
    )

    if mod.w3 is not None:
        mod.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(
                    mod.w3,  # pyrefly: ignore [bad-argument-type]
                    device_mesh,
                    [Shard(0), Shard(1)],
                )
            ),
        )

    if getattr(mod, "w13", None) is not None:
        w13_dt = _distribute_w13_interleaved_etp(mod.w13, device_mesh)
        mod.register_parameter("w13", nn.Parameter(w13_dt))


_ORIG_EXPERT_TOKEN_DISPATCH = ep_module.ExpertParallel._token_dispatch
_ORIG_EXPERT_TOKEN_COMBINE = ep_module.ExpertParallel._token_combine


def _expert_parallel_token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh):
    if not is_fake_process_group(device_mesh.get_group()):
        return _ORIG_EXPERT_TOKEN_DISPATCH(self, mod, inputs, device_mesh)

    routed_input, num_tokens_per_expert = inputs
    ep_degree = device_mesh.shape[0]
    num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
    num_routed_tokens = routed_input.shape[0]

    # The simulator forces round-robin routing, so every EP destination receives
    # an analytically-known contiguous expert range. Avoid reading values from
    # meta token-count tensors while still executing the fake all-to-all so the
    # communication recorder sees the real collective.
    num_experts = num_tokens_per_expert.shape[0]
    tokens_per_expert, extra_experts = divmod(
        num_routed_tokens,
        num_experts,
    )
    self.input_splits = []
    for ep_rank in range(ep_degree):
        expert_start = ep_rank * num_local_experts
        num_extra = max(
            0,
            min(extra_experts, expert_start + num_local_experts)
            - expert_start,
        )
        self.input_splits.append(
            tokens_per_expert * num_local_experts + num_extra
        )
    self.output_splits = list(self.input_splits)
    routed_input = funcol.all_to_all_single_autograd(
        routed_input,
        self.output_splits,
        self.input_splits,
        device_mesh.get_group(),
    )

    self.input_shape = routed_input.shape
    self.permuted_indices = torch.arange(
        routed_input.shape[0],
        device=routed_input.device,
    )
    num_tokens_per_local_expert = torch.empty(
        num_local_experts,
        dtype=num_tokens_per_expert.dtype,
        device=num_tokens_per_expert.device,
    )
    return routed_input, num_tokens_per_local_expert


def _expert_parallel_token_combine(self, mod: nn.Module, routed_output: torch.Tensor, device_mesh):
    if not is_fake_process_group(device_mesh.get_group()):
        return _ORIG_EXPERT_TOKEN_COMBINE(self, mod, routed_output, device_mesh)

    routed_output = ep_module._unpermute(
        routed_output,
        self.input_shape,
        self.permuted_indices,
    )
    return funcol.all_to_all_single_autograd(
        routed_output,
        self.input_splits,
        self.output_splits,
        device_mesh.get_group(),
    )


_PARTITION_FN = "_partition_fn"
setattr(ep_module.TensorParallel, _PARTITION_FN, _tp_partition_fn)
if hasattr(ep_module, "ExpertTensorParallel"):
    setattr(ep_module.ExpertTensorParallel, _PARTITION_FN, _etp_partition_fn)
ep_module.ExpertParallel._token_dispatch = _expert_parallel_token_dispatch
ep_module.ExpertParallel._token_combine = _expert_parallel_token_combine
