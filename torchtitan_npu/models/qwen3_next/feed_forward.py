# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next feed-forward modules: dense MLP (upstream) and sparse MoE.

The MoE block strictly mirrors the MindSpeed-LLM fsdp2 implementation
(``mindspeed_llm/fsdp2/models/qwen3_next/modeling_qwen3_next.py``):

- ``Qwen3NextMoeExperts``: fused ``gate_up_proj`` [E*hidden, 2*inter] /
  ``down_proj`` [E*inter, hidden] parameters and the grouped-GEMM ->
  ``npu_swiglu`` -> grouped-GEMM core. Token permute/unpermute live at the
  MoE boundary so the torchtitan ExpertParallel dispatcher (which wraps the
  experts module with the ``(routed_input, num_tokens_per_expert)`` contract)
  stays intact; routing probs are folded into the permuted tokens before the
  first GEMM (numerically equivalent to MindSpeed folding them into the
  unpermute).
- ``Qwen3NextMoeSharedExpert``: dense SwiGLU MLP with a sigmoid gate.

The eager helpers below provide the same semantics with plain aten ops so the
model also runs without the fused NPU kernels; the simulator binds virtual
fused operators (``qwen3_next_converter``) so the captured workload matches
the MindSpeed production op stream.
"""

from dataclasses import dataclass

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


def _token_permute(
    hidden_states: torch.Tensor, selected_experts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager equivalent of ``npu_moe_token_permute`` (expert-major sort)."""
    flat_indices = selected_experts.reshape(-1).to(torch.int64)
    row_ids_map = torch.argsort(flat_indices, stable=True)
    permuted_hidden_states = hidden_states.index_select(0, row_ids_map)
    return permuted_hidden_states, row_ids_map


def _swiglu(fc1_output: torch.Tensor) -> torch.Tensor:
    """Eager equivalent of ``torch_npu.npu_swiglu(x, dim=-1)``."""
    gate, up = fc1_output.chunk(2, dim=-1)
    return F.silu(gate) * up


def _token_unpermute(
    routed_output: torch.Tensor,
    row_ids_map: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Eager equivalent of ``npu_moe_token_unpermute`` (probs pre-folded)."""
    token_ids = row_ids_map // top_k
    output = torch.zeros(
        (num_tokens, routed_output.shape[-1]),
        dtype=routed_output.dtype,
        device=routed_output.device,
    )
    output.index_add_(0, token_ids, routed_output)
    return output


def _record_all_to_all(tensor: torch.Tensor, output: torch.Tensor, group) -> None:
    from torchtitan_npu.simulator import meta_env
    from torchtitan_npu.simulator.capture.comm_events import (
        _record_comm_with_l0,
        get_active_recorder,
    )

    recorder = get_active_recorder()
    if recorder is None:
        return
    meta_env._comm_layer = "L1"
    _record_comm_with_l0(recorder, "all_to_all", group, tensor, output)


def _record_allgather(tensor: torch.Tensor, output: torch.Tensor, group) -> None:
    from torchtitan_npu.simulator import meta_env
    from torchtitan_npu.simulator.capture.comm_events import (
        _record_comm_with_l0,
        get_active_recorder,
    )

    recorder = get_active_recorder()
    if recorder is None:
        return
    meta_env._comm_layer = "L1"
    _record_comm_with_l0(recorder, "allgather", group, tensor, output)


def _synthetic_comm(op_name: str, tensor: torch.Tensor, output: torch.Tensor, group):
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
    from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

    capture = get_active_capture()
    module_path = (
        capture.module_path_tracker.current_path()
        if capture is not None and capture.module_path_tracker is not None
        else ""
    )

    def produce_output():
        if op_name == "allgather":
            _record_allgather(tensor, output, group)
        else:
            _record_all_to_all(tensor, output, group)
        return output

    return run_synthetic_op(
        f"comm.{op_name}",
        inputs=[tensor],
        output_factory=produce_output,
        module_path=module_path,
        record_op=False,
    )


def _sim_all_gather_counts(tensor: torch.Tensor, group) -> torch.Tensor:
    """``gather_along_first_dim_expert_parallel`` ([E] -> [EP*E] counts)."""
    ep_size = torch.distributed.get_world_size(group=group)
    if ep_size == 1:
        return tensor
    output = torch.empty(
        [tensor.shape[0] * ep_size], dtype=tensor.dtype, device=tensor.device
    )
    return _synthetic_comm("allgather", tensor, output, group)


def _synthetic_all_to_all(tensor: torch.Tensor, group, out_rows: int) -> torch.Tensor:
    output = torch.empty(
        [out_rows] + list(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    return _synthetic_comm("all_to_all", tensor, output, group)


class _SimAllToAllV(torch.autograd.Function):
    """Simulated a2a-v with an autograd-linked reverse exchange.

    The forward records the dispatch/combine ``comm.all_to_all``; the
    backward records the reverse exchange so the measured stream's
    backward a2as appear and gradients flow to the experts.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, out_rows: int):
        ctx.group = group
        ctx.in_rows = tensor.shape[0]
        ctx.out_rows = out_rows
        return _synthetic_all_to_all(tensor, group, out_rows)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _synthetic_all_to_all(grad_output, ctx.group, ctx.in_rows)
        return grad_input, None, None


def _sim_all_to_all_v(tensor: torch.Tensor, group, out_rows: int) -> torch.Tensor:
    """MindSpeed ``dist_ops.all_to_all``: a2a-v with unequal splits.

    The simulated exchange records one ``comm.all_to_all`` per direction
    (the autograd backward records the reverse); the output row count is
    the analytically derived received-row total.
    """
    if torch.distributed.get_world_size(group=group) == 1:
        return tensor
    return _SimAllToAllV.apply(tensor, group, out_rows)


def _sim_d2h_copy(tensor: torch.Tensor, real_values: list[int]) -> torch.Tensor:
    """The dispatcher's ``.to("cpu", non_blocking=True)`` split transfer.

    Real execution copies to CPU; under meta capture the copy is recorded
    (aten._to_copy.default, matching the measured D2H stream) and the
    python-side ``real_values`` stand in for the unusable meta values.
    """
    if tensor.device.type != "meta":
        return tensor.to(torch.device("cpu"), non_blocking=True)
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
    from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

    capture = get_active_capture()
    module_path = (
        capture.module_path_tracker.current_path()
        if capture is not None and capture.module_path_tracker is not None
        else ""
    )
    out = torch.tensor(real_values, dtype=tensor.dtype, device="cpu")
    return run_synthetic_op(
        "aten._to_copy.default",
        inputs=[tensor],
        output_factory=lambda: out,
        module_path=module_path,
    )


def _eager_permute(
    tokens: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """MindSpeed fsdp2 eager permute: stable argsort + row gather."""
    topk = 1 if indices.dim() == 1 else indices.size(1)
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True).to(
        indices.dtype
    )
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def _eager_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """MindSpeed fsdp2 eager unpermute: zeros + index_copy_ scatter back."""
    num_tokens, topk = (
        (permuted_tokens.size(0), 1)
        if probs is None
        else (probs.numel(), probs.size(1))
    )
    unpermuted_tokens = torch.zeros(
        [num_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices.long(), permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)


def _experts_ep_forward(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> torch.Tensor:
    """MindSpeed eager-dispatcher experts forward (``ep > 1``).

    Mirrors ``mindspeed_llm/fsdp2/distributed/expert_parallel/dispatcher.py``
    (``dispatch_mlp_combine``) op for op: count preprocess with an
    all-gather over the EP group, a2a-v token dispatch, per-expert matmul
    loop, and the combine a2a with the final probs-weighted unpermute.

    Under the simulator's meta capture the routing counts carry no values,
    but ``debug.moe_force_load_balance`` makes the routing uniform, so the
    split sizes and grouped-matmul loop bounds are derived analytically
    (N*K/E tokens per expert, N*K/ep rows exchanged per rank pair) while
    the recorded op stream keeps the exact dispatcher call pattern.
    """
    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_states_shape[-1])
    gate_up_proj, down_proj = self._view_experts_weight()

    ep_group = self.ep_group
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_global_experts = self.num_global_experts
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    num_rows = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    tokens_per_expert_num = (num_rows * top_k) // num_global_experts
    if tokens_per_expert_num * num_global_experts != num_rows * top_k:
        raise ValueError(
            "Uniform routing assumed by the simulated eager dispatcher is "
            f"violated: {num_rows}*{top_k} rows over {num_global_experts} "
            "experts is not divisible"
        )
    rows_per_rank = (num_rows * top_k) // ep_size

    # dispatch_preprocess: [B*S*K] -> [E] counts, all-gathered to [EP*E]
    num_local_tokens_per_expert = torch.bincount(
        top_k_index.reshape(-1), minlength=num_global_experts
    )
    num_global_tokens_per_expert = _sim_all_gather_counts(
        num_local_tokens_per_expert, ep_group
    )
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(
        ep_size, num_global_experts
    )[:, local_experts_start_id:local_experts_end_id]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    input_split = _sim_d2h_copy(
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1),
        [tokens_per_expert_num * num_local_experts] * ep_size,
    )
    output_splits = _sim_d2h_copy(
        num_global_tokens_per_local_expert.sum(axis=-1),
        [rows_per_rank] * ep_size,
    )
    global_indices = torch.repeat_interleave(
        self.expert_ids_per_ep_rank,
        num_global_tokens_per_local_expert.ravel(),
        output_size=rows_per_rank,
    )

    # alltoall_dispatch: permute -> a2a -> permute
    hidden_states, unpermute_indices1 = _eager_permute(hidden_states, top_k_index)
    torch.accelerator.current_stream().synchronize()
    hidden_states = _sim_all_to_all_v(hidden_states, ep_group, rows_per_rank)
    hidden_states, unpermute_indices2 = _eager_permute(hidden_states, global_indices)

    # fused experts computation: npu_grouped_matmul (single fused op per
    # grouped GEMM), silu*up between
    fc1_output = _fused_grouped_matmul(hidden_states, gate_up_proj, tokens_per_expert_num)
    gate, up = fc1_output.chunk(2, dim=-1)
    act = F.silu(gate) * up
    fc2_out = _fused_grouped_matmul(act, down_proj, tokens_per_expert_num)

    # alltoall_combine: unpermute -> a2a -> probs-weighted unpermute
    hidden_states = _eager_unpermute(fc2_out, unpermute_indices2)
    hidden_states = _sim_all_to_all_v(hidden_states, ep_group, num_rows * top_k)
    hidden_states = _eager_unpermute(
        hidden_states, unpermute_indices1, probs=top_k_weights
    )
    return hidden_states.view(*hidden_states_shape)


def _record_grouped_matmul(
    inputs: list[torch.Tensor], out_shape: list[int], dtype: torch.dtype
) -> torch.Tensor:
    """Record one fused ``npu_grouped_matmul`` op under meta capture."""
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
    from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

    capture = get_active_capture()
    module_path = (
        capture.module_path_tracker.current_path()
        if capture is not None and capture.module_path_tracker is not None
        else ""
    )
    out = torch.empty(out_shape, dtype=dtype, device=inputs[0].device)
    return run_synthetic_op(
        "npu.npu_grouped_matmul.default",
        inputs=inputs,
        output_factory=lambda: out,
        module_path=module_path,
    )


class _SimGroupedMatmul(torch.autograd.Function):
    """``mindspeed.ops.grouped_matmul._GroupedMatmul`` (npu_grouped_matmul).

    Forward: one fused grouped-matmul op over the expert segments. The
    measured backward is explicit (custom autograd Function): one fused
    op for the input grads (transposed weights) and one fused op for the
    weight grads (``split_item=3 / group_type=2``), stacked per expert —
    no per-expert autograd ops (unlike the eager python loop, which the
    measured stream does not contain).
    """

    @staticmethod
    def forward(ctx, inputs, weights, group_list):
        ctx.save_for_backward(inputs, weights, group_list)
        out_shape = [inputs.shape[0], weights.shape[2]]
        return _record_grouped_matmul(
            [inputs, weights, group_list], out_shape, inputs.dtype
        )

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, group_list = ctx.saved_tensors
        grad_input = _record_grouped_matmul(
            [grad_output, weights.transpose(1, 2), group_list],
            [grad_output.shape[0], weights.shape[1]],
            grad_output.dtype,
        )
        grad_weight = _record_grouped_matmul(
            [inputs.transpose(0, 1), grad_output, group_list],
            list(weights.shape),
            weights.dtype,
        )
        return grad_input, grad_weight, None


def _fused_grouped_matmul(
    inputs: torch.Tensor, weights: torch.Tensor, tokens_per_expert: int
) -> torch.Tensor:
    """Fused grouped matmul with the analytic uniform group list."""
    m_split = torch.full(
        (weights.shape[0],),
        tokens_per_expert,
        dtype=torch.int64,
        device=inputs.device,
    )
    return _SimGroupedMatmul.apply(inputs, weights, m_split)


class Qwen3NextGroupedExperts(nn.Module):
    """Routed experts mirroring MindSpeed ``Qwen3NextMoeExperts``.

    Fused weight layout: ``gate_up_proj`` [E*hidden, 2*inter] (viewed
    [E, hidden, 2*inter], trans_b=False in the GMM) and ``down_proj``
    [E*inter, hidden] (viewed [E, inter, hidden]). The expert dimension is
    dim 0 so ``ExpertParallel`` shards both parameters with Shard(0). The
    module keeps torchtitan's ``(routed_input, num_tokens_per_expert)``
    contract: token permute/unpermute are handled at the MoE boundary.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts * hidden_size, 2 * intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts * intermediate_size, hidden_size)
        )

    def _view_experts_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )
        # The fused flat layout [E*hidden, 2*inter] shards row-wise under EP
        # (Shard(0)), which evenly splits whole experts; derive the local
        # expert count from the actual storage instead of self.num_experts.
        local_experts = gate_up_proj.shape[0] // self.hidden_size
        gate_up_proj = gate_up_proj.view(local_experts, self.hidden_size, -1)
        down_proj = down_proj.view(local_experts, -1, self.hidden_size)
        return gate_up_proj, down_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.to_local()
        gate_up_proj, down_proj = self._view_experts_weight()

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        fc1_output = torch._grouped_mm(hidden_states, gate_up_proj, offs=offsets)
        fc1_activation = _swiglu(fc1_output)
        fc2_out = torch._grouped_mm(fc1_activation, down_proj, offs=offsets)
        return fc2_out


class Qwen3NextMoE(nn.Module):
    """Sparse MoE block mirroring MindSpeed ``Qwen3NextSparseFusedMoeBlock``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int = 2048
        num_experts: int = 512
        top_k: int = 10
        moe_intermediate_size: int = 512
        shared_expert_intermediate_size: int = 512
        norm_topk_prob: bool = True
        debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = config.norm_topk_prob
        self.debug_force_load_balance = config.debug_force_load_balance

        # gating
        self.gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        ).build()
        self.experts = Qwen3NextGroupedExperts(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )

        # shared expert
        self.shared_expert = _SharedExpertMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=1,
            bias=False,
        ).build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(hidden_states, DTensor):
            if hidden_states.device_mesh.ndim != 1:
                raise ValueError(
                    "Qwen3-Next MoE expects a 1D TP DTensor input, "
                    f"got {hidden_states.device_mesh.ndim}D"
                )
            # Plain to_local: the residual stream travels as plain local
            # tensors between TP style boundaries (the simulator's meta
            # from_local-backward patch bridges plain grads). The old
            # grad_placements=(Partial(),) marker only mattered for the
            # ep-only implicit-rewrap flow and breaks the backward shapes
            # under TP.
            hidden_states = hidden_states.to_local()

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # gating
        gate_weight = (
            self.gate.weight.to_local()
            if isinstance(self.gate.weight, DTensor)
            else self.gate.weight
        )
        router_logits = F.linear(hidden_flat, gate_weight)
        routing_weights_f32 = F.softmax(router_logits, dim=1, dtype=torch.float)

        if self.debug_force_load_balance:
            # Value-forcing for analytic a2a splits: only the selected
            # indices are overridden; the weights flow through the real
            # softmax -> topk -> div -> cast graph so the router backward
            # chain (grad muls) is recorded exactly like the measured run
            # (a constant weight tensor would cut the gradient path).
            routing_weights, _ = torch.topk(
                routing_weights_f32, k=self.top_k, dim=-1
            )
            token_offsets = torch.arange(
                hidden_flat.shape[0], device=hidden_flat.device, dtype=torch.int64
            ) * self.top_k
            expert_offsets = torch.arange(
                self.top_k, device=hidden_flat.device, dtype=torch.int64
            )
            selected_experts = (
                token_offsets.unsqueeze(1) + expert_offsets.unsqueeze(0)
            ) % self.num_experts
        else:
            routing_weights_f32 = F.softmax(
                router_logits, dim=1, dtype=torch.float
            )
            routing_weights, selected_experts = torch.topk(
                routing_weights_f32, k=self.top_k, dim=-1
            )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(hidden_states.dtype)

        if getattr(self.experts, "_ep_dispatcher", False):
            # MindSpeed eager dispatcher (ep > 1): the experts module owns
            # permute/unpermute and the a2a exchange; probs are applied in
            # its final unpermute, so the block hands over raw routing.
            combined = self.experts(hidden_flat, selected_experts, routing_weights)
        else:
            # permute (npu_moe_token_permute semantics, probs pre-folded)
            permuted_hidden_states, row_ids_map = _token_permute(
                hidden_flat, selected_experts
            )
            tokens_per_expert = torch.histc(
                selected_experts.reshape(-1).float(),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )
            row_probs = routing_weights.reshape(-1).index_select(0, row_ids_map)
            permuted_hidden_states = (
                permuted_hidden_states * row_probs.to(permuted_hidden_states.dtype).unsqueeze(-1)
            )

            routed_output = self.experts(permuted_hidden_states, tokens_per_expert)

            # unpermute (npu_moe_token_unpermute semantics)
            combined = _token_unpermute(
                routed_output,
                row_ids_map,
                num_tokens=hidden_flat.shape[0],
                top_k=self.top_k,
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        combined = combined + shared_gate * shared_expert_output

        return combined.view(batch_size, sequence_length, hidden_dim)


class _SharedExpertMLP(nn.Module):
    """Dense SwiGLU MLP used as the Qwen3-Next shared expert."""

    def __init__(self, *, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear.Config(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        ).build()
        self.up_proj = Linear.Config(
            in_features=hidden_size,
            out_features=intermediate_size,
            bias=False,
        ).build()
        self.down_proj = Linear.Config(
            in_features=intermediate_size,
            out_features=hidden_size,
            bias=False,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
