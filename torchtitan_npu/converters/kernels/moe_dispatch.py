# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn
import torch_npu
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Partial
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.common.moe import MoE

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.permutation import (
    NPUMoeReRouting,
    NPUMoeTokenUnpermute,
)
from torchtitan_npu.converters.model_custom_interface import (
    ModelCustomConfig,
    ModelCustomConverter,
    ParallelizePlanUpdater,
)
from torchtitan_npu.converters.registry import register_model_converter
from torchtitan_npu.distributed.process_group import is_fake_process_group
from torchtitan_npu.models.deepseek_v4.moe import MoE as DeepSeekV4MoE

NpuTokenDispatchResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


class _LocalExpertInputs(NamedTuple):
    tokens: torch.Tensor
    top_scores: torch.Tensor
    selected_experts_indices: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    total_tokens: int
    dim: int


def _flatten_moe_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size, int, int]:
    original_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    return x, original_shape, dim, x.shape[0]


def _select_standard_moe_experts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.router(x, self.expert_bias)


def _shared_expert_output(self, x: torch.Tensor) -> torch.Tensor:
    if self.shared_experts is None:
        return torch.zeros_like(x)

    out = self.shared_experts(x)
    if isinstance(out, DTensor):
        out = out.to_local()
    return out


def _record_tokens_per_expert(self, num_tokens_per_expert: torch.Tensor) -> None:
    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)


def _run_local_experts(self, inputs: _LocalExpertInputs) -> torch.Tensor:
    indices = inputs.selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(inputs.tokens, indices)
    routed_scores, _ = torch_npu.npu_moe_token_permute(
        inputs.top_scores.reshape(-1).unsqueeze(-1),
        indices.reshape(-1, 1),
    )

    routed_output = self.experts(
        routed_input,
        inputs.num_tokens_per_expert,
        routed_scores,
    )

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        None,
    )
    return unpermuted.view(inputs.total_tokens, self.reorderer.top_k, inputs.dim).sum(dim=1)


def _npu_moe_forward(self, x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.to_local(grad_placements=(Partial(),))
    x, original_shape, dim, total_tokens = _flatten_moe_input(x)
    top_scores, selected_experts_indices, num_tokens_per_expert = _select_standard_moe_experts(self, x)
    out = _shared_expert_output(self, x)
    _record_tokens_per_expert(self, num_tokens_per_expert)
    expert_out = _run_local_experts(
        self,
        _LocalExpertInputs(x, top_scores, selected_experts_indices, num_tokens_per_expert, total_tokens, dim),
    )
    return (out + expert_out).reshape(original_shape)


def _adopt_module_state(wrapper_type: type[nn.Module], source: nn.Module) -> nn.Module:
    wrapper = cast("nn.Module", object.__new__(wrapper_type))
    nn.Module.__init__(wrapper)
    wrapper.__dict__.update(source.__dict__)
    return wrapper


class NpuMoE(MoE):
    def forward(self, x):
        return _npu_moe_forward(self, x)


class NpuDeepSeekV4MoE(DeepSeekV4MoE):
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return _npu_moe_forward_for_dsv4(self, x, input_ids)


def _npu_moe_forward_for_dsv4(self, x, input_ids):
    x, original_shape, dim, total_tokens = _flatten_moe_input(x)
    input_ids_flat = input_ids.flatten() if input_ids is not None else None
    bias = getattr(self, "expert_bias", None)
    top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, input_ids_flat, bias)

    _record_tokens_per_expert(self, num_tokens_per_expert)
    expert_out = _run_local_experts(
        self,
        _LocalExpertInputs(x, top_scores, selected_experts_indices, num_tokens_per_expert, total_tokens, dim),
    )
    out = _shared_expert_output(self, x)
    return (out + expert_out).reshape(original_shape)


class NpuMoeDispatchConverter(ModelCustomConverter):
    """Replace MoE modules with NPU MoE dispatch implementations."""

    def convert(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (NpuMoE, NpuDeepSeekV4MoE)):
                continue
            if isinstance(module, DeepSeekV4MoE):
                replace_module_with_name(model, name, _adopt_module_state(NpuDeepSeekV4MoE, module))
            elif isinstance(module, MoE):
                replace_module_with_name(model, name, _adopt_module_state(NpuMoE, module))


class NpuExpertParallel(ExpertParallel):
    def _compute_all_to_all_splits(
        self,
        num_tokens_per_expert: torch.Tensor,
        ep_degree: int,
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, list[int]]:
        # Generate the input/output splits for all-to-all and stash them on
        # self. Returns (num_tokens_per_expert_group, output_splits) for the
        # downstream local shuffle.
        group = device_mesh.get_group()
        is_fake = is_fake_process_group(group)
        with torch.no_grad():
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1).sum(dim=1).to(torch.device("cpu"), non_blocking=not is_fake)
            )
            if is_fake:
                num_tokens_per_expert_group = num_tokens_per_expert
                output_splits = input_splits
            else:
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=group,
                )
                # NOTE: this would incur a device-to-host sync
                output_splits = (
                    num_tokens_per_expert_group.view(ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()
        return num_tokens_per_expert_group, self.output_splits

    def _token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh) -> Any:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert, routed_scores = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        num_tokens_per_expert_group, output_splits = self._compute_all_to_all_splits(
            num_tokens_per_expert, ep_degree, device_mesh
        )

        is_fake = is_fake_process_group(device_mesh.get_group())
        if not is_fake:
            # perform all-to-all
            routed_input = all_to_all_single_autograd(
                routed_input,
                self.output_splits,
                self.input_splits,
                device_mesh.get_group(),
            )

            if routed_scores is not None:
                routed_scores = all_to_all_single_autograd(
                    routed_scores,
                    self.output_splits,
                    self.input_splits,
                    device_mesh.get_group(),
                )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout
        if is_fake:
            indices = (
                torch.arange(
                    num_local_experts,
                    dtype=torch.int64,
                    device=routed_input.device,
                )
                .repeat(ep_degree)
                .repeat_interleave(
                    num_tokens_per_expert_group.view(-1),
                    output_size=sum(output_splits),
                )
            )

            routed_input, self.permuted_indices = torch_npu.npu_moe_token_permute(routed_input, indices)
            if routed_scores is not None:
                routed_scores, _ = torch_npu.npu_moe_token_permute(routed_scores, indices)

            num_tokens_per_local_expert = num_tokens_per_expert_group.view(ep_degree, -1).sum(0)
        else:
            counts_2d = num_tokens_per_expert_group.view(ep_degree, -1)
            (
                routed_input,
                routed_scores,
                self.permuted_indices,
                num_tokens_per_local_expert,
            ) = NPUMoeReRouting.apply(routed_input, counts_2d, routed_scores)

        result: NpuTokenDispatchResult = (routed_input, num_tokens_per_local_expert, routed_scores)
        return result

    def _token_combine(self, mod: nn.Module, routed_output: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
        # Using NPUMoeTokenUnpermute.apply and npu_moe_token_unpermute is equivalent here,
        # and avoid storing tensor routed_output during backpropagation.
        routed_output = NPUMoeTokenUnpermute.apply(routed_output, self.permuted_indices, routed_output.shape)
        if is_fake_process_group(device_mesh.get_group()):
            return routed_output

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output


class NpuMoeDispatchParallelizePlanUpdater(ParallelizePlanUpdater):
    @classmethod
    def update(
        cls, parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None
    ) -> ParallelStyle | dict[str, ParallelStyle] | None:
        """Use the NPU MoE dispatch path for standard ExpertParallel."""
        if isinstance(parallelize_plan, ExpertParallel):
            return NpuExpertParallel()
        return parallelize_plan


@register_model_converter("npu_moe_dispatch")
class MoeDispatchModelConfig(ModelCustomConfig):
    model_converter = NpuMoeDispatchConverter
    parallelize_plan_updater = NpuMoeDispatchParallelizePlanUpdater
