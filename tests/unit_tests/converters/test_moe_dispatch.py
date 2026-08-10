# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from operator import attrgetter
from types import SimpleNamespace
from typing import Any, cast

import torch

from torchtitan_npu.converters.kernels import moe_dispatch, permutation
from torchtitan_npu.converters.kernels.permutation import NPUMoeReRouting


def _expert_parallel_module():
    from torchtitan.distributed import expert_parallel

    return expert_parallel


class _FakeDeviceMesh:
    shape = (2,)

    def __init__(self, group):
        self.group = group

    def get_group(self):
        return self.group


def _call_token_dispatch(parallel_style, inputs, device_mesh):
    token_dispatch = attrgetter("_token_dispatch")(parallel_style)
    return token_dispatch(torch.nn.Module(), inputs, device_mesh)


def _call_token_combine(parallel_style, routed_output, device_mesh):
    token_combine = attrgetter("_token_combine")(parallel_style)
    return token_combine(torch.nn.Module(), routed_output, device_mesh)


def test_moe_dispatch_config_wires_converter_and_plan_updater():
    expert_parallel = _expert_parallel_module()

    assert moe_dispatch.MoeDispatchModelConfig.model_converter is moe_dispatch.NpuMoeDispatchConverter
    assert (
        moe_dispatch.MoeDispatchModelConfig.parallelize_plan_updater
        is moe_dispatch.NpuMoeDispatchParallelizePlanUpdater
    )

    updater = moe_dispatch.MoeDispatchModelConfig.parallelize_plan_updater
    standard = updater.update(expert_parallel.ExpertParallel())
    assert isinstance(standard, moe_dispatch.NpuExpertParallel)


def test_tilelang_moe_reduce_config_is_independent_opt_in():
    assert moe_dispatch.TileLangMoeReduceModelConfig.model_converter is moe_dispatch.TileLangMoeReduceConverter
    assert issubclass(
        moe_dispatch.NpuDeepSeekV4MoETileLangReduce,
        moe_dispatch.NpuDeepSeekV4MoE,
    )


def test_tilelang_moe_reduce_path_is_selected_only_by_opt_in_module(monkeypatch):
    tilelang_calls = []
    cann_calls = []

    def fake_permute(data, indices):
        routed = data.repeat_interleave(2, dim=0) if data.shape[0] == 2 else data
        return routed, torch.arange(indices.numel(), dtype=torch.int32)

    def fake_tilelang(routed_output, mapping):
        tilelang_calls.append((routed_output.clone(), mapping.clone()))
        return torch.full((2, 3), 7.0)

    def fake_unpermute(routed_output, sorted_indices, _probs):
        cann_calls.append(sorted_indices.clone())
        return routed_output

    class FakeExperts(torch.nn.Module):
        def forward(self, routed_input, _counts, _scores):
            return routed_input

    class FakeRouter(torch.nn.Module):
        def forward(self, _tokens, _input_ids, _bias):
            return (
                torch.ones(2, 2),
                torch.tensor([[0, 1], [1, 0]]),
                torch.tensor([2, 2]),
            )

    def make_module(module_type):
        module = cast("Any", object.__new__(module_type))
        torch.nn.Module.__init__(module)
        module.reorderer = SimpleNamespace(top_k=2)
        module.experts = FakeExperts()
        module.router = FakeRouter()
        module.expert_bias = None
        module.shared_experts = None
        module.tokens_per_expert = torch.zeros(2, dtype=torch.int64)
        return module

    monkeypatch.setattr(moe_dispatch.torch_npu, "npu_moe_token_permute", fake_permute)
    monkeypatch.setattr(moe_dispatch.torch_npu, "npu_moe_token_unpermute", fake_unpermute)
    tokens = torch.arange(6, dtype=torch.float32).view(2, 3)
    input_ids = torch.zeros(2, dtype=torch.int64)

    tilelang_module = make_module(moe_dispatch.NpuDeepSeekV4MoETileLangReduce)
    tilelang_module.tilelang_reduce_op = fake_tilelang
    tilelang_output = tilelang_module(tokens, input_ids)
    assert torch.equal(tilelang_output, torch.full((2, 3), 7.0))
    assert len(tilelang_calls) == 1
    assert not cann_calls

    default_module = make_module(moe_dispatch.NpuDeepSeekV4MoE)
    default_output = default_module(tokens, input_ids)
    expected = tokens.repeat_interleave(2, dim=0).view(2, 2, 3).sum(dim=1)
    assert torch.equal(default_output, expected)
    assert len(tilelang_calls) == 1
    assert len(cann_calls) == 1


def test_tilelang_moe_reduce_converter_loads_custom_op_only_when_needed(monkeypatch):
    load_calls = []

    def fake_tilelang(routed_output, _mapping):
        return routed_output

    def fake_loader():
        load_calls.append(True)
        return fake_tilelang

    monkeypatch.setattr(moe_dispatch, "load_tilelang_moe_reduce_fused", fake_loader)
    converter = moe_dispatch.TileLangMoeReduceConverter(cast("Any", SimpleNamespace(name="test")))

    empty_model = torch.nn.Module()
    converter.convert(empty_model)
    assert not load_calls

    model = torch.nn.Module()
    tilelang_module = cast(
        "Any",
        object.__new__(moe_dispatch.NpuDeepSeekV4MoETileLangReduce),
    )
    torch.nn.Module.__init__(tilelang_module)
    model.add_module("moe", tilelang_module)

    converter.convert(model)
    assert tilelang_module.tilelang_reduce_op is fake_tilelang
    assert len(load_calls) == 1

    converter.convert(model)
    assert len(load_calls) == 1


def test_re_routing_backward_unpermutes_token_and_scale_grads(monkeypatch):
    token_order_indices = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    seen = {}

    def fake_re_routing(routed_tokens, expert_token_num_per_rank, *, per_token_scales, **kwargs):
        assert kwargs == {
            "expert_token_num_type": 1,
            "idx_type": 0,
        }
        seen.update(
            token_dtype=routed_tokens.dtype,
            count_dtype=expert_token_num_per_rank.dtype,
            scale_dtype=per_token_scales.dtype,
        )
        return (
            routed_tokens[token_order_indices],
            per_token_scales[token_order_indices],
            token_order_indices,
            expert_token_num_per_rank.sum(dim=0).to(torch.int32),
        )

    unpermute_indices = []

    def fake_unpermute(permuted_tokens, sorted_indices, _probs):
        unpermute_indices.append(sorted_indices.clone())
        return permuted_tokens[sorted_indices.to(torch.int64)]

    monkeypatch.setattr(permutation.torch_npu, "npu_moe_re_routing", fake_re_routing)
    monkeypatch.setattr(permutation.torch_npu, "npu_moe_token_unpermute", fake_unpermute)

    routed_tokens = torch.arange(12, dtype=torch.float32).view(4, 3).requires_grad_()
    counts = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    scales = torch.arange(4, dtype=torch.float32).view(4, 1).requires_grad_()
    token_grad = torch.arange(12, 24, dtype=torch.float32).view(4, 3)
    scale_grad = torch.arange(4, 8, dtype=torch.float32).view(4, 1)

    permuted_tokens, permuted_scales, restore_indices, num_tokens_per_expert = NPUMoeReRouting.apply(
        routed_tokens, counts, scales
    )
    loss = (permuted_tokens * token_grad).sum() + (permuted_scales * scale_grad).sum()
    loss.backward()

    assert seen == {
        "token_dtype": torch.float32,
        "count_dtype": torch.int64,
        "scale_dtype": torch.float32,
    }
    assert num_tokens_per_expert.dtype is torch.int64
    expected_restore_indices = torch.argsort(token_order_indices)
    assert torch.equal(restore_indices, expected_restore_indices)
    assert len(unpermute_indices) == 2
    assert torch.equal(unpermute_indices[0], expected_restore_indices)
    assert torch.equal(unpermute_indices[1], expected_restore_indices)
    assert routed_tokens.grad is not None
    assert scales.grad is not None
    assert torch.equal(routed_tokens.grad, token_grad[expected_restore_indices])
    assert torch.equal(scales.grad, scale_grad[expected_restore_indices])


def test_token_dispatch_real_process_group_routes_scores_through_re_routing(monkeypatch):
    group = object()
    device_mesh = cast("Any", _FakeDeviceMesh(group))
    num_tokens_per_expert_group = torch.tensor([2, 1, 4, 3], dtype=torch.int64)
    seen = {}

    monkeypatch.setattr(moe_dispatch, "is_fake_process_group", lambda _group: False)

    def fake_all_to_all_single(tensor, *_args, group):
        seen["count_group"] = group
        seen["count_input"] = tensor.clone()
        return num_tokens_per_expert_group

    def fake_all_to_all_single_autograd(tensor, output_splits, input_splits, received_group):
        assert received_group is group
        assert output_splits == [3, 7]
        assert input_splits == [3, 7]
        if tensor.shape[1] == 2:
            seen["tokens_before_rerouting"] = tensor.clone()
            return tensor + 100
        seen["scores_before_rerouting"] = tensor.clone()
        return tensor + 200

    def fake_re_routing_apply(routed_tokens, counts_2d, per_token_scales):
        seen["rerouting_tokens"] = routed_tokens.clone()
        seen["rerouting_counts"] = counts_2d.clone()
        seen["rerouting_scores"] = per_token_scales.clone()
        return (
            routed_tokens + 1,
            per_token_scales + 1,
            torch.arange(routed_tokens.shape[0]),
            counts_2d.sum(dim=0),
        )

    class FakeReRouting:
        apply = staticmethod(fake_re_routing_apply)

    class FakeUnpermute:
        @staticmethod
        def apply(routed_output, _indices, _shape):
            return routed_output

    def fake_retain(save_ops, function, *args, **kwargs):
        seen.setdefault("retained_ops", []).append(set(save_ops))
        return function(*args, **kwargs)

    monkeypatch.setattr(moe_dispatch, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(moe_dispatch, "all_to_all_single_autograd", fake_all_to_all_single_autograd)
    monkeypatch.setattr(moe_dispatch, "NPUMoeReRouting", FakeReRouting)
    monkeypatch.setattr(moe_dispatch, "NPUMoeTokenUnpermute", FakeUnpermute)
    monkeypatch.setattr(moe_dispatch, "retain_op_output", fake_retain)

    parallel_style = moe_dispatch.NpuExpertParallel()
    routed_input = torch.arange(20, dtype=torch.float32).view(10, 2)
    num_tokens_per_expert = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    routed_scores = torch.arange(10, dtype=torch.float32).view(10, 1)

    output, num_tokens_per_local_expert, output_scores = _call_token_dispatch(
        parallel_style,
        (routed_input, num_tokens_per_expert, routed_scores),
        device_mesh,
    )

    assert seen["count_group"] is group
    assert torch.equal(seen["count_input"], num_tokens_per_expert)
    assert torch.equal(seen["tokens_before_rerouting"], routed_input)
    assert torch.equal(seen["scores_before_rerouting"], routed_scores)
    assert torch.equal(seen["rerouting_tokens"], routed_input + 100)
    assert torch.equal(seen["rerouting_scores"], routed_scores + 200)
    assert torch.equal(seen["rerouting_counts"], num_tokens_per_expert_group.view(2, 2))
    assert torch.equal(output, routed_input + 101)
    assert torch.equal(output_scores, routed_scores + 201)
    combined = _call_token_combine(parallel_style, output, device_mesh)
    assert torch.equal(combined, output + 100)
    assert torch.equal(parallel_style.permuted_indices, torch.arange(10))
    assert torch.equal(num_tokens_per_local_expert, torch.tensor([6, 4], dtype=torch.int64))
    assert seen["retained_ops"] == [
        set(moe_dispatch.ALL_TO_ALL_SAC_SAVE_OPS),
        set(moe_dispatch.ALL_TO_ALL_SAC_SAVE_OPS),
        set(moe_dispatch.ALL_TO_ALL_SAC_SAVE_OPS),
        set(moe_dispatch.ALL_TO_ALL_SAC_SAVE_OPS),
    ]


def test_dsv4_forward_runs_routed_experts_before_shared_experts(monkeypatch):
    events = []
    x = torch.ones(1, 2, 3)
    input_ids = torch.ones(1, 2, dtype=torch.int64)

    class FakeRouter:
        @staticmethod
        def __call__(tokens, input_ids_flat, bias):
            events.append("router")
            assert tokens.shape == (2, 3)
            assert torch.equal(input_ids_flat, input_ids.flatten())
            assert bias is None
            return (
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.int64),
                torch.tensor([2.0]),
            )

    class FakeDeepSeekV4MoE(torch.nn.Module):
        forward = moe_dispatch.NpuDeepSeekV4MoE.forward

        def __init__(self):
            super().__init__()
            self.router = FakeRouter()
            self.expert_bias = None
            self.tokens_per_expert = torch.zeros(1)

    def fake_run_local_experts(_module, inputs):
        events.append("experts")
        assert inputs.total_tokens == 2
        return torch.full((2, 3), 2.0)

    def fake_shared_expert_output(_module, tokens):
        events.append("shared")
        return torch.ones_like(tokens)

    monkeypatch.setattr(moe_dispatch, "_run_local_experts", fake_run_local_experts)
    monkeypatch.setattr(moe_dispatch, "_shared_expert_output", fake_shared_expert_output)

    module = FakeDeepSeekV4MoE()
    output = module(x, input_ids)

    assert events == ["router", "experts", "shared"]
    assert torch.equal(output, torch.full_like(x, 3.0))
