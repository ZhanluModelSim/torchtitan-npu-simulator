# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch_npu

from torchtitan_npu.converters.kernels import gmm as gmm_module
from torchtitan_npu.models.deepseek_v4 import model as model_module
from torchtitan_npu.models.deepseek_v4 import parallelize
from torchtitan_npu.patches.torchao_npu import mxfp8_grouped_mm as mxfp8_gmm_module

_EXPERT_ACTIVATION_ATTR = "_expert_activation"
_EXPERT_ACTIVATION_FN_ATTR = "_expert_activation_fn"
_EXPERT_ACTIVATION_COMPILE_KEY_ATTR = "_expert_activation_compile_key"
_RUN_GROUPED_MM_ATTR = "_run_experts_grouped_mm"
_GROUPED_MM_ATTR = "_grouped_mm"
_DYNAMO_ATTR = "_dynamo"
_REQUIRES_COMPILE_GRAPH_BREAK_ATTR = "_requires_compile_graph_break"
_RETAINED_OUTPUT_PROJECTION_ATTR = "_retained_output_projection"
_RETAIN_OUTPUT_PROJECTION_ATTR = "_retain_output_projection"


@pytest.mark.parametrize(
    ("requires_graph_break", "pre_attention_fullgraph"),
    [(False, True), (True, False)],
)
def test_compile_respects_module_graph_break_requirement(
    monkeypatch,
    requires_graph_break,
    pre_attention_fullgraph,
):
    module = nn.Module()
    pre_attention = nn.Identity()
    projection = nn.Identity()
    inner_attention = nn.Identity()
    module.add_module("pre_attention", pre_attention)
    module.add_module("projection", projection)
    module.add_module("inner_attention", inner_attention)
    if requires_graph_break:
        setattr(pre_attention, _REQUIRES_COMPILE_GRAPH_BREAK_ATTR, True)

    compile_calls = []

    def fake_compile(child, *, backend, fullgraph):
        compile_calls.append((child, backend, fullgraph))
        return child

    monkeypatch.setattr(torch, "compile", fake_compile)

    compile_config = SimpleNamespace(backend="inductor_npu")
    parallelize._compile_children_except(
        module,
        {"inner_attention"},
        compile_config,
    )

    assert compile_calls == [
        (pre_attention, "inductor_npu", pre_attention_fullgraph),
        (projection, "inductor_npu", True),
    ]


def _new_npu_grouped_experts() -> gmm_module.NpuGroupedExperts:
    module = gmm_module.NpuGroupedExperts.__new__(gmm_module.NpuGroupedExperts)
    torch.nn.Module.__init__(module)
    setattr(module, _EXPERT_ACTIVATION_FN_ATTR, getattr(gmm_module, _EXPERT_ACTIVATION_ATTR))
    setattr(module, _EXPERT_ACTIVATION_COMPILE_KEY_ATTR, None)
    return module


def _run_grouped_mm(*args, **kwargs):
    return getattr(gmm_module, _RUN_GROUPED_MM_ATTR)(*args, **kwargs)


def _get_grouped_mm_op():
    return getattr(torch.ops.aten, _GROUPED_MM_ATTR).default


def _get_npu_grouped_mm_op():
    return torch.ops.npu.npu_grouped_matmul.default


def test_mxfp8_scopes_output_retention_to_npu_grouped_matmul(monkeypatch):
    events = []
    expected = torch.empty((4, 8), dtype=torch.bfloat16)

    def fake_quant(tensor, **_kwargs):
        events.append("quant")
        return tensor, torch.ones(1)

    def fake_grouped_matmul(*_args, **_kwargs):
        events.append("grouped_matmul")
        return [expected]

    def fake_retain(save_ops, function, *args, **kwargs):
        events.append(("retain", set(save_ops)))
        return function(*args, **kwargs)

    monkeypatch.setattr(mxfp8_gmm_module, "should_save_bwd_quant_for_mx", lambda: False)
    monkeypatch.setattr(mxfp8_gmm_module, "is_in_recomputation", lambda: False)
    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", fake_quant)
    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", fake_grouped_matmul)
    monkeypatch.setattr(mxfp8_gmm_module, "retain_op_output", fake_retain)

    result = mxfp8_gmm_module.NpuMXFP8GroupedMM.forward(
        SimpleNamespace(save_for_backward=lambda *_tensors: None),
        torch.empty((4, 8), dtype=torch.bfloat16),
        torch.empty((1, 8, 8), dtype=torch.bfloat16),
        torch.tensor([4], dtype=torch.int32),
    )

    assert result is expected
    assert events == [
        "quant",
        "quant",
        ("retain", {_get_npu_grouped_mm_op()}),
        "grouped_matmul",
    ]


def test_grouped_mm_rejects_offsets_that_exceed_int32_before_kernel_call():
    x = torch.empty((torch.iinfo(torch.int32).max + 1, 0), device="meta")
    num_tokens_per_expert = torch.empty((0,), dtype=torch.int64, device="meta")

    with pytest.raises(ValueError, match="int32 grouped_mm offsets"):
        _run_grouped_mm(
            None,
            torch.empty((0,), device="meta"),
            None,
            x,
            num_tokens_per_expert,
        )


@pytest.mark.parametrize(
    "mxfp8",
    [False, True],
)
def test_grouped_mm_calls_expert_activation_between_gmms(
    monkeypatch,
    mxfp8,
):
    grouped_mm_calls = []
    retained_calls = []
    activation_calls = []
    h13 = torch.empty((4, 8), dtype=torch.bfloat16)
    h2_input = torch.empty((4, 4), dtype=torch.bfloat16)
    expected = torch.empty((4, 8), dtype=torch.bfloat16)

    def fake_grouped_mm(x, weight, *, offs):
        grouped_mm_calls.append((x, weight, offs))
        return h13 if len(grouped_mm_calls) == 1 else expected

    def fake_activation(h, swiglu_limit, routed_scores):
        activation_calls.append((h, swiglu_limit, routed_scores))
        return h2_input

    def fake_retain(save_ops, function, *args, **kwargs):
        retained_calls.append(set(save_ops))
        return function(*args, **kwargs)

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm)
    monkeypatch.setattr(gmm_module, "retain_op_output", fake_retain)
    x = torch.empty((4, 8), dtype=torch.bfloat16)
    w13 = torch.empty((1, 8, 8), dtype=torch.bfloat16)
    w2 = torch.empty((1, 8, 4), dtype=torch.bfloat16)
    if mxfp8:

        class FunctionTensor(torch.Tensor):
            pass

        w13 = w13.as_subclass(FunctionTensor)
        w2 = w2.as_subclass(FunctionTensor)
        monkeypatch.setattr(gmm_module, "_get_mxfp8_weight_wrapper_type", lambda: FunctionTensor)
    else:
        monkeypatch.setattr(gmm_module, "_get_mxfp8_weight_wrapper_type", lambda: None)
    counts = torch.tensor([4], dtype=torch.int64)
    scores = torch.empty((4, 1), dtype=torch.float32)

    result = _run_grouped_mm(
        w13,
        w2,
        None,
        x,
        counts,
        swiglu_limit=3.0,
        routed_scores=scores,
        activation_fn=fake_activation,
    )

    assert result is expected
    assert len(grouped_mm_calls) == 2
    expected_retained_calls = [] if mxfp8 else [{_get_grouped_mm_op()}] * 2
    assert retained_calls == expected_retained_calls
    assert grouped_mm_calls[0][0] is x
    assert grouped_mm_calls[1][0] is h2_input
    assert activation_calls[0] == (h13, 3.0, scores)
    assert grouped_mm_calls[0][2].dtype == torch.int32


def test_retained_output_projection_scopes_sac_to_bmm(monkeypatch):
    retained_calls = []

    def fake_retain(save_ops, function, *args, **kwargs):
        retained_calls.append((save_ops, function))
        return function(*args, **kwargs)

    monkeypatch.setattr(model_module, "retain_op_output", fake_retain)
    output = torch.randn(2, 3, 2, 4)
    projection = nn.Linear(4, 10, bias=False)
    weight = projection.weight.view(2, 5, 4)

    actual = getattr(model_module, _RETAINED_OUTPUT_PROJECTION_ATTR)(output, projection, 5)
    expected = torch.einsum("...gd,grd->...gr", output, weight)

    torch.testing.assert_close(actual, expected)
    assert retained_calls == [(model_module.BMM_SAC_SAVE_OPS, torch.bmm)]


def test_compile_expert_activation_is_shared_and_idempotent(monkeypatch):
    compile_calls = []
    dynamic_calls = []
    compiled_calls = []

    def fake_compile(fn, *, backend, fullgraph, options):
        compile_calls.append((fn, backend, fullgraph, options))

        def compiled_fn(h, swiglu_limit=None, routed_scores=None):
            compiled_calls.append((h, swiglu_limit, routed_scores))
            return "compiled-result"

        return compiled_fn

    def fake_maybe_mark_dynamic(tensor, dim):
        dynamic_calls.append((tensor, dim))

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(getattr(torch, _DYNAMO_ATTR), "maybe_mark_dynamic", fake_maybe_mark_dynamic)

    model = torch.nn.Module()
    expert_a = _new_npu_grouped_experts()
    expert_b = _new_npu_grouped_experts()
    model.add_module("expert_a", expert_a)
    model.add_module("expert_b", expert_b)
    eager_activation = getattr(gmm_module, _EXPERT_ACTIVATION_ATTR)

    gmm_module.compile_expert_activation(model, backend="inductor_npu", dynamic_tokens=True)
    gmm_module.compile_expert_activation(model, backend="inductor_npu", dynamic_tokens=True)

    assert len(compile_calls) == 1
    fn, backend, fullgraph, options = compile_calls[0]
    assert fn is eager_activation
    assert backend == "inductor_npu"
    assert fullgraph is True
    assert options["custom_partitioner_fn"].uuid() == "npu_expert_activation_default_partition"
    assert getattr(expert_a, _EXPERT_ACTIVATION_FN_ATTR) is getattr(expert_b, _EXPERT_ACTIVATION_FN_ATTR)

    h = torch.empty((4, 8), device="meta")
    scores = torch.empty((4, 1), device="meta")
    assert getattr(expert_a, _EXPERT_ACTIVATION_FN_ATTR)(h, 3.0, scores) == "compiled-result"
    assert dynamic_calls[0][0] is h
    assert dynamic_calls[1][0] is scores
    assert dynamic_calls[0][1] == dynamic_calls[1][1] == 0
    assert compiled_calls[0][0] is h
    assert compiled_calls[0][1:] == (3.0, scores)


def _record_call(calls):
    def record(*args, **kwargs):
        calls.append((args, kwargs))

    return record


def _noop(*args, **kwargs):
    return None


def _fake_mesh(*args, **kwargs):
    return object()


def _parallelize_test_model(ac_mode, compile_enabled, tp_enabled=False):
    model = torch.nn.Module()
    post_attention = parallelize.PostAttention.__new__(parallelize.PostAttention)
    torch.nn.Module.__init__(post_attention)
    setattr(post_attention, _RETAIN_OUTPUT_PROJECTION_ATTR, False)
    model.add_module("post_attention", post_attention)
    model.model_args = SimpleNamespace(use_global_tnd=False, n_layers=0, compress_ratios=())
    parallel_dims = SimpleNamespace(
        seq_len_divisor=1,
        tp=2 if tp_enabled else 1,
        cp=1,
        ep=1,
        tp_enabled=tp_enabled,
        cp_enabled=False,
        ep_enabled=False,
        etp_enabled=False,
        fsdp_enabled=False,
        dp_replicate_enabled=False,
        pp_enabled=False,
        get_mesh=_fake_mesh,
        get_optional_mesh=_noop,
    )
    parallelize.parallelize_deepseek_v4(
        model,
        parallel_dims=parallel_dims,
        training=SimpleNamespace(seq_len=8),
        model_converters=SimpleNamespace(converters=[]),
        parallelism=SimpleNamespace(
            expert_parallel_comm_backend="alltoall",
            disable_loss_parallel=False,
            fsdp_preserve_parameter_patterns=[],
        ),
        compile_config=SimpleNamespace(
            enable=compile_enabled,
            components=("model",),
            backend="inductor_npu",
        ),
        ac_config=SimpleNamespace(mode=ac_mode),
        dump_folder="test-dump",
    )
    return model


@pytest.mark.parametrize(
    "case",
    [
        ("selective", True, True, False, True),
        ("full", True, True, False, False),
        ("selective", False, True, False, False),
        ("selective", True, False, False, False),
        ("selective", True, True, True, False),
    ],
    ids=(
        "selective-compiled",
        "full-compiled",
        "selective-eager",
        "selective-without-gmm",
        "selective-compiled-tp",
    ),
)
def test_parallelize_selective_ac_wires_scoped_save_ops(
    monkeypatch,
    case,
):
    ac_mode, compile_enabled, gmm_enabled, tp_enabled, use_scoped_ac = case
    save_ops_calls = []
    scoped_ac_calls = []
    native_ac_calls = []
    bridge_compile_calls = []
    model_compile_calls = []

    def record_save_ops(save_ops):
        save_ops_calls.append(set(save_ops))
        return nullcontext()

    def has_converter(_converters, name):
        return gmm_enabled if name == "npu_gmm" else name == "npu_moe_dispatch"

    def record_scoped_ac(model, ac_config, save_ops):
        scoped_ac_calls.append(set(save_ops))

    monkeypatch.setattr(parallelize, "apply_distributed_indexer_loss_tracking", _noop)
    monkeypatch.setattr(parallelize, "find_float8_linear_config", lambda _converters: None)
    monkeypatch.setattr(parallelize, "has_npu_converter", has_converter)
    monkeypatch.setattr(parallelize, "apply_non_moe_tp", _noop)
    monkeypatch.setattr(parallelize, "maybe_enable_async_tp", _noop)
    monkeypatch.setattr(parallelize, "apply_moe_ep_tp", _noop)
    monkeypatch.setattr(parallelize, "extend_selective_ac_save_ops", record_save_ops)
    monkeypatch.setattr(parallelize, "apply_scoped_selective_ac", record_scoped_ac)
    monkeypatch.setattr(parallelize, "apply_ac", _record_call(native_ac_calls))
    monkeypatch.setattr(gmm_module, "compile_expert_activation", _record_call(bridge_compile_calls))
    monkeypatch.setattr(parallelize, "apply_compile", _record_call(model_compile_calls))

    model = _parallelize_test_model(ac_mode, compile_enabled, tp_enabled=tp_enabled)

    target_ops = (
        parallelize.BMM_SAC_SAVE_OPS
        | parallelize.ALL_TO_ALL_SAC_SAVE_OPS
        | parallelize.NPU_GMM_SAC_SAVE_OPS
    )
    expected_scoped_ops = (
        set(target_ops) if use_scoped_ac else set()
    )
    assert scoped_ac_calls == ([expected_scoped_ops] if use_scoped_ac else [])
    assert getattr(model.post_attention, _RETAIN_OUTPUT_PROJECTION_ATTR) is use_scoped_ac
    assert save_ops_calls == ([] if use_scoped_ac else [set()])
    assert len(native_ac_calls) == int(not use_scoped_ac)
    assert len(bridge_compile_calls) == int(compile_enabled and gmm_enabled)
    assert len(model_compile_calls) == int(compile_enabled)
    if compile_enabled:
        assert model_compile_calls[0][1] == {}
