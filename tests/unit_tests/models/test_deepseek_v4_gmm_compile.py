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

from torchtitan_npu.converters.kernels import gmm as gmm_module
from torchtitan_npu.models.deepseek_v4 import parallelize

_EXPERT_ACTIVATION_ATTR = "_expert_activation"
_EXPERT_ACTIVATION_FN_ATTR = "_expert_activation_fn"
_EXPERT_ACTIVATION_COMPILE_KEY_ATTR = "_expert_activation_compile_key"
_RUN_GROUPED_MM_ATTR = "_run_experts_grouped_mm"
_GROUPED_MM_ATTR = "_grouped_mm"
_DYNAMO_ATTR = "_dynamo"


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


def test_grouped_mm_calls_expert_activation_between_gmms(monkeypatch):
    grouped_mm_calls = []
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

    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm)
    x = torch.empty((4, 8), dtype=torch.bfloat16)
    w13 = torch.empty((1, 8, 8), dtype=torch.bfloat16)
    w2 = torch.empty((1, 8, 4), dtype=torch.bfloat16)
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
    assert grouped_mm_calls[0][0] is x
    assert grouped_mm_calls[1][0] is h2_input
    assert activation_calls[0] == (h13, 3.0, scores)
    assert grouped_mm_calls[0][2].dtype == torch.int32


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


def _parallelize_test_model(ac_mode, compile_enabled):
    model = torch.nn.Module()
    model.model_args = SimpleNamespace(use_global_tnd=False, n_layers=0, compress_ratios=())
    parallel_dims = SimpleNamespace(
        seq_len_divisor=1,
        tp=1,
        cp=1,
        ep=1,
        tp_enabled=False,
        cp_enabled=False,
        ep_enabled=False,
        etp_enabled=False,
        fsdp_enabled=False,
        dp_replicate_enabled=False,
        pp_enabled=False,
    )
    parallelize.parallelize_deepseek_v4(
        model,
        parallel_dims=parallel_dims,
        training=SimpleNamespace(seq_len=8),
        model_converters=SimpleNamespace(converters=[]),
        parallelism=SimpleNamespace(expert_parallel_comm_backend="alltoall"),
        compile_config=SimpleNamespace(
            enable=compile_enabled,
            components=("model",),
            backend="inductor_npu",
        ),
        ac_config=SimpleNamespace(mode=ac_mode),
        dump_folder="test-dump",
    )


@pytest.mark.parametrize(
    ("ac_mode", "compile_enabled", "gmm_enabled", "save_grouped_mm"),
    [
        ("selective", True, True, True),
        ("full", True, True, False),
        ("selective", False, True, False),
        ("selective", True, False, False),
    ],
)
def test_parallelize_selective_ac_wires_grouped_mm_save(
    monkeypatch,
    ac_mode,
    compile_enabled,
    gmm_enabled,
    save_grouped_mm,
):
    save_ops_calls = []
    bridge_compile_calls = []
    model_compile_calls = []

    def record_save_ops(save_ops):
        save_ops_calls.append(set(save_ops))
        return nullcontext()

    def has_converter(*args):
        return gmm_enabled

    monkeypatch.setattr(parallelize, "apply_distributed_indexer_loss_tracking", _noop)
    monkeypatch.setattr(parallelize, "has_npu_converter", has_converter)
    monkeypatch.setattr(parallelize, "extend_selective_ac_save_ops", record_save_ops)
    monkeypatch.setattr(parallelize, "apply_ac", _noop)
    monkeypatch.setattr(gmm_module, "compile_expert_activation", _record_call(bridge_compile_calls))
    monkeypatch.setattr(parallelize, "apply_compile", _record_call(model_compile_calls))

    _parallelize_test_model(ac_mode, compile_enabled)

    expected_save_ops = {_get_grouped_mm_op()} if save_grouped_mm else set()
    assert save_ops_calls == [expected_save_ops]
    assert len(bridge_compile_calls) == int(compile_enabled and gmm_enabled)
    assert len(model_compile_calls) == int(compile_enabled)
