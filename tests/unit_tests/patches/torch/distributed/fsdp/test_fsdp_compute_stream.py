# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native AG/RS multi-rank behavior and CPU contracts for RS allocation; not NPU allocator tests."""

import copy
import inspect
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

import torchtitan_npu.patches.torch.distributed.fsdp as fsdp_patch
from torchtitan_npu.patches.torch.distributed.fsdp import foreach_reduce as foreach_reduce_owner

_NON_CPU_DEVICE = torch.device("meta")


def _run_two_rank_compute_stream_worker(rank: int, rendezvous: str, enabled: bool) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    orig_ag = _fsdp_param_group.foreach_all_gather
    orig_reduce = foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE
    ag_calls = []
    rs_calls = []

    def spy_foreach_all_gather(
        fsdp_params,
        group,
        async_op,
        all_gather_copy_in_stream,
        all_gather_stream,
        device,
        all_gather_comm,
    ):
        current = torch.get_device_module("cpu").current_stream()
        ag_calls.append((async_op, all_gather_copy_in_stream == current, all_gather_stream == current))
        return orig_ag(
            fsdp_params,
            group,
            async_op,
            all_gather_copy_in_stream,
            all_gather_stream,
            device,
            all_gather_comm,
        )

    def spy_foreach_reduce(*args, **kwargs):
        bound = foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE.bind(*args, **kwargs)
        rs_calls.append(isinstance(bound.arguments["reduce_scatter_comm"], fsdp_patch._ComputeStreamAllocator))
        return orig_reduce(*args, **kwargs)

    _fsdp_param_group.foreach_all_gather = spy_foreach_all_gather
    foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE = spy_foreach_reduce
    try:
        mesh = init_device_mesh("cpu", (2,))
        fsdp_patch._ALLOC_ON_COMPUTE_STREAM = enabled
        fsdp_patch.apply()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            reference = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.SiLU(), torch.nn.Linear(6, 4))
        model = copy.deepcopy(reference)
        for module in (model[0], model[2], model):
            fully_shard(module, mesh=mesh)
            assert all(group.unshard_async_op == enabled for group in module._get_fsdp_state()._fsdp_param_groups)
        model[0].set_modules_to_forward_prefetch([model[2]])
        model[2].set_modules_to_backward_prefetch([model[0]])

        full_x = torch.arange(16, dtype=torch.float32).reshape(4, 4).div(9)
        local_x = full_x[rank * 2 : (rank + 1) * 2].clone().requires_grad_()
        ref_x = full_x.clone().requires_grad_()
        expected = reference(ref_x)
        actual = model(local_x)
        torch.testing.assert_close(actual, expected[rank * 2 : (rank + 1) * 2], rtol=0, atol=0)

        actual.square().sum().backward()
        expected.square().sum().backward()

        assert ag_calls, "Expected world_size=2 FSDP to invoke foreach_all_gather"
        assert all(call == (enabled, enabled, enabled) for call in ag_calls)
        assert rs_calls, "Expected world_size=2 FSDP to invoke foreach_reduce"
        # CPU tensors have no device stream caching allocator, so RS must bypass wrapping on CPU.
        assert all(wrapped is False for wrapped in rs_calls)
        torch.testing.assert_close(local_x.grad, ref_x.grad[rank * 2 : (rank + 1) * 2], rtol=1e-6, atol=1e-6)
        for param, ref_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param.grad.full_tensor(), ref_param.grad / 2, rtol=1e-6, atol=1e-6)
    finally:
        foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE = orig_reduce
        fsdp_patch._ALLOC_ON_COMPUTE_STREAM = False
        fsdp_patch.apply()
        _fsdp_param_group.foreach_all_gather = orig_ag
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("enabled", [False, True], ids=["native-default", "compute-stream-ag-rs"])
def test_compute_stream_mode_preserves_forward_and_gradients_across_ranks(tmp_path, enabled):
    torch.multiprocessing.start_processes(
        _run_two_rank_compute_stream_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", enabled),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_apply_toggles_compute_stream_patches_and_reuse_env(monkeypatch):
    group_type = _fsdp_param_group.FSDPParamGroup
    monkeypatch.setattr(group_type, "__init__", group_type.__init__)
    monkeypatch.setattr(_fsdp_collectives, "foreach_reduce", _fsdp_collectives.foreach_reduce)
    monkeypatch.setattr(_fsdp_param_group, "foreach_reduce", _fsdp_param_group.foreach_reduce)
    monkeypatch.setattr(fsdp_patch.FSDPParam, "all_gather_inputs", fsdp_patch.FSDPParam.all_gather_inputs)
    monkeypatch.setattr(foreach_reduce_owner, "_ACTIVE_LAYERS", dict(foreach_reduce_owner._ACTIVE_LAYERS))

    monkeypatch.delenv("MULTI_STREAM_MEMORY_REUSE", raising=False)
    monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", False)
    fsdp_patch.apply()
    assert group_type.__init__ is fsdp_patch._ORIGINAL_PARAM_GROUP_INIT
    assert _fsdp_collectives.foreach_reduce is foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE
    assert _fsdp_param_group.foreach_reduce is foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE
    assert "MULTI_STREAM_MEMORY_REUSE" not in os.environ

    monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", True)
    fsdp_patch.apply()
    assert group_type.__init__ is fsdp_patch._patched_param_group_init
    assert _fsdp_collectives.foreach_reduce is _fsdp_param_group.foreach_reduce
    assert _fsdp_collectives.foreach_reduce is not foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE
    assert _fsdp_collectives.foreach_reduce.__wrapped__ is foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE
    assert os.environ["MULTI_STREAM_MEMORY_REUSE"] == "2"
    assert inspect.signature(_fsdp_collectives.foreach_reduce) == foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE

    monkeypatch.setenv("MULTI_STREAM_MEMORY_REUSE", "")
    fsdp_patch.apply()
    assert os.environ["MULTI_STREAM_MEMORY_REUSE"] == "2"

    monkeypatch.setenv("MULTI_STREAM_MEMORY_REUSE", "0")
    fsdp_patch.apply()
    assert os.environ["MULTI_STREAM_MEMORY_REUSE"] == "0"


@pytest.fixture
def streams(monkeypatch):
    events = []
    buffers = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait_stream", self.name, other.name))

    owner, reduce, all_reduce = Stream("owner"), Stream("reduce"), Stream("all_reduce")
    state = SimpleNamespace(current=owner)

    @contextmanager
    def use_stream(stream):
        previous, state.current = state.current, stream
        try:
            yield
        finally:
            state.current = previous

    handle = SimpleNamespace(current_stream=lambda: state.current, stream=use_stream)

    def allocate(size, *, dtype, device):
        assert size == (8,) and dtype == torch.float32 and device == _NON_CPU_DEVICE
        buf = SimpleNamespace(record_stream=Mock())
        buffers.append(buf)
        events.append(("allocate", state.current.name))
        return buf

    comm = Mock(allocate=Mock(side_effect=allocate))
    monkeypatch.setattr(torch, "get_device_module", lambda device: handle)
    return SimpleNamespace(
        owner=owner,
        reduce=reduce,
        all_reduce=all_reduce,
        state=state,
        handle=handle,
        buffers=buffers,
        comm=comm,
        events=events,
    )


@pytest.mark.parametrize(
    ("side_caller", "use_extra"),
    [(False, False), (True, False), (True, True)],
    ids=["same-stream", "single-side-stream", "extra-all-reduce-stream"],
)
def test_rs_allocator_orders_reuse_and_records_consumer(streams, side_caller, use_extra):
    s = streams
    extra = (s.all_reduce, s.owner, s.reduce, None) if use_extra else ()
    allocator = fsdp_patch._ComputeStreamAllocator(s.comm, s.handle, extra)
    caller = s.reduce if side_caller else s.owner
    with s.handle.stream(caller):
        buf = allocator.allocate((8,), dtype=torch.float32, device=_NON_CPU_DEVICE)
        assert buf is s.buffers[0]
        assert s.state.current is caller
        allocator(output_tensor=buf)
    assert s.events == [("allocate", "owner")] + ([("wait_stream", "reduce", "owner")] if side_caller else [])
    expected_recorded = ([s.reduce] + ([s.all_reduce] if use_extra else [])) if side_caller else []
    assert [call.args[0] for call in buf.record_stream.call_args_list] == expected_recorded
    s.comm.assert_called_once_with(output_tensor=buf)


def test_rs_disabled_preserves_native_arguments(monkeypatch):
    native = Mock(return_value=object())
    monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", False)
    allocator = Mock(side_effect=AssertionError("Disabled RS must not wrap the comm"))
    monkeypatch.setattr(fsdp_patch, "_ComputeStreamAllocator", allocator)
    wrapped = fsdp_patch._make_compute_stream_foreach_reduce(native)
    params, grads = [SimpleNamespace(reduce_dtype=torch.float32)], [torch.ones(8)]
    assert wrapped(params, grads, marker="native") is native.return_value
    native.assert_called_once_with(params, grads, marker="native")
    allocator.assert_not_called()


@pytest.mark.parametrize(
    ("size", "factor", "all_reduce", "all_reduce_grads", "has_hook", "expect_all_reduce_record"),
    [
        (None, 1.0, False, False, False, False),
        (1, 1.0, False, False, False, False),
        (2, 1.0, False, False, False, False),
        (1, 1.0, True, True, False, True),
        (2, 1.0, True, False, False, False),
        (2, 1.0, True, False, True, False),
        (2, 1.0, False, False, True, True),
        (1, 2.0, False, False, False, False),
        (1, None, False, False, False, False),
    ],
    ids=[
        "no-group-unit",
        "single-rank-unit",
        "multi-rank-unit",
        "hsdp-sync-unit",
        "hsdp-accum-only-unit",
        "hsdp-accum-with-hook-unit",
        "custom-hook-unit",
        "custom-factor",
        "none-factor",
    ],
)
def test_rs_wrapper_preserves_arguments_and_native_streams(
    monkeypatch,
    streams,
    size,
    factor,
    all_reduce,
    all_reduce_grads,
    has_hook,
    expect_all_reduce_record,
):
    s = streams
    monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", True)
    grads = [torch.ones(8, dtype=torch.bfloat16), torch.ones(8, dtype=torch.float32)]
    hook = (lambda _tensor: None) if has_hook else None
    prepare_inputs = Mock(name="prepare_reduce_scatter_inputs")
    arguments = dict(
        fsdp_params=[SimpleNamespace(reduce_dtype=torch.float32), SimpleNamespace(reduce_dtype=torch.float32)],
        unsharded_grads=grads,
        reduce_scatter_group=None if size is None else SimpleNamespace(size=lambda: size),
        reduce_scatter_stream=s.reduce,
        reduce_scatter_comm=s.comm,
        reduce_dtype=torch.float32,
        device=_NON_CPU_DEVICE,
        gradient_divide_factor=factor,
        all_reduce_group=object() if all_reduce else None,
        all_reduce_stream=s.all_reduce,
        all_reduce_grads=all_reduce_grads,
        partial_reduce_output=None,
        all_reduce_hook=hook,
        force_sum_reduction_for_comms=False,
        prepare_reduce_scatter_inputs=prepare_inputs,
    )

    expect_wrap = not (all_reduce and not all_reduce_grads)

    def native(*args, **kwargs):
        received = foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE.bind(*args, **kwargs).arguments
        for name, value in arguments.items():
            if name != "reduce_scatter_comm":
                assert received[name] is value
        if not expect_wrap:
            assert received["reduce_scatter_comm"] is s.comm
        else:
            assert isinstance(received["reduce_scatter_comm"], fsdp_patch._ComputeStreamAllocator)
        received["reduce_scatter_comm"].allocate((8,), dtype=torch.float32, device=_NON_CPU_DEVICE)
        with s.handle.stream(received["reduce_scatter_stream"]):
            return received["reduce_scatter_comm"].allocate((8,), dtype=torch.float32, device=_NON_CPU_DEVICE)

    wrapped = fsdp_patch._make_compute_stream_foreach_reduce(native)
    out = wrapped(**arguments)
    assert out is s.buffers[1]
    s.buffers[0].record_stream.assert_not_called()
    if expect_wrap:
        assert s.events == [("allocate", "owner"), ("allocate", "owner"), ("wait_stream", "reduce", "owner")]
        expected_streams = [s.reduce, s.all_reduce] if expect_all_reduce_record else [s.reduce]
        assert [call.args[0] for call in s.buffers[1].record_stream.call_args_list] == expected_streams
    else:
        assert s.events == [("allocate", "owner"), ("allocate", "reduce")]
        s.buffers[1].record_stream.assert_not_called()


@pytest.mark.parametrize("order", ["fsdp-first", "grad-accum-first"])
def test_foreach_reduce_sole_owner_preserves_grad_accum_across_reapply_and_order(monkeypatch, streams, order):
    from functools import wraps

    from torchtitan_npu.extensions.distributed import grad_accum

    s = streams
    group_type = _fsdp_param_group.FSDPParamGroup
    monkeypatch.setattr(group_type, "__init__", group_type.__init__)
    monkeypatch.setattr(group_type, "unshard", group_type.unshard)
    monkeypatch.setattr(group_type, "wait_for_unshard", group_type.wait_for_unshard)
    monkeypatch.setattr(_fsdp_collectives, "foreach_reduce", _fsdp_collectives.foreach_reduce)
    monkeypatch.setattr(_fsdp_param_group, "foreach_reduce", _fsdp_param_group.foreach_reduce)
    monkeypatch.setattr(fsdp_patch.FSDPParam, "all_gather_inputs", fsdp_patch.FSDPParam.all_gather_inputs)
    monkeypatch.setattr(foreach_reduce_owner, "_ACTIVE_LAYERS", {})

    observed_comms = []
    cpu_offload_calls = []

    @wraps(foreach_reduce_owner.ORIGINAL_FOREACH_REDUCE)
    def fake_native(*args, **kwargs):
        bound = foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE.bind(*args, **kwargs)
        observed_comms.append(bound.arguments["reduce_scatter_comm"])
        return "native"

    def fake_cpu_offload(**kwargs):
        cpu_offload_calls.append(kwargs["reduce_scatter_comm"])
        return "cpu-offload"

    monkeypatch.setattr(foreach_reduce_owner, "ORIGINAL_FOREACH_REDUCE", fake_native)
    monkeypatch.setattr(grad_accum, "_cpu_offload_foreach_reduce", fake_cpu_offload)

    if order == "fsdp-first":
        monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", True)
        fsdp_patch.apply()
        grad_accum.install()
    else:
        grad_accum.install()
        monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", True)
        fsdp_patch.apply()

    call_kwargs = dict(
        fsdp_params=[SimpleNamespace(reduce_dtype=torch.float32)],
        unsharded_grads=[torch.ones(8)],
        reduce_scatter_group=SimpleNamespace(size=lambda: 2),
        reduce_scatter_stream=s.reduce,
        reduce_scatter_comm=s.comm,
        reduce_dtype=torch.float32,
        device=_NON_CPU_DEVICE,
        gradient_divide_factor=1.0,
        all_reduce_group=None,
        all_reduce_stream=s.all_reduce,
        all_reduce_grads=False,
        partial_reduce_output=None,
        all_reduce_hook=None,
        force_sum_reduction_for_comms=False,
        prepare_reduce_scatter_inputs=None,
    )

    for compute_stream_enabled in (True, False, True):
        monkeypatch.setattr(fsdp_patch, "_ALLOC_ON_COMPUTE_STREAM", compute_stream_enabled)
        fsdp_patch.apply()
        installed = _fsdp_collectives.foreach_reduce
        assert installed is _fsdp_param_group.foreach_reduce
        assert getattr(installed, grad_accum._PATCH_MARKER, False)
        assert installed.__wrapped__ is not None
        assert inspect.signature(installed) == foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE

        monkeypatch.setattr(grad_accum, "_uses_cpu_canonical_grads", lambda _params, _dev: True)
        assert _fsdp_param_group.foreach_reduce(**call_kwargs) == "cpu-offload"

        monkeypatch.setattr(grad_accum, "_uses_cpu_canonical_grads", lambda _params, _dev: False)
        assert _fsdp_param_group.foreach_reduce(**call_kwargs) == "native"
        assert isinstance(observed_comms[-1], fsdp_patch._ComputeStreamAllocator) is compute_stream_enabled

    assert len(cpu_offload_calls) == 3

    with pytest.raises(ValueError, match="Unknown foreach_reduce layer"):
        foreach_reduce_owner.set_layer("unknown", None)
