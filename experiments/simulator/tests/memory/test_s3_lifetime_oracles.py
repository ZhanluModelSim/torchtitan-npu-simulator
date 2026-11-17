# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent byte/causal oracles, specified before estimator adaptation."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import create_selective_checkpoint_contexts
from torchtitan_simulator.capture.checkpoint_execution import install_checkpoint_execution_tracking
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.module_path import ModulePathTracker
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.memory.estimator import estimate_static_memory
from torchtitan_simulator.memory.records import (
    AutogradSavedTensorEvent,
    FSDPResidencyEvent,
    RawMemoryEvent,
    TensorRef,
)


@pytest.fixture(autouse=True)
def cpu_checkpoint_device_default(monkeypatch):
    from torch.utils.checkpoint import DefaultDeviceType

    # CPU oracles must not inherit torch_npu's process-global default device.
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")


def ref(tensor_id, size, key, *, alias_of=None):
    return TensorRef(
        tensor_id,
        str(tensor_id),
        (size // 4,),
        "float32",
        "meta",
        size,
        requires_grad=True,
        storage_key=key,
        storage_bytes=size,
        alias_of=alias_of,
    )


def event(seq, op_id, name, inputs=(), outputs=(), phase="forward"):
    return RawMemoryEvent(
        seq,
        op_id,
        seq,
        name,
        name,
        phase,
        "",
        tuple(inputs),
        tuple(outputs),
        "backward" if phase == "backward" else "original_forward",
    )


@pytest.mark.parametrize("staging_view", [False, True])
@pytest.mark.parametrize("deferred_staging", [False, True])
def test_fsdp_shard_is_persistent_but_full_storage_dies_between_uses(staging_view, deferred_staging):
    model = torch.nn.Linear(4, 1, bias=False, device="meta")
    model.weight.grad = torch.empty_like(model.weight)
    shard = ref(id(model.weight), 16, str(model.weight.untyped_storage()._cdata))
    stage_f = ref(200, 64, "stage_f", alias_of=199 if staging_view else None)
    stage_b = ref(201, 64, "stage_b")
    full = ref(300, 64, "full")
    activation = ref(400, 16, "activation")
    gradient = ref(500, 16, "gradient")
    events = [
        event(1, 10, "comm.allgather", [shard], [stage_f]),
        event(4, 11, "aten.mm.default", [full], [activation]),
        event(8, 12, "test.gap", [activation]),
        event(10, 20, "comm.allgather", [shard], [stage_b], phase="backward"),
        event(13, 21, "aten.mm.default", [full, activation], [gradient], phase="backward"),
        event(18, 30, "optimizer.step", [shard, gradient], phase="optimizer"),
    ]
    if staging_view:
        events.insert(0, event(0, 9, "aten.empty.memory_format", outputs=[ref(199, 64, "stage_f")]))
    markers = [
        FSDPResidencyEvent(
            "layer", action, seq, phase, 64, (full.tensor_id,), transition_id=transition, storage_keys=("full",)
        )
        for action, seq, phase, transition in [
            ("alloc", 3, "forward", "f"),
            ("free", 6, "forward", "f"),
            ("alloc", 11, "backward", "b"),
            ("free", 16, "backward", "b"),
        ]
    ]
    if deferred_staging:
        markers.extend(
            [
                FSDPResidencyEvent(
                    "layer", "staging_free", 9, "forward", 64, (200,), transition_id="f", storage_keys=("stage_f",)
                ),
                FSDPResidencyEvent(
                    "layer", "staging_free", 11, "backward", 64, (201,), transition_id="b", storage_keys=("stage_b",)
                ),
            ]
        )
    comms = [
        SimpleNamespace(
            op_id=op_id, comm_primitive="allgather", comm_layer="L2", comm_dim="fsdp", fsdp_transition_id=transition
        )
        for op_id, transition in [(10, "f"), (20, "b")]
    ]
    saved_weight = AutogradSavedTensorEvent(
        1, 300, "full", 64, 64, (16,), "float32", 4, unpack_seq=13, phase="forward", execution_kind="original_forward"
    )
    plan = estimate_static_memory(
        events,
        model_parts=[model],
        comm_events=comms,
        fsdp_residency_events=markers,
        autograd_saved_tensor_events=[saved_weight],
    )

    def active_after(seq):
        return sum(item.resident_num_bytes for item in plan.tensor_lifetimes if item.birth_seq <= seq < item.death_seq)

    # Four float32 values remain sharded throughout. The saved Python weight
    # view does not keep a freed FSDP allocation alive between F and B.
    assert plan.persistent_param_bytes == 16
    assert active_after(8) == 16 + 16 + (64 if deferred_staging else 0)
    # An AG output view shares the preallocated staging storage; it remains
    # live until copy-out, rather than disappearing at the launch call.
    assert active_after(2) == 16 + 64
    assert [(item.birth_seq, item.death_seq) for item in plan.tensor_lifetimes if item.kind == "fsdp_full_param"] == [
        (3, 6),
        (11, 16),
    ]
    # At backward copy-out both staging and full parameters coexist, together
    # with the saved activation and shard: 64 + 64 + 16 + 16 bytes.
    assert plan.peak_active_bytes == 160
    if deferred_staging:
        stages = [item for item in plan.tensor_lifetimes if item.reason == "fsdp_allgather_observed_release"]
        assert [(item.birth_seq, item.death_seq) for item in stages] == [(0 if staging_view else 1, 9), (10, 11)]


def test_full_ac_output_saved_by_external_loss_survives_until_actual_backward_use():
    class Sin(torch.nn.Module):
        def forward(self, x):
            return x.sin()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = checkpoint_wrapper(Sin(), early_stop=False, preserve_rng_state=False)

        def forward(self, x):
            return self.block(x)

    model = Model()
    install_checkpoint_execution_tracking([model])
    x = torch.linspace(0.1, 0.4, 4, dtype=torch.float64, requires_grad=True)
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0], record_memory=True)
    with ModulePathTracker(model) as tracker:
        capture.module_path_tracker = tracker
        with capture, AutogradSavedTensorCapture():
            y = model(x)
            loss = y.square().sum()
            phase[0] = "backward"
            loss.backward()
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(),
        checkpoint_boundary_events=capture.checkpoint_boundary_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
    )
    original_sin = next(
        raw
        for raw in plan.raw_events
        if raw.raw_op_type == "aten.sin.default" and raw.execution_kind == "original_forward"
    )
    storage_key = original_sin.outputs[0].storage_key
    lifetime = next(
        item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{original_sin.outputs[0].tensor_id}"
    )
    actual_uses = [
        raw.seq_idx
        for raw in plan.raw_events
        if raw.phase == "backward" and any(value.storage_key == storage_key for value in raw.inputs)
    ]
    assert actual_uses
    assert lifetime.death_seq >= max(actual_uses)
    reference = x.detach().clone().requires_grad_()
    expected = reference.sin().square().sum()
    expected.backward()
    assert torch.equal(loss, expected)
    assert torch.equal(x.grad, reference.grad)


def test_selective_cache_retains_even_an_output_with_no_dispatch_consumer():
    contexts = []

    def context_fn():
        pair = create_selective_checkpoint_contexts([torch.ops.aten.sin.default])
        contexts.append(pair)
        return pair

    class Block(torch.nn.Module):
        def forward(self, x):
            x.sin()  # Unused by compute, but MUST_SAVE keeps its actual storage.
            return x.cos()

    model = torch.nn.Sequential(
        checkpoint_wrapper(
            Block(),
            context_fn=context_fn,
            early_stop=False,
            preserve_rng_state=False,
        )
    )
    install_checkpoint_execution_tracking([model])
    x = torch.linspace(0.1, 0.4, 4, dtype=torch.float64, requires_grad=True)
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0], record_memory=True)
    with ModulePathTracker(model) as tracker:
        capture.module_path_tracker = tracker
        with capture, AutogradSavedTensorCapture():
            loss = model(x).sum()
            # Inspect the native cache, not the requested policy: it really
            # retains one 32-byte sin output across this forward/backward gap.
            native_cache = contexts[0][0].storage
            entries = native_cache[torch.ops.aten.sin.default]
            assert len(entries) == 1
            assert entries[0].val.numel() * entries[0].val.element_size() == 32
            capture.record_synthetic_op("test.forward_backward_gap", [], [])
            gap_seq = capture.memory_events()[-1].seq_idx
            phase[0] = "backward"
            loss.backward()
            capture.record_synthetic_op("test.after_backward", [], [])
            after_backward_seq = capture.memory_events()[-1].seq_idx
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(),
        checkpoint_boundary_events=capture.checkpoint_boundary_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
    )
    sin_event = next(raw for raw in plan.raw_events if raw.raw_op_type == "aten.sin.default")
    lifetime = next(
        item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{sin_event.outputs[0].tensor_id}"
    )
    assert lifetime.num_bytes == 32
    assert lifetime.death_seq > gap_seq
    assert lifetime.death_seq < after_backward_seq
    reference = x.detach().clone().requires_grad_()
    expected = reference.cos().sum()
    expected.backward()
    assert torch.equal(loss, expected)
    assert torch.equal(x.grad, reference.grad)


def test_autograd_save_is_retained_until_node_release_after_its_last_tensor_read():
    class DelayedRelease(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x.sin())
            return x.cos()

        @staticmethod
        def backward(ctx, grad):
            (saved,) = ctx.saved_tensors
            result = -grad * saved
            # Autograd still owns its saved slot while this later allocation
            # executes, even though the final tensor read happened above.
            torch.empty(64, dtype=grad.dtype)
            return result

    x = torch.linspace(0.1, 0.4, 4, dtype=torch.float64, requires_grad=True)
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0], record_memory=True)
    with capture, AutogradSavedTensorCapture():
        loss = DelayedRelease.apply(x).sum()
        phase[0] = "backward"
        loss.backward()
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(), autograd_saved_tensor_events=capture.autograd_saved_tensor_events()
    )
    sin = next(event for event in plan.raw_events if event.raw_op_type == "aten.sin.default")
    workspace = next(
        event
        for event in plan.raw_events
        if event.raw_op_type == "aten.empty.memory_format" and event.outputs[0].shape == (64,)
    )
    saved = next(item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{sin.outputs[0].tensor_id}")
    assert saved.death_seq >= workspace.seq_idx
    assert torch.equal(x.grad, -x.detach().sin())


def test_python_local_storage_stays_live_across_later_allocation():
    import weakref

    witnesses = []

    def forward(x):
        retained = x.sin()
        witness = weakref.ref(retained)
        output = retained.cos()
        torch.empty(64, dtype=x.dtype)
        witnesses.append(witness() is not None)
        return output

    x = torch.linspace(0.1, 0.4, 4, dtype=torch.float64)
    capture = OpDispatchCapture(record_memory=True)
    with capture:
        output = forward(x)
    plan = estimate_static_memory(capture.memory_events(), storage_lifetimes=capture.storage_lifetime_events())
    sin = next(event for event in plan.raw_events if event.raw_op_type == "aten.sin.default")
    workspace = next(
        event
        for event in plan.raw_events
        if event.raw_op_type == "aten.empty.memory_format" and event.outputs[0].shape == (64,)
    )
    lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{sin.outputs[0].tensor_id}")
    assert witnesses == [True]
    assert torch.equal(output, x.sin().cos())
    assert lifetime.death_seq >= workspace.seq_idx
    # x, retained sin, returned cos, and the 64-double temporary coexist.
    assert plan.peak_active_bytes == 32 + 32 + 32 + 512


def test_device_memory_excludes_host_metadata_but_preserves_raw_dependencies():
    from dataclasses import replace

    device_input = ref(1, 16, "input")
    host_metadata = replace(ref(2, 128, "metadata"), device="cpu")
    device_output = ref(3, 32, "output")
    raw = event(0, 10, "test.metadata_consumer", [device_input, host_metadata], [device_output])
    plan = estimate_static_memory([raw], counted_device_type="meta")
    assert plan.peak_active_bytes == 16 + 32
    assert plan.raw_events[0].inputs == (device_input, host_metadata)
    assert not any(item.tensor_id == "external:2" for item in plan.tensor_lifetimes)


def test_registered_buffers_are_resident_before_first_use_and_not_parameter_dtype_overridden():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(4, device="meta"))
    model.weight.grad = torch.empty_like(model.weight)
    model.register_buffer("constant", torch.empty(4, device="meta"))
    model.register_buffer("parameter_alias", model.weight.detach())
    plan = estimate_static_memory([], model_parts=[model], parameter_storage_dtype="bfloat16")
    assert plan.persistent_param_bytes == 8
    # The independent 16-byte buffer is resident even before any use; its
    # dtype is not a parameter-shard dtype assumption. The alias adds zero.
    assert plan.peak_active_bytes == 8 + 16
    assert plan.persistent_buffer_bytes == 16


def test_preallocated_inputs_are_resident_before_their_first_dispatch_use():
    from dataclasses import dataclass

    @dataclass
    class Metadata:
        value: torch.Tensor

    early = torch.empty(4, dtype=torch.float64)
    late = torch.empty(64, dtype=torch.float64)
    capture = OpDispatchCapture(record_memory=True)
    capture.record_initial_memory_tensors((early, {"metadata": Metadata(late), "alias": [late[:1], late]}))
    with capture:
        torch.empty(128, dtype=torch.float64)
        early.cos()
        late.cos()
    plan = estimate_static_memory(
        capture.memory_events(),
        storage_lifetimes=capture.storage_lifetime_events(),
        initial_tensors=capture.initial_memory_tensors(),
    )
    assert plan.peak_active_bytes == 32 + 512 + 1024


@pytest.fixture
def native_meta_group(tmp_path):
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=2)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("primitive", ["all_to_all", "reduce_scatter"])
def test_native_selective_collective_policy_avoids_forward_recommunication(primitive, native_meta_group):
    from torch.distributed import _functional_collectives as funcol
    from torchtitan_simulator.capture.comm_events import observe_collectives

    class Block(torch.nn.Module):
        def forward(self, x):
            value = x.sin()
            if primitive == "all_to_all":
                value = funcol.all_to_all_single_autograd(value, [2, 2], [2, 2], native_meta_group)
            else:
                value = funcol.reduce_scatter_tensor_autograd(value, "sum", 0, native_meta_group)
            return value.cos()

    model = torch.nn.Sequential(
        checkpoint_wrapper(
            Block(),
            early_stop=False,
            preserve_rng_state=False,
            context_fn=lambda: create_selective_checkpoint_contexts(
                [
                    torch.ops._c10d_functional.all_to_all_single.default,
                    torch.ops._c10d_functional.reduce_scatter_tensor.default,
                ]
            ),
        )
    )
    install_checkpoint_execution_tracking([model])
    x = torch.empty(4, 2, device="meta", requires_grad=True)
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0], record_memory=True)
    with ModulePathTracker(model) as tracker:
        capture.module_path_tracker = tracker
        with observe_collectives() as recorder, capture, AutogradSavedTensorCapture():
            loss = model(x).sum()
            phase[0] = "backward"
            loss.backward()
    nodes = capture.build_nodes()
    collective_kinds = [
        nodes[event.op_id].annotations["execution_kind"]
        for event in recorder.events
        if event.comm_primitive == primitive
    ]
    assert collective_kinds.count("original_forward") == 1
    assert collective_kinds.count("backward") == (1 if primitive == "all_to_all" else 0)
    if primitive == "reduce_scatter":
        assert sum(event.comm_primitive == "allgather" for event in recorder.events) == 1
    assert collective_kinds.count("recompute") == 0
    assert x.grad is not None and x.grad.shape == x.shape


def test_quant_mm_selection_caches_the_registered_native_meta_output():
    import torch_npu  # noqa: F401
    from torchtitan_simulator.runtime import meta_runtime
    from torchtitan_simulator.selective_ac import SimulatorSelectiveAC

    class Block(torch.nn.Module):
        def forward(self, x):
            codes = torch.empty(4, 4, dtype=torch.int8, device="meta")
            scale = torch.ones(4, dtype=torch.float32, device="meta")
            quantized = torch.ops.npu.npu_quant_matmul.default(
                codes,
                codes,
                scale,
                output_dtype=torch.bfloat16,
            )
            return (x.sin() + quantized).cos()

    with meta_runtime():
        policy = SimulatorSelectiveAC.Config(save_ops=["quant-mm"]).build()
        model = torch.nn.Sequential(policy._wrap_block(Block(), base_fqn="block"))
        install_checkpoint_execution_tracking([model])
        phase = ["forward"]
        capture = OpDispatchCapture(phase_provider=lambda: phase[0], record_memory=True)
        x = torch.empty(4, 4, device="meta", requires_grad=True)
        with ModulePathTracker(model) as tracker:
            capture.module_path_tracker = tracker
            with capture, AutogradSavedTensorCapture():
                loss = model(x).sum()
                phase[0] = "backward"
                loss.backward()
        events = [event for event in capture.memory_events() if event.raw_op_type == "npu.npu_quant_matmul.default"]
        assert len(events) == 1 and events[0].execution_kind == "original_forward"
        cached = [slot for slot in capture.autograd_saved_tensor_events() if slot.source == "selective_cache"]
        assert len(cached) == 1
        assert cached[0].storage_key == events[0].outputs[0].storage_key
        assert cached[0].release_seq >= cached[0].unpack_seq >= cached[0].pack_seq
        assert x.grad is not None and x.grad.shape == x.shape
