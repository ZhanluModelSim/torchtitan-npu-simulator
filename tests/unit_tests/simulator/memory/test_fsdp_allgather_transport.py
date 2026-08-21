# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.fsdp_allgather_transport import (
    apply_fsdp_allgather_transport_dtype,
)
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent, RawMemoryEvent, TensorRef


def _ref(tensor_id: int, num_bytes: int) -> TensorRef:
    return TensorRef(tensor_id, f"t{tensor_id}", (num_bytes // 4,), "float32", "meta", num_bytes)


def _event(seq_idx: int, op_id: int, inputs: tuple[TensorRef, ...], outputs: tuple[TensorRef, ...]) -> RawMemoryEvent:
    return RawMemoryEvent(seq_idx, op_id, seq_idx, "comm.allgather", "allgather", "forward", "", inputs, outputs, "original_forward")


def test_transport_override_changes_only_fsdp_allgather_traffic():
    node = OpNode(10, "allgather", [], [], {}, [], [], comm_bytes=64, peak_mem=128)
    fsdp = SimpleNamespace(
        op_id=10,
        comm_primitive="allgather",
        comm_dim="fsdp",
        tensor_shape=(64,),
        dtype="bfloat16",
        volume_bytes=128,
    )
    tp = SimpleNamespace(
        op_id=11,
        comm_primitive="allgather",
        comm_dim="tp",
        tensor_shape=(64,),
        dtype="bfloat16",
        volume_bytes=128,
    )

    assert apply_fsdp_allgather_transport_dtype([fsdp, tp], {10: node}, "float8_e4m3fn") == 1
    assert (fsdp.dtype, fsdp.volume_bytes, node.comm_bytes, node.peak_mem) == ("float8_e4m3fn", 64, 64, 128)
    assert (tp.dtype, tp.volume_bytes) == ("bfloat16", 128)


def test_transport_override_scales_staging_but_not_full_parameter_residency():
    shard, staging, full = _ref(1, 32), _ref(2, 128), _ref(3, 128)
    comm = SimpleNamespace(op_id=10, comm_primitive="allgather", comm_dim="fsdp", comm_layer="L2")
    plan = estimate_static_memory(
        [_event(0, 10, (shard,), (staging,))],
        comm_events=[comm],
        fsdp_residency_events=[
            FSDPResidencyEvent("layer0", "alloc", 2, "forward", 128, (full.tensor_id,)),
            FSDPResidencyEvent("layer0", "free", 5, "forward", 128, (full.tensor_id,)),
        ],
        fsdp_allgather_transport_dtype="float8_e4m3fn",
    )

    staging_lifetime = next(item for item in plan.tensor_lifetimes if item.tensor_id == "tensor:2")
    full_lifetime = next(item for item in plan.tensor_lifetimes if item.kind == "fsdp_full_param")
    assert (staging_lifetime.num_bytes, staging_lifetime.resident_num_bytes) == (128, 32)
    assert full_lifetime.resident_num_bytes == 128
