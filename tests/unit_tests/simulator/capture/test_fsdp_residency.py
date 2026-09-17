# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace

from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo

from torchtitan_npu.simulator.capture.communication_ownership import (
    _fsdp_prefetch_anchor,
    _FSDPGroupRegion,
)
from torchtitan_npu.simulator.capture.fsdp_residency import (
    _uses_fsdp_sharding,
)
from torchtitan_npu.simulator.ir.op_node import OpNode


def test_fsdp_residency_distinguishes_sharded_and_replicated_param_groups():
    fsdp_mesh_info = object.__new__(FSDPMeshInfo)

    assert _uses_fsdp_sharding(SimpleNamespace(mesh_info=fsdp_mesh_info))
    assert not _uses_fsdp_sharding(SimpleNamespace(mesh_info=object()))


def test_fsdp_prefetch_can_be_anchored_to_replicated_source_compute():
    predecessor = OpNode(
        op_id=90,
        op_type="input",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[],
        successors=[100],
        seq_idx=5,
        annotations={"raw_op_type": "aten.clone.default"},
    )
    source_entry = OpNode(
        op_id=100,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[90],
        successors=[101],
        seq_idx=20,
        annotations={
            "raw_op_type": "aten.mm.default",
            "module_path": "layers.0.attention",
        },
    )
    source_exit = OpNode(
        op_id=101,
        op_type="matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=[100],
        successors=[],
        seq_idx=30,
        annotations={
            "raw_op_type": "aten.mm.default",
            "module_path": "layers.0.moe",
        },
    )
    target_region = _FSDPGroupRegion(
        group_id="group1",
        module_fqn="layers.1",
        wait_seq_idx=40,
        release_seq_idx=60,
        entry_op_ids=(200,),
        exit_op_ids=(201,),
        external_predecessors=(),
    )

    anchor = _fsdp_prefetch_anchor(
        template_id="s0_F",
        target_group_id="group1",
        target_module_fqn="layers.1",
        target_region=target_region,
        target_collective_seq_idx=10,
        prefetch_source_fqn="layers.0",
        regions_by_module={},
        wait_seq_idxs_by_module={},
        invocation_starts_by_module={},
        comm_id_by_region={},
        nodes_by_id={
            predecessor.op_id: predecessor,
            source_entry.op_id: source_entry,
            source_exit.op_id: source_exit,
        },
    )

    assert anchor.predecessor_op_ids == (predecessor.op_id,)
    assert anchor.source_entry_op_ids == (source_entry.op_id,)
