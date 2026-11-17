# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 dense activation/optimizer offload: transfer nodes and derived budgets."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload


def _capture(**kwargs):
    workload = simulate(SimulationConfig(config="qwen3_32b_smoketest", seq_len=16, **kwargs))
    validation = validate_workload(workload)
    assert validation["graph_acyclic"] and validation["peak_storage_unique"]
    annotations = workload.iteration.schedule.annotations
    types = Counter(
        node.annotations["raw_op_type"]
        for template in workload.step_templates.values()
        for node in template.nodes.values()
    )
    return workload, annotations, types


def test_qwen3_offload_disabled_leaves_no_transfer_nodes():
    _, _, types = _capture(activation_checkpoint_mode="full")
    assert not [op_type for op_type in types if op_type.startswith("simulator.activation")]
    assert not [op_type for op_type in types if op_type.startswith("simulator.optimizer")]


def test_qwen3_activation_and_optimizer_offload_insert_transfer_nodes_and_reduce_peaks():
    base_workload, base_annotations, base_types = _capture(activation_checkpoint_mode="full")
    offload_workload, offload_annotations, offload_types = _capture(
        activation_checkpoint_mode="full",
        memory_offload_ac_saved_tensors=True,
        memory_offload_optimizer_states=True,
    )
    # One H2D/D2H pair per checkpointed layer plus one bidirectional window
    # covering the whole fused AdamW step; both switches stay independent.
    assert offload_types["simulator.activation_d2h"] == 4
    assert offload_types["simulator.activation_h2d"] == 4
    assert offload_types["simulator.optimizer_h2d"] == 1
    assert offload_types["simulator.optimizer_d2h"] == 1
    assert offload_types["simulator.stream_wait"] > base_types["simulator.stream_wait"]

    base_memory = base_annotations["memory_plan"]
    offload_memory = offload_annotations["memory_plan"]
    assert len(offload_memory.activation_buffer_lifetimes) > 0
    assert (
        offload_memory.steady_state_forward_peak_active_bytes
        < base_memory.steady_state_forward_peak_active_bytes
    )
    assert (
        offload_memory.steady_state_backward_peak_active_bytes
        < base_memory.steady_state_backward_peak_active_bytes
    )
    assert (
        offload_memory.steady_state_optimizer_peak_active_bytes
        < base_memory.steady_state_optimizer_peak_active_bytes
    )
    # Raw storage facts are shared, not rewritten by the offload projection.
    assert offload_memory.raw_events is not None and base_memory.raw_events is not None


def test_qwen3_optimizer_offload_alone_reduces_only_optimizer_peak():
    base_workload, base_annotations, base_types = _capture()
    offload_workload, offload_annotations, offload_types = _capture(
        memory_offload_optimizer_states=True,
    )
    assert offload_types["simulator.optimizer_h2d"] == 1
    assert offload_types["simulator.optimizer_d2h"] == 1
    assert not [op_type for op_type in offload_types if op_type.startswith("simulator.activation")]
    base_memory = base_annotations["memory_plan"]
    offload_memory = offload_annotations["memory_plan"]
    assert (
        offload_memory.steady_state_optimizer_peak_active_bytes
        < base_memory.steady_state_optimizer_peak_active_bytes
    )
