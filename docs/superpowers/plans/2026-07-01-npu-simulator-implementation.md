# torchtitan_npu Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a side-loaded `torchtitan_npu/simulator/` package that captures the four-layer IR (L0 OpNode -> L1 StepGraph -> L2 ScheduleGraph -> L3 WorkloadGraph) of one training step of any torchtitan_npu model -- proven against `deepseek_v4_pro_debug_61_layers_4k_384die` (61 layers, 384 experts, 384 simulated dies) -- with zero real NPU hardware and zero real memory allocation.

**Architecture:** See `docs/superpowers/specs/2026-07-01-npu-simulator-design.md` for full rationale. Summary: patch `torchtitan.tools.utils.device_type` to `"meta"` so `Trainer.__init__` builds/materializes/initializes the model entirely on the meta device with zero code changes to `Trainer`; force `comm.mode="fake_backend"` so `ParallelDims.build_mesh()` creates a full `world_size`-rank mesh in one process; intercept `torch.distributed`/`_functional_collectives` calls so collectives never touch the real (Meta-kernel-less) c10d dispatcher; capture every dispatched op via `TorchDispatchMode`; force MoE routing into deterministic round-robin load balancing (reusing the existing `debug.moe_force_load_balance` flag); expand the captured template across all ranks using a `RankTable` built from `ParallelDims`/`DeviceMesh`.

**Tech Stack:** Python 3.10+, PyTorch (`torch.utils._python_dispatch.TorchDispatchMode`, `torch.distributed`), pytest. NPU-specific verification requires `torch_npu` + CANN (validated in container `titan-npu-sim-validate`, image `quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12`).

## Global Constraints

- Every new file starts with this exact header:
  ```python
  # Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
  #
  # This source code is licensed under the BSD-style license found in the
  # LICENSE file in the root directory of this source tree.
  ```
- **Clean project**: do not add a dependency on, or vendor code from, any other repository (`workload-model-platform`, `torchtitan-simulator`, `virtual-npu`, etc.). The four-layer IR dataclasses are implemented fresh from the public spec at `https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec`.
- **Side-loaded, additive-only**: every task creates new files under `torchtitan_npu/simulator/` and `tests/`. No existing file in this repository is modified by this plan.
- Package root: `torchtitan_npu/simulator/`. `pyproject.toml`'s `[tool.setuptools.packages.find] include = ["torchtitan_npu*"]` already covers new subpackages -- no packaging config changes needed.
- Python `>=3.10`, ruff `target-version = "py310"`, `line-length = 120`, double-quote strings (matches `pyproject.toml`).
- Test runner: `python3 -m pytest -v --tb=short <path>` (matches `pyproject.toml` `[tool.pytest.ini_options]`; `pythonpath`/`testpaths=["tests"]` already configured).
- Unit tests needing only `torch` (no `torch_npu`) live in `tests/unit_tests/simulator/...` and run in this sandbox directly.
- Tests that call real `torch_npu.*` ops live in `tests/smoke_tests/simulator/...`, gated by a `torch_npu`-availability skip (mirrors the existing `npu_available` fixture pattern in `tests/conftest.py`), and run inside the CANN container `titan-npu-sim-validate` (see Task 20 for exact commands to reach it).
- After each task's tests pass: `git add`, `git commit`, **and `git push origin master`** (per repository workflow convention).
- Acceptance target: `torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_pro_debug_61_layers_4k_384die` (61 layers, `num_experts=384`, `expert_parallel_degree=192`, world_size implied by `384die`).

---

### Task 1: L0 IR -- TensorMeta + OpNode

**Files:**
- Create: `torchtitan_npu/simulator/__init__.py`
- Create: `torchtitan_npu/simulator/ir/__init__.py`
- Create: `torchtitan_npu/simulator/ir/tensor_meta.py`
- Create: `torchtitan_npu/simulator/ir/op_node.py`
- Test: `tests/unit_tests/simulator/ir/test_op_node.py`

**Interfaces:**
- Produces: `TensorMeta(name: str, shape: tuple[int, ...], dtype: str, device: str, is_parameter: bool = False)`; `OpNode(op_id: str, op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict, predecessors: list[str], successors: list[str], flops: int = 0, peak_mem: int = 0, param_mem: int = 0, comm_bytes: int = 0, annotations: dict = {})`.

- [ ] **Step 1: Create package `__init__.py` files**

```python
# torchtitan_npu/simulator/__init__.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Side-loaded simulator package: captures the four-layer IR (OpNode ->
StepGraph -> ScheduleGraph -> WorkloadGraph, per
https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec)
of one torchtitan_npu training step, without real NPU hardware or real
memory allocation. See
docs/superpowers/specs/2026-07-01-npu-simulator-design.md for the design.
"""
```

```python
# torchtitan_npu/simulator/ir/__init__.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four-layer simulator IR dataclasses (L0 OpNode, L1 StepGraph, L2
ScheduleGraph, L3 WorkloadGraph), implemented from the public spec at
https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec
-- this package has no dependency on that (or any other) external repo.
"""
```

Create empty `tests/unit_tests/simulator/__init__.py` and
`tests/unit_tests/simulator/ir/__init__.py` (both zero-byte files are fine;
pytest's `--import-mode=importlib` does not require `__init__.py`, but the
repo's existing `tests/unit_tests/**` subpackages all have one, so match
that convention):

```python
# tests/unit_tests/simulator/__init__.py
```

```python
# tests/unit_tests/simulator/ir/__init__.py
```

- [ ] **Step 2: Write `TensorMeta`**

```python
# torchtitan_npu/simulator/ir/tensor_meta.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 tensor metadata: see spec/L0-OpNode.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TensorMeta:
    """Minimal, framework-agnostic description of a single tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    is_parameter: bool = False
```

- [ ] **Step 3: Write `OpNode`**

```python
# torchtitan_npu/simulator/ir/op_node.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 OpNode: the smallest modeling unit in the four-layer IR. See
spec/L0-OpNode.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


@dataclass
class OpNode:
    """A single normalized operator invocation captured during a train step."""

    op_id: str
    op_type: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    attrs: dict[str, Any]
    predecessors: list[str]
    successors: list[str]
    flops: int = 0
    peak_mem: int = 0
    param_mem: int = 0
    comm_bytes: int = 0
    annotations: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Write the failing tests**

```python
# tests/unit_tests/simulator/ir/test_op_node.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


def test_tensor_meta_defaults_is_parameter_false():
    t = TensorMeta(name="x", shape=(2, 4), dtype="float32", device="meta")
    assert t.is_parameter is False


def test_tensor_meta_stores_all_fields():
    t = TensorMeta(name="w", shape=(8, 16), dtype="bfloat16", device="meta", is_parameter=True)
    assert t.name == "w"
    assert t.shape == (8, 16)
    assert t.dtype == "bfloat16"
    assert t.device == "meta"
    assert t.is_parameter is True


def test_op_node_construction_and_defaults():
    inp = TensorMeta(name="in_0", shape=(2, 4), dtype="float32", device="meta")
    out = TensorMeta(name="out_0", shape=(2, 4), dtype="float32", device="meta")
    node = OpNode(
        op_id="op_1",
        op_type="matmul",
        inputs=[inp],
        outputs=[out],
        attrs={},
        predecessors=[],
        successors=[],
    )
    assert node.flops == 0
    assert node.peak_mem == 0
    assert node.param_mem == 0
    assert node.comm_bytes == 0
    assert node.annotations == {}


def test_op_node_annotations_are_independent_between_instances():
    # dataclass default_factory must not share state across instances
    a = OpNode(op_id="a", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    b = OpNode(op_id="b", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    a.annotations["k"] = 1
    assert b.annotations == {}
```

- [ ] **Step 5: Run tests to verify they fail (import errors expected)**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_op_node.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator'` (files not yet on disk if you run this before Steps 1-3; if Steps 1-3 are already done, skip to Step 6).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_op_node.py`
Expected: `4 passed`

- [ ] **Step 7: Commit and push**

```bash
git add torchtitan_npu/simulator/__init__.py torchtitan_npu/simulator/ir/ \
        tests/unit_tests/simulator/__init__.py tests/unit_tests/simulator/ir/
git commit -m "feat(simulator): add L0 IR (TensorMeta, OpNode)"
git push origin master
```

---

### Task 2: L1 IR -- StepGraph (DAG validation)

**Files:**
- Create: `torchtitan_npu/simulator/ir/step_graph.py`
- Test: `tests/unit_tests/simulator/ir/test_step_graph.py`

**Interfaces:**
- Consumes: `OpNode` from Task 1 (`torchtitan_npu.simulator.ir.op_node.OpNode`).
- Produces: `StepGraph(step_id: str, step_type: str, nodes: dict[str, OpNode], entry_nodes: list[str] = [], exit_nodes: list[str] = [], tensor_lifetimes: dict[str, int] = {}, total_flops: int = 0, peak_active_mem: int = 0, param_mem: int = 0, comm_volume: int = 0, device_placement: dict[str, int] = {}, is_acyclic: bool = True, annotations: dict = {}, fused_regions: list = [])`. `entry_nodes`/`exit_nodes`/`is_acyclic` are auto-computed in `__post_init__` when not explicitly supplied.

- [ ] **Step 1: Write `StepGraph` with Kahn's-algorithm DAG validation**

```python
# torchtitan_npu/simulator/ir/step_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L1 StepGraph: a bounded DAG for one forward/backward/optimizer step. See
spec/L1-StepGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan_npu.simulator.ir.op_node import OpNode


def _compute_entry_exit(nodes: dict[str, OpNode]) -> tuple[list[str], list[str]]:
    """Entry nodes have no predecessors *within this graph*. A predecessor
    absent from `nodes` is external to this StepGraph -- e.g. a
    backward-phase op referencing a forward-phase activation, or an
    optimizer-phase op referencing a backward-phase gradient. Per
    spec/L1-StepGraph.md: "entry_node 的 input 无内部 producer：依赖链追溯
    到外部" -- external predecessors do not disqualify a node from being an
    entry point. Exit nodes have no successors (successors are only ever
    populated for in-graph nodes, so no such adjustment is needed there)."""
    entry = [op_id for op_id, node in nodes.items() if not any(p in nodes for p in node.predecessors)]
    exit_ = [op_id for op_id, node in nodes.items() if not node.successors]
    return entry, exit_


def _check_acyclic(nodes: dict[str, OpNode]) -> bool:
    """Kahn's algorithm restricted to in-graph edges: a predecessor that is
    not itself a key of `nodes` is external to this StepGraph and is
    treated as an already-satisfied prerequisite (not counted toward
    in-degree) -- otherwise every node with an external predecessor would
    never reach in-degree zero, and `_check_acyclic` would incorrectly
    report every backward/optimizer StepGraph as cyclic (this exact bug was
    caught by an end-to-end integration run during design: a real
    forward->backward->optimizer step produced `is_acyclic=False` for the
    backward and optimizer graphs before this fix, `True` after)."""
    in_degree = {op_id: sum(1 for p in node.predecessors if p in nodes) for op_id, node in nodes.items()}
    queue = [op_id for op_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        op_id = queue.pop(0)
        visited += 1
        for succ in nodes[op_id].successors:
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
    return visited == len(nodes)


@dataclass
class StepGraph:
    """A DAG of OpNodes for one forward, backward, or optimizer step."""

    step_id: str
    step_type: str
    nodes: dict[str, OpNode]
    entry_nodes: list[str] = field(default_factory=list)
    exit_nodes: list[str] = field(default_factory=list)
    tensor_lifetimes: dict[str, int] = field(default_factory=dict)
    total_flops: int = 0
    peak_active_mem: int = 0
    param_mem: int = 0
    comm_volume: int = 0
    device_placement: dict[str, int] = field(default_factory=dict)
    is_acyclic: bool = True
    annotations: dict[str, Any] = field(default_factory=dict)
    fused_regions: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.nodes and (not self.entry_nodes or not self.exit_nodes):
            self.entry_nodes, self.exit_nodes = _compute_entry_exit(self.nodes)
        if self.nodes:
            self.is_acyclic = _check_acyclic(self.nodes)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/ir/test_step_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def _node(op_id: str, preds: list[str], succs: list[str]) -> OpNode:
    return OpNode(op_id=op_id, op_type="x", inputs=[], outputs=[], attrs={}, predecessors=preds, successors=succs)


def test_step_graph_computes_entry_and_exit_nodes():
    nodes = {
        "a": _node("a", [], ["b"]),
        "b": _node("b", ["a"], ["c"]),
        "c": _node("c", ["b"], []),
    }
    graph = StepGraph(step_id="s1", step_type="forward", nodes=nodes)
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["c"]
    assert graph.is_acyclic is True


def test_step_graph_detects_cycle():
    nodes = {
        "a": _node("a", ["b"], ["b"]),
        "b": _node("b", ["a"], ["a"]),
    }
    graph = StepGraph(step_id="s2", step_type="forward", nodes=nodes)
    assert graph.is_acyclic is False


def test_step_graph_empty_nodes_keeps_defaults():
    graph = StepGraph(step_id="s3", step_type="forward", nodes={})
    assert graph.entry_nodes == []
    assert graph.exit_nodes == []
    assert graph.is_acyclic is True


def test_step_graph_respects_explicit_entry_exit_override():
    nodes = {"a": _node("a", [], [])}
    graph = StepGraph(step_id="s4", step_type="forward", nodes=nodes, entry_nodes=["a"], exit_nodes=["a"])
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["a"]


def test_step_graph_diamond_dependency_is_acyclic():
    # a -> b, a -> c, b -> d, c -> d (classic diamond, must stay acyclic)
    nodes = {
        "a": _node("a", [], ["b", "c"]),
        "b": _node("b", ["a"], ["d"]),
        "c": _node("c", ["a"], ["d"]),
        "d": _node("d", ["b", "c"], []),
    }
    graph = StepGraph(step_id="s5", step_type="forward", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["d"]


def test_step_graph_external_predecessor_does_not_break_acyclic_check():
    # Regression test: a node whose predecessor lives in a DIFFERENT
    # StepGraph (e.g. this is a "backward" graph and "fwd_activation" is a
    # forward-phase op_id) must not be treated as creating a cycle, and
    # must still count as an entry node -- this exact bug was caught via
    # end-to-end integration testing: before the fix, every real
    # backward/optimizer StepGraph (whose nodes reference forward/backward
    # activations and gradients as external predecessors) was incorrectly
    # reported as `is_acyclic=False`.
    nodes = {
        "b1": _node("b1", ["fwd_activation_NOT_IN_THIS_DICT"], ["b2"]),
        "b2": _node("b2", ["b1"], []),
    }
    graph = StepGraph(step_id="s6", step_type="backward", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["b1"]
    assert graph.exit_nodes == ["b2"]


def test_step_graph_multiple_external_predecessors_on_same_node():
    nodes = {
        "opt1": _node("opt1", ["grad_from_backward_1", "grad_from_backward_2"], []),
    }
    graph = StepGraph(step_id="s7", step_type="optimizer", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["opt1"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_step_graph.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.ir.step_graph'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_step_graph.py`
Expected: `7 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/ir/step_graph.py tests/unit_tests/simulator/ir/test_step_graph.py
git commit -m "feat(simulator): add L1 IR (StepGraph) with DAG validation"
git push origin master
```

---

### Task 3: L2 IR -- StepInstance/TensorSlot/DataPass/ScheduleGraph

**Files:**
- Create: `torchtitan_npu/simulator/ir/schedule_graph.py`
- Test: `tests/unit_tests/simulator/ir/test_schedule_graph.py`

**Interfaces:**
- Consumes: `StepGraph` from Task 2.
- Produces: `StepInstance(instance_id, step_ref, step_type, micro_batch_idx, pipeline_stage, device_ids, dp_group, estimated_runtime=0.0)`; `TensorSlot(name, src_exit_op, dst_entry_op, shape, dtype, volume_bytes, is_incremental=False)`; `DataPass(src_instance, dst_instance, slots, src_device=None, dst_device=None, requires_communication=False, comm_primitive="")`; `ScheduleGraph(schedule_id, workload_type, step_templates: dict[str, StepGraph], instances: list[StepInstance], instance_map={}, data_passes=[], ctrl_edges=[], dp_degree=1, tp_degree=1, pp_degree=1, num_micro_batches=1, pipeline_schedule="none", gradient_accumulation=1, zero_stage=0, timeline=[])`. `instance_map` is auto-built from `instances` in `__post_init__` when not supplied.

- [ ] **Step 1: Write the L2 dataclasses**

```python
# torchtitan_npu/simulator/ir/schedule_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L2 ScheduleGraph: describes how StepGraph instances are orchestrated --
parallel strategy, pipeline, microbatch loop, multi-device coordination.
See spec/L2-ScheduleGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class StepInstance:
    """One concrete execution of a StepGraph template."""

    instance_id: str
    step_ref: str
    step_type: str
    micro_batch_idx: int
    pipeline_stage: int
    device_ids: list[int]
    dp_group: int
    estimated_runtime: float = 0.0


@dataclass
class TensorSlot:
    """A named tensor transferred between two StepInstances."""

    name: str
    src_exit_op: str
    dst_entry_op: str
    shape: tuple[int | str, ...]
    dtype: str
    volume_bytes: int
    is_incremental: bool = False


@dataclass
class DataPass:
    """A data dependency (possibly requiring communication) between two
    StepInstances."""

    src_instance: str
    dst_instance: str
    slots: list[TensorSlot]
    src_device: int | None = None
    dst_device: int | None = None
    requires_communication: bool = False
    comm_primitive: str = ""


@dataclass
class ScheduleGraph:
    """Orchestration graph: StepGraph templates + concrete StepInstances +
    the DataPasses that connect them."""

    schedule_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    instances: list[StepInstance]
    instance_map: dict[str, StepInstance] = field(default_factory=dict)
    data_passes: list[DataPass] = field(default_factory=list)
    ctrl_edges: list[tuple[str, str]] = field(default_factory=list)
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: str = "none"
    gradient_accumulation: int = 1
    zero_stage: int = 0
    timeline: list = field(default_factory=list)
    annotations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_map and self.instances:
            self.instance_map = {instance.instance_id: instance for instance in self.instances}
```

Note: `annotations` is not in the L2 spec doc's illustrative code block but is added here (consistent with L0/L1/L3, which all carry an `annotations` field) so `rank_table.py` (Task 11) has somewhere to attach the RankTable -- see design doc §5.6.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/ir/test_schedule_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.schedule_graph import (
    DataPass,
    ScheduleGraph,
    StepInstance,
    TensorSlot,
)
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def _instance(instance_id: str, step_ref: str = "tmpl") -> StepInstance:
    return StepInstance(
        instance_id=instance_id,
        step_ref=step_ref,
        step_type="forward",
        micro_batch_idx=0,
        pipeline_stage=0,
        device_ids=[0],
        dp_group=0,
    )


def test_step_instance_defaults():
    inst = _instance("rank0")
    assert inst.estimated_runtime == 0.0


def test_tensor_slot_defaults():
    slot = TensorSlot(name="act", src_exit_op="op1", dst_entry_op="op2", shape=(2, 4), dtype="float32", volume_bytes=32)
    assert slot.is_incremental is False


def test_data_pass_defaults():
    slot = TensorSlot(name="act", src_exit_op="op1", dst_entry_op="op2", shape=(2, 4), dtype="float32", volume_bytes=32)
    dp = DataPass(src_instance="rank0", dst_instance="rank1", slots=[slot])
    assert dp.requires_communication is False
    assert dp.comm_primitive == ""
    assert dp.src_device is None


def test_schedule_graph_builds_instance_map_from_instances():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    instances = [_instance("rank0"), _instance("rank1")]
    graph = ScheduleGraph(
        schedule_id="sched1",
        workload_type="train",
        step_templates={"tmpl": template},
        instances=instances,
    )
    assert set(graph.instance_map.keys()) == {"rank0", "rank1"}
    assert graph.instance_map["rank0"] is instances[0]


def test_schedule_graph_respects_explicit_instance_map():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    instances = [_instance("rank0")]
    explicit_map = {"rank0": instances[0]}
    graph = ScheduleGraph(
        schedule_id="sched2",
        workload_type="train",
        step_templates={"tmpl": template},
        instances=instances,
        instance_map=explicit_map,
    )
    assert graph.instance_map is explicit_map


def test_schedule_graph_defaults():
    graph = ScheduleGraph(schedule_id="sched3", workload_type="train", step_templates={}, instances=[])
    assert graph.dp_degree == 1
    assert graph.pipeline_schedule == "none"
    assert graph.annotations == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_schedule_graph.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.ir.schedule_graph'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_schedule_graph.py`
Expected: `6 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/ir/schedule_graph.py tests/unit_tests/simulator/ir/test_schedule_graph.py
git commit -m "feat(simulator): add L2 IR (StepInstance/TensorSlot/DataPass/ScheduleGraph)"
git push origin master
```

---

### Task 4: L3 IR -- DataFlow/IterationSpec/WorkloadGraph

**Files:**
- Create: `torchtitan_npu/simulator/ir/workload_graph.py`
- Test: `tests/unit_tests/simulator/ir/test_workload_graph.py`

**Interfaces:**
- Consumes: `ScheduleGraph`, `DataPass` from Task 3; `StepGraph` from Task 2.
- Produces: `DataFlow(source, tensor_shape, dtype, volume_per_iter, is_streaming=False, interleave_strategy="synced")`; `IterationSpec(schedule: ScheduleGraph, microbatch_count: int, iteration_time_est=0.0)`; `WorkloadGraph(workload_id, workload_type, step_templates: dict[str, StepGraph], iteration: IterationSpec, num_iterations: int, warmup_iterations=0, data_inputs=[], data_outputs=[], cross_iter_passes=[], total_runtime_est=0.0, total_cost_est=0.0)`.

- [ ] **Step 1: Write the L3 dataclasses**

```python
# torchtitan_npu/simulator/ir/workload_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L3 WorkloadGraph: the outermost container -- holds a ScheduleGraph
template plus iteration semantics and data-flow cadence. See
spec/L3-WorkloadGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class DataFlow:
    """Describes one input or output data stream of the workload."""

    source: str
    tensor_shape: tuple[int | str, ...]
    dtype: str
    volume_per_iter: int
    is_streaming: bool = False
    interleave_strategy: str = "synced"


@dataclass
class IterationSpec:
    """One training/inference iteration: which ScheduleGraph it runs, and
    how many microbatches it contains."""

    schedule: ScheduleGraph
    microbatch_count: int
    iteration_time_est: float = 0.0


@dataclass
class WorkloadGraph:
    """Top-level container for a complete workload: train/inference/rag/
    recommendation, iteration semantics, and cross-iteration data flow."""

    workload_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    iteration: IterationSpec
    num_iterations: int
    warmup_iterations: int = 0
    data_inputs: list[DataFlow] = field(default_factory=list)
    data_outputs: list[DataFlow] = field(default_factory=list)
    cross_iter_passes: list[DataPass] = field(default_factory=list)
    total_runtime_est: float = 0.0
    total_cost_est: float = 0.0
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/ir/test_workload_graph.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph


def _empty_schedule() -> ScheduleGraph:
    return ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={}, instances=[])


def test_data_flow_defaults():
    flow = DataFlow(source="dataloader", tensor_shape=(1, 4096), dtype="int64", volume_per_iter=32768)
    assert flow.is_streaming is False
    assert flow.interleave_strategy == "synced"


def test_iteration_spec_defaults():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    assert spec.iteration_time_est == 0.0


def test_workload_graph_construction_and_defaults():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    graph = WorkloadGraph(
        workload_id="wl1",
        workload_type="train",
        step_templates={},
        iteration=spec,
        num_iterations=1,
    )
    assert graph.warmup_iterations == 0
    assert graph.data_inputs == []
    assert graph.data_outputs == []
    assert graph.cross_iter_passes == []
    assert graph.total_runtime_est == 0.0
    assert graph.total_cost_est == 0.0


def test_workload_graph_data_inputs_independent_between_instances():
    spec = IterationSpec(schedule=_empty_schedule(), microbatch_count=1)
    a = WorkloadGraph(workload_id="a", workload_type="train", step_templates={}, iteration=spec, num_iterations=1)
    b = WorkloadGraph(workload_id="b", workload_type="train", step_templates={}, iteration=spec, num_iterations=1)
    a.data_inputs.append(DataFlow(source="x", tensor_shape=(1,), dtype="int64", volume_per_iter=8))
    assert b.data_inputs == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_workload_graph.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.ir.workload_graph'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/ir/test_workload_graph.py`
Expected: `4 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/ir/workload_graph.py tests/unit_tests/simulator/ir/test_workload_graph.py
git commit -m "feat(simulator): add L3 IR (DataFlow/IterationSpec/WorkloadGraph)"
git push origin master
```

---

### Task 5: Tensor -> TensorMeta conversion + dtype/byte-size utilities

**Files:**
- Create: `torchtitan_npu/simulator/capture/__init__.py`
- Create: `torchtitan_npu/simulator/capture/tensor_utils.py`
- Test: `tests/unit_tests/simulator/capture/__init__.py`
- Test: `tests/unit_tests/simulator/capture/test_tensor_utils.py`

**Interfaces:**
- Consumes: `TensorMeta` from Task 1.
- Produces: `dtype_to_str(dtype: torch.dtype) -> str`; `dtype_byte_size(dtype_str: str) -> int`; `tensor_volume_bytes(shape: tuple[int, ...], dtype_str: str) -> int`; `to_tensor_meta(tensor: torch.Tensor, name: str, is_parameter: bool = False) -> TensorMeta`.

- [ ] **Step 1: Create the capture package**

```python
# torchtitan_npu/simulator/capture/__init__.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture layer: turns a live torchtitan_npu training step (running on the
meta device under a fake process group) into the L0/L1 IR."""
```

```python
# tests/unit_tests/simulator/capture/__init__.py
```

- [ ] **Step 2: Write `tensor_utils.py`**

```python
# torchtitan_npu/simulator/capture/tensor_utils.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conversions between torch.Tensor metadata and the simulator's
framework-agnostic TensorMeta (L0 IR)."""

from __future__ import annotations

import torch

from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta

_DTYPE_STR_OVERRIDES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_DTYPE_BYTE_SIZES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

_DEFAULT_DTYPE_BYTE_SIZE = 4  # fall back to fp32-sized for unrecognized dtypes


def dtype_to_str(dtype: torch.dtype) -> str:
    """Canonical string name for a torch dtype (e.g. `torch.bfloat16` -> `"bfloat16"`)."""
    return _DTYPE_STR_OVERRIDES.get(dtype, str(dtype).replace("torch.", ""))


def dtype_byte_size(dtype_str: str) -> int:
    """Bytes per element for a canonical dtype string."""
    return _DTYPE_BYTE_SIZES.get(dtype_str, _DEFAULT_DTYPE_BYTE_SIZE)


def tensor_volume_bytes(shape: tuple[int, ...], dtype_str: str) -> int:
    """Total byte size of a tensor with the given shape and dtype."""
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * dtype_byte_size(dtype_str)


def to_tensor_meta(tensor: torch.Tensor, name: str, is_parameter: bool = False) -> TensorMeta:
    """Build a TensorMeta from a live tensor (works for real, meta, or fake tensors --
    only `.shape`/`.dtype`/`.device` are read, never the underlying storage)."""
    return TensorMeta(
        name=name,
        shape=tuple(int(d) for d in tensor.shape),
        dtype=dtype_to_str(tensor.dtype),
        device=str(tensor.device),
        is_parameter=is_parameter,
    )
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_tensor_utils.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.tensor_utils import (
    dtype_byte_size,
    dtype_to_str,
    tensor_volume_bytes,
    to_tensor_meta,
)


def test_dtype_to_str_known_dtypes():
    assert dtype_to_str(torch.float32) == "float32"
    assert dtype_to_str(torch.bfloat16) == "bfloat16"
    assert dtype_to_str(torch.int64) == "int64"


def test_dtype_byte_size_known_and_unknown():
    assert dtype_byte_size("float32") == 4
    assert dtype_byte_size("bfloat16") == 2
    assert dtype_byte_size("int64") == 8
    assert dtype_byte_size("totally_unknown_dtype") == 4  # graceful fallback


def test_tensor_volume_bytes_computes_correctly():
    assert tensor_volume_bytes((2, 3, 4), "float32") == 2 * 3 * 4 * 4
    assert tensor_volume_bytes((10,), "bfloat16") == 20


def test_to_tensor_meta_from_meta_tensor():
    t = torch.empty(2, 3, dtype=torch.bfloat16, device="meta")
    meta = to_tensor_meta(t, name="x")
    assert meta.name == "x"
    assert meta.shape == (2, 3)
    assert meta.dtype == "bfloat16"
    assert meta.device == "meta"
    assert meta.is_parameter is False


def test_to_tensor_meta_marks_parameter():
    t = torch.empty(4, 4, device="meta")
    meta = to_tensor_meta(t, name="w", is_parameter=True)
    assert meta.is_parameter is True
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_tensor_utils.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture'`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_tensor_utils.py`
Expected: `5 passed`

- [ ] **Step 6: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/__init__.py torchtitan_npu/simulator/capture/tensor_utils.py \
        tests/unit_tests/simulator/capture/__init__.py tests/unit_tests/simulator/capture/test_tensor_utils.py
git commit -m "feat(simulator): add tensor<->TensorMeta conversion utilities"
git push origin master
```

---

### Task 6: NPU op cost model registry

**Files:**
- Create: `torchtitan_npu/simulator/cost/__init__.py`
- Create: `torchtitan_npu/simulator/cost/op_cost_model.py`
- Test: `tests/unit_tests/simulator/cost/__init__.py`
- Test: `tests/unit_tests/simulator/cost/test_op_cost_model.py`

**Interfaces:**
- Consumes: `TensorMeta` from Task 1; `tensor_volume_bytes` from Task 5.
- Produces: `CostEstimate(flops=0, peak_mem=0, param_mem=0, comm_bytes=0, unknown=False)` with classmethod `CostEstimate.unknown_cost()`; `OpCostModel().compute(op_type: str, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict | None = None) -> CostEstimate`.

- [ ] **Step 1: Create the cost package**

```python
# torchtitan_npu/simulator/cost/__init__.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extensible per-op FLOPs/memory/communication-byte cost model, used to
annotate L0 OpNodes for observability. Never raises: an op_type with no
registered handler returns a zeroed, explicitly-flagged CostEstimate (see
docs/superpowers/specs/2026-07-01-npu-simulator-design.md §5.8/§9)."""
```

```python
# tests/unit_tests/simulator/cost/__init__.py
```

- [ ] **Step 2: Write `op_cost_model.py`**

```python
# torchtitan_npu/simulator/cost/op_cost_model.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maps a canonical L0 op_type + tensor metadata to a CostEstimate.
See design doc §5.8 for the formulas and the rationale for never raising on
an unrecognized op_type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torchtitan_npu.simulator.capture.tensor_utils import tensor_volume_bytes
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


@dataclass
class CostEstimate:
    flops: int = 0
    peak_mem: int = 0
    param_mem: int = 0
    comm_bytes: int = 0
    unknown: bool = False

    @classmethod
    def unknown_cost(cls) -> "CostEstimate":
        return cls(unknown=True)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


class OpCostModel:
    """Registry of `op_type -> handler` cost estimators."""

    def __init__(self) -> None:
        Handler = Callable[[list[TensorMeta], list[TensorMeta], dict[str, Any]], CostEstimate]
        self._handlers: dict[str, Handler] = {
            "matmul": self._matmul,
            "addmm": self._matmul,
            "bmm": self._bmm,
            "grouped_mm": self._matmul,
            "sdpa": self._attention,
            "flash_attention_fwd": self._attention,
            "layer_norm": self._norm,
            "rms_norm": self._norm,
            "gelu": self._elementwise,
            "silu": self._elementwise,
            "swiglu": self._elementwise,
            "softmax": self._elementwise,
            "rope": self._elementwise,
            "moe_token_permute": self._data_move,
            "moe_token_unpermute": self._data_move,
            "moe_re_routing": self._data_move,
            "allreduce": self._allreduce,
            "reduce_scatter": self._allreduce,
            "allgather": self._allgather,
            "all_to_all": self._allgather,
        }

    def compute(
        self,
        op_type: str,
        inputs: list[TensorMeta],
        outputs: list[TensorMeta],
        attrs: dict[str, Any] | None = None,
    ) -> CostEstimate:
        handler = self._handlers.get(op_type)
        if handler is None:
            return CostEstimate.unknown_cost()
        return handler(inputs, outputs, attrs or {})

    def _matmul(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if len(inputs) < 2 or not outputs:
            return CostEstimate.unknown_cost()
        k = inputs[0].shape[-1]
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        param_bytes = tensor_volume_bytes(inputs[1].shape, inputs[1].dtype) if inputs[1].is_parameter else 0
        return CostEstimate(flops=2 * _numel(outputs[0].shape) * k, peak_mem=out_bytes, param_mem=param_bytes)

    def _bmm(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        return self._matmul(inputs, outputs, attrs)

    def _attention(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if len(inputs) < 2 or not outputs:
            return CostEstimate.unknown_cost()
        key_shape = inputs[1].shape
        seq_k = key_shape[-2] if len(key_shape) >= 2 else key_shape[-1]
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=2 * _numel(outputs[0].shape) * seq_k, peak_mem=out_bytes)

    def _norm(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs or not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=5 * _numel(inputs[0].shape), peak_mem=out_bytes)

    def _elementwise(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs or not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(flops=_numel(inputs[0].shape), peak_mem=out_bytes)

    def _data_move(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not outputs:
            return CostEstimate.unknown_cost()
        out_bytes = tensor_volume_bytes(outputs[0].shape, outputs[0].dtype)
        return CostEstimate(peak_mem=out_bytes)

    def _allreduce(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs:
            return CostEstimate.unknown_cost()
        total_bytes = tensor_volume_bytes(inputs[0].shape, inputs[0].dtype)
        return CostEstimate(comm_bytes=total_bytes * 2)  # reduce + broadcast

    def _allgather(self, inputs: list[TensorMeta], outputs: list[TensorMeta], attrs: dict[str, Any]) -> CostEstimate:
        if not inputs:
            return CostEstimate.unknown_cost()
        return CostEstimate(comm_bytes=tensor_volume_bytes(inputs[0].shape, inputs[0].dtype))
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/unit_tests/simulator/cost/test_op_cost_model.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.cost.op_cost_model import CostEstimate, OpCostModel
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


def test_unknown_op_type_returns_unknown_cost():
    model = OpCostModel()
    result = model.compute("some_op_nobody_registered", [], [], {})
    assert result == CostEstimate.unknown_cost()
    assert result.unknown is True
    assert result.flops == 0


def test_matmul_cost_matches_formula():
    model = OpCostModel()
    a = TensorMeta(name="a", shape=(8, 16), dtype="float32", device="meta")
    w = TensorMeta(name="w", shape=(16, 32), dtype="float32", device="meta", is_parameter=True)
    out = TensorMeta(name="out", shape=(8, 32), dtype="float32", device="meta")
    result = model.compute("matmul", [a, w], [out], {})
    assert result.flops == 2 * 8 * 32 * 16
    assert result.peak_mem == 8 * 32 * 4
    assert result.param_mem == 16 * 32 * 4
    assert result.unknown is False


def test_matmul_missing_inputs_returns_unknown():
    model = OpCostModel()
    out = TensorMeta(name="out", shape=(8, 32), dtype="float32", device="meta")
    result = model.compute("matmul", [], [out], {})
    assert result.unknown is True


def test_rms_norm_cost():
    model = OpCostModel()
    x = TensorMeta(name="x", shape=(2, 8, 16), dtype="float32", device="meta")
    out = TensorMeta(name="out", shape=(2, 8, 16), dtype="float32", device="meta")
    result = model.compute("rms_norm", [x], [out], {})
    assert result.flops == 5 * 2 * 8 * 16
    assert result.peak_mem == 2 * 8 * 16 * 4


def test_allreduce_cost_doubles_bytes():
    model = OpCostModel()
    t = TensorMeta(name="t", shape=(1024,), dtype="bfloat16", device="meta")
    result = model.compute("allreduce", [t], [t], {})
    assert result.comm_bytes == 1024 * 2 * 2


def test_allgather_cost_single_multiple():
    model = OpCostModel()
    t = TensorMeta(name="t", shape=(512,), dtype="float16", device="meta")
    result = model.compute("allgather", [t], [t], {})
    assert result.comm_bytes == 512 * 2


def test_moe_token_permute_is_data_move_not_flops():
    model = OpCostModel()
    tokens = TensorMeta(name="tok", shape=(64, 128), dtype="bfloat16", device="meta")
    out = TensorMeta(name="out", shape=(64, 128), dtype="bfloat16", device="meta")
    result = model.compute("moe_token_permute", [tokens], [out], {})
    assert result.flops == 0
    assert result.peak_mem == 64 * 128 * 2
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/cost/test_op_cost_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.cost'`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/cost/test_op_cost_model.py`
Expected: `7 passed`

- [ ] **Step 6: Commit and push**

```bash
git add torchtitan_npu/simulator/cost/ tests/unit_tests/simulator/cost/
git commit -m "feat(simulator): add extensible NPU op cost model"
git push origin master
```

---

### Task 7: Canonical op-name mapping (OP_MAPPING)

**Files:**
- Create: `torchtitan_npu/simulator/capture/op_mapping.py`
- Test: `tests/unit_tests/simulator/capture/test_op_mapping.py`

**Interfaces:**
- Produces: `OP_MAPPING: dict[str, str]`; `to_canonical_op_type(raw_op_type: str) -> str`.

- [ ] **Step 1: Write `op_mapping.py`**

```python
# torchtitan_npu/simulator/capture/op_mapping.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maps raw dispatcher op names (aten/npu) to canonical L0 op_type strings
(the canonical op set from spec/L0-OpNode.md, plus NPU-specific extensions
used by DeepSeek-V4: rms_norm, rope, swiglu, MoE dispatch, sparse-attention
family). Any raw op name absent from OP_MAPPING resolves to "unknown" --
OpCostModel (Task 6) already handles "unknown" op_type gracefully."""

from __future__ import annotations

OP_MAPPING: dict[str, str] = {
    # linear algebra
    "aten.addmm.default": "matmul",
    "aten.mm.default": "matmul",
    "aten.bmm.default": "bmm",
    "aten.matmul.default": "matmul",
    "aten.einsum.default": "einsum",
    # attention
    "aten.scaled_dot_product_attention.default": "sdpa",
    "aten._scaled_dot_product_flash_attention.default": "flash_attention_fwd",
    "npu.npu_fusion_attention.default": "flash_attention_fwd",
    "npu.npu_sparse_flash_attention.default": "flash_attention_fwd",
    "npu.npu_sparse_attn_sharedkv.default": "flash_attention_fwd",
    "npu.npu_lightning_indexer.default": "flash_attention_fwd",
    # normalization
    "aten.native_layer_norm.default": "layer_norm",
    "npu.npu_rms_norm.default": "rms_norm",
    # activation
    "aten.gelu.default": "gelu",
    "aten.silu.default": "silu",
    "aten.softmax.default": "softmax",
    "aten._softmax.default": "softmax",
    "npu.npu_swiglu.default": "swiglu",
    "npu.npu_rotary_mul.default": "rope",
    # MoE dispatch
    "npu.npu_moe_token_permute.default": "moe_token_permute",
    "npu.npu_moe_token_unpermute.default": "moe_token_unpermute",
    "npu.npu_moe_re_routing.default": "moe_re_routing",
    # memory / IO
    "aten.view.default": "view",
    "aten.reshape.default": "reshape",
    "aten.permute.default": "transpose",
    "aten.transpose.int": "transpose",
    "aten.cat.default": "cat",
    "aten.split.default": "split",
    "aten.split_with_sizes.default": "split",
    # optimizer
    "aten._fused_adamw_.default": "adamw_step",
}


def to_canonical_op_type(raw_op_type: str) -> str:
    """Map a raw dispatcher op name (e.g. `str(func)` from
    `__torch_dispatch__`) to its canonical L0 op_type, or `"unknown"`."""
    return OP_MAPPING.get(raw_op_type, "unknown")
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_op_mapping.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.op_mapping import OP_MAPPING, to_canonical_op_type


def test_known_aten_op_maps_to_canonical_type():
    assert to_canonical_op_type("aten.addmm.default") == "matmul"
    assert to_canonical_op_type("aten.bmm.default") == "bmm"


def test_known_npu_op_maps_to_canonical_type():
    assert to_canonical_op_type("npu.npu_rms_norm.default") == "rms_norm"
    assert to_canonical_op_type("npu.npu_moe_token_permute.default") == "moe_token_permute"
    assert to_canonical_op_type("npu.npu_rotary_mul.default") == "rope"


def test_unknown_op_maps_to_unknown():
    assert to_canonical_op_type("aten.some_brand_new_op.default") == "unknown"


def test_op_mapping_has_no_duplicate_canonical_names_missing():
    # sanity: every value should be a non-empty string
    assert all(isinstance(v, str) and v for v in OP_MAPPING.values())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_op_mapping.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.op_mapping'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_op_mapping.py`
Expected: `4 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/op_mapping.py tests/unit_tests/simulator/capture/test_op_mapping.py
git commit -m "feat(simulator): add canonical op-name mapping"
git push origin master
```

---

### Task 8: Dispatch-level op capture (TorchDispatchMode) with dedup + module-path tagging

**Files:**
- Create: `torchtitan_npu/simulator/capture/dispatch_capture.py`
- Test: `tests/unit_tests/simulator/capture/test_dispatch_capture.py`

**Interfaces:**
- Consumes: `to_canonical_op_type` (Task 7), `to_tensor_meta` (Task 5), `OpCostModel` (Task 6), `OpNode` (Task 1).
- Produces: `ModulePathTracker(root: torch.nn.Module)` context manager with `.current_path() -> str`; `OpDispatchCapture(cost_model: OpCostModel | None = None, module_path_tracker: ModulePathTracker | None = None, phase_provider: Callable[[], str] | None = None)` -- a `TorchDispatchMode` subclass usable as `with capture: ...`, exposing `.build_nodes() -> dict[str, OpNode]` after exit. Every `OpNode.annotations` always includes `"phase"` (one of `"forward"`/`"backward"`/`"optimizer"`, default `"forward"` when no `phase_provider` given) -- Task 9 reads this key to bucket nodes into per-phase StepGraphs.

- [ ] **Step 1: Write `ModulePathTracker`**

```python
# torchtitan_npu/simulator/capture/module_path.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tracks the current module call stack via forward hooks, so captured ops
can be tagged with the dotted module path (e.g. "layers.5.attention.wq")
that produced them. This tagging is what lets the HTML exporter (Task 15)
fold visually-identical repeated layers (e.g. 61 TransformerBlocks) instead
of rendering every op of every layer."""

from __future__ import annotations

import torch.nn as nn


class ModulePathTracker:
    """Context manager that maintains a stack of "currently executing
    module" names, updated via forward pre/post hooks on every submodule of
    `root`."""

    def __init__(self, root: nn.Module) -> None:
        self.root = root
        self.stack: list[str] = []
        self._handles: list[object] = []

    def __enter__(self) -> "ModulePathTracker":
        names = {id(module): name or module.__class__.__name__ for name, module in self.root.named_modules()}

        def pre_hook(module: nn.Module, _args: object) -> None:
            self.stack.append(names.get(id(module), module.__class__.__name__))

        def post_hook(module: nn.Module, _args: object, _output: object) -> None:
            if self.stack:
                self.stack.pop()

        for _, module in self.root.named_modules():
            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()  # type: ignore[attr-defined]
        self._handles.clear()

    def current_path(self) -> str:
        return self.stack[-1] if self.stack else ""
```

- [ ] **Step 2: Write `dispatch_capture.py`**

```python
# torchtitan_npu/simulator/capture/dispatch_capture.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 op-level capture via TorchDispatchMode. Captures every dispatched
operator (aten or NPU custom op) during a training step, building a
producer/consumer dependency graph keyed by `id(tensor)` (meta tensors have
no storage to alias-track, matching spec/L0-OpNode.md's "Meta tensor
环境下关闭存储级追踪，退化到纯 id(tensor) 级" rule)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan_npu.simulator.capture.module_path import ModulePathTracker
from torchtitan_npu.simulator.capture.op_mapping import to_canonical_op_type
from torchtitan_npu.simulator.capture.tensor_utils import to_tensor_meta
from torchtitan_npu.simulator.cost.op_cost_model import OpCostModel
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta

_id_counter = itertools.count()


def _next_op_id() -> str:
    return f"op_{next(_id_counter)}"


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_flatten_tensors(item))
    return tensors


@dataclass
class _RawEvent:
    op_id: str
    raw_op_type: str
    op_type: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    predecessors: list[str]
    module_path: str = ""
    phase: str = "forward"
    repeat_count: int = 1


def _shape_signature(event: _RawEvent) -> tuple:
    return (
        event.op_type,
        event.module_path,
        event.phase,
        tuple(tuple(i.shape) for i in event.inputs),
        tuple(tuple(o.shape) for o in event.outputs),
    )


class OpDispatchCapture(TorchDispatchMode):
    """Records one L0 op stream. Usage::

        capture = OpDispatchCapture()
        with capture:
            out = model(x)
            out.sum().backward()
        nodes = capture.build_nodes()
    """

    def __init__(
        self,
        cost_model: OpCostModel | None = None,
        module_path_tracker: ModulePathTracker | None = None,
        phase_provider: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self.cost_model = cost_model or OpCostModel()
        self.module_path_tracker = module_path_tracker
        self.phase_provider = phase_provider
        self._events: list[_RawEvent] = []
        self._producer: dict[int, str] = {}
        self._last_signature: tuple | None = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN201
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        flat_inputs = _flatten_tensors(args) + _flatten_tensors(tuple(kwargs.values()))
        flat_outputs = _flatten_tensors(result if isinstance(result, (tuple, list)) else (result,))

        predecessors = sorted({self._producer[id(t)] for t in flat_inputs if id(t) in self._producer})
        input_metas = [to_tensor_meta(t, name=f"in_{i}") for i, t in enumerate(flat_inputs)]
        output_metas = [to_tensor_meta(t, name=f"out_{i}") for i, t in enumerate(flat_outputs)]

        raw_op_type = str(func)
        op_type = to_canonical_op_type(raw_op_type)
        module_path = self.module_path_tracker.current_path() if self.module_path_tracker else ""
        phase = self.phase_provider() if self.phase_provider else "forward"

        candidate = _RawEvent(
            op_id="",
            raw_op_type=raw_op_type,
            op_type=op_type,
            inputs=input_metas,
            outputs=output_metas,
            predecessors=predecessors,
            module_path=module_path,
            phase=phase,
        )
        signature = _shape_signature(candidate)

        if self._events and signature == self._last_signature:
            retained = self._events[-1]
            retained.repeat_count += 1
            op_id = retained.op_id
        else:
            op_id = _next_op_id()
            candidate.op_id = op_id
            self._events.append(candidate)
            self._last_signature = signature

        # Always (re)bind producer ids to whichever event now represents this
        # position, even when the event itself was deduped away, so later
        # ops' predecessor lookups stay correct.
        for t in flat_outputs:
            self._producer[id(t)] = op_id

        return result

    def build_nodes(self) -> dict[str, OpNode]:
        """Assemble captured events into OpNode objects with cost annotations."""
        nodes: dict[str, OpNode] = {}
        for event in self._events:
            cost = self.cost_model.compute(event.op_type, event.inputs, event.outputs, {})
            annotations: dict[str, Any] = {"raw_op_type": event.raw_op_type, "phase": event.phase}
            if event.module_path:
                annotations["module_path"] = event.module_path
            if event.repeat_count > 1:
                annotations["repeat_count"] = event.repeat_count
            if cost.unknown:
                annotations["cost_unknown"] = True
            nodes[event.op_id] = OpNode(
                op_id=event.op_id,
                op_type=event.op_type,
                inputs=event.inputs,
                outputs=event.outputs,
                attrs={},
                predecessors=list(event.predecessors),
                successors=[],
                flops=cost.flops,
                peak_mem=cost.peak_mem,
                param_mem=cost.param_mem,
                comm_bytes=cost.comm_bytes,
                annotations=annotations,
            )
        for op_id, node in nodes.items():
            for pred_id in node.predecessors:
                if pred_id in nodes:
                    nodes[pred_id].successors.append(op_id)
        return nodes
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_dispatch_capture.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.module_path import ModulePathTracker


def test_capture_records_ops_on_meta_tensors():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(4, 8, device="meta")
        b = torch.randn(8, 16, device="meta")
        c = a @ b
        c.sum()
    nodes = capture.build_nodes()
    assert len(nodes) >= 3  # randn, randn, matmul, sum (at least)


def test_capture_builds_predecessor_successor_edges():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(4, 8, device="meta")
        b = a.relu()
        b.sum()
    nodes = capture.build_nodes()
    relu_nodes = [n for n in nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert len(relu_nodes) == 1
    relu_node = relu_nodes[0]
    assert len(relu_node.predecessors) == 1
    producer = nodes[relu_node.predecessors[0]]
    assert relu_node.op_id in producer.successors


def test_capture_deduplicates_consecutive_identical_ops():
    capture = OpDispatchCapture()
    with capture:
        x = torch.zeros(4, device="meta")
        for _ in range(5):
            x = x.relu()
    nodes = capture.build_nodes()
    relu_nodes = [n for n in nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert len(relu_nodes) == 1
    assert relu_nodes[0].annotations["repeat_count"] == 5


def test_capture_tags_module_path_when_tracker_supplied():
    model = nn.Sequential(nn.Linear(4, 8, device="meta"), nn.ReLU())
    tracker = ModulePathTracker(model)
    capture = OpDispatchCapture(module_path_tracker=tracker)
    with tracker, capture:
        model(torch.randn(2, 4, device="meta"))
    nodes = capture.build_nodes()
    tagged = [n for n in nodes.values() if "module_path" in n.annotations]
    assert tagged, "expected at least one op tagged with a module_path"
    assert any("0" in n.annotations["module_path"] for n in tagged)  # Sequential child "0" (Linear)


def test_unknown_op_type_is_flagged_in_annotations():
    capture = OpDispatchCapture()
    with capture:
        # aten.arange.default has no entry in OP_MAPPING -> canonical "unknown"
        torch.arange(4, device="meta")
    nodes = capture.build_nodes()
    unknown_nodes = [n for n in nodes.values() if n.op_type == "unknown"]
    assert unknown_nodes
    assert all(n.annotations.get("cost_unknown") for n in unknown_nodes)


def test_phase_provider_tags_every_node_and_defaults_to_forward():
    capture_no_provider = OpDispatchCapture()
    with capture_no_provider:
        torch.randn(2, 2, device="meta")
    default_nodes = capture_no_provider.build_nodes()
    assert all(n.annotations["phase"] == "forward" for n in default_nodes.values())

    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        torch.randn(2, 2, device="meta")
        phase_box["value"] = "backward"
        torch.randn(2, 2, device="meta")
    nodes = capture.build_nodes()
    phases = sorted({n.annotations["phase"] for n in nodes.values()})
    assert phases == ["backward", "forward"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_dispatch_capture.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.dispatch_capture'`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_dispatch_capture.py`
Expected: `6 passed`

If `test_capture_tags_module_path_when_tracker_supplied` fails because the
tagged path doesn't contain `"0"`, print `nodes` and inspect actual
`module_path` values -- `nn.Sequential` names children by index, but the
exact hook-firing order for a 2-op `Linear` (addmm) can attribute the op to
either `""` (root) or `"0"` depending on hook nesting; adjust the assertion
to check `tracker.stack` behavior directly if needed, keeping the test's
intent (module path tagging works) rather than a brittle exact string match.

- [ ] **Step 6: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/module_path.py torchtitan_npu/simulator/capture/dispatch_capture.py \
        tests/unit_tests/simulator/capture/test_dispatch_capture.py
git commit -m "feat(simulator): add TorchDispatchMode-based L0 op capture with dedup, module-path, and phase tagging"
git push origin master
```

---

### Task 9: Step boundary hooks -> L1 StepGraph assembly

**Files:**
- Create: `torchtitan_npu/simulator/capture/step_boundary.py`
- Test: `tests/unit_tests/simulator/capture/test_step_boundary.py`

**Interfaces:**
- Consumes: `OpNode` (Task 1), `StepGraph` (Task 2). Reads `OpNode.annotations["phase"]` as produced by Task 8's `OpDispatchCapture(phase_provider=...)`.
- Produces: `StepBoundaryTracker()` context manager exposing `.current_phase: str` (starts `"forward"`) and `.mark(phase: str) -> None`; `build_step_graphs(nodes: dict[str, OpNode]) -> dict[str, StepGraph]` (buckets by `node.annotations["phase"]`, default `"forward"` if absent).

- [ ] **Step 1: Write `step_boundary.py`**

```python
# torchtitan_npu/simulator/capture/step_boundary.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Detects forward/backward/optimizer step boundaries (see
spec/L1-StepGraph.md: "框架通过 autograd.backward hook + Optimizer.step
wrapper 自动识别边界") and buckets already-captured OpNodes into per-phase
StepGraphs."""

from __future__ import annotations

import uuid
from typing import Callable

import torch

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph

_PHASES = ("forward", "backward", "optimizer")


def _collect_optimizer_classes() -> list[type]:
    """Every currently-imported subclass of torch.optim.Optimizer (AdamW,
    Muon, swap/virtual optimizer wrappers, etc.)."""
    result: list[type] = []

    def _recurse(base: type) -> None:
        for sub in base.__subclasses__():
            result.append(sub)
            _recurse(sub)

    _recurse(torch.optim.Optimizer)
    return result


class StepBoundaryTracker:
    """Context manager that monkeypatches `torch.Tensor.backward` and every
    currently-loaded `Optimizer.step` to flip `self.current_phase`, plus
    exposes `.mark()` for callers that want to set the phase explicitly
    (e.g. before/after a pipeline-parallel schedule step)."""

    def __init__(self) -> None:
        self.current_phase = "forward"
        self._original_backward: Callable | None = None
        self._original_optimizer_steps: dict[type, Callable] = {}

    def __enter__(self) -> "StepBoundaryTracker":
        self.current_phase = "forward"
        self._original_backward = torch.Tensor.backward
        tracker = self

        def hooked_backward(self_tensor, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            tracker.current_phase = "backward"
            return tracker._original_backward(self_tensor, *args, **kwargs)  # type: ignore[misc]

        torch.Tensor.backward = hooked_backward  # type: ignore[method-assign]

        for optimizer_cls in _collect_optimizer_classes():
            if "step" not in optimizer_cls.__dict__:
                continue
            original_step = optimizer_cls.step
            self._original_optimizer_steps[optimizer_cls] = original_step

            def make_hooked_step(orig: Callable) -> Callable:
                def hooked_step(self_opt, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                    tracker.current_phase = "optimizer"
                    return orig(self_opt, *args, **kwargs)

                return hooked_step

            optimizer_cls.step = make_hooked_step(original_step)  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._original_backward is not None:
            torch.Tensor.backward = self._original_backward  # type: ignore[method-assign]
        for cls, original_step in self._original_optimizer_steps.items():
            cls.step = original_step  # type: ignore[method-assign]
        self._original_optimizer_steps.clear()

    def mark(self, phase: str) -> None:
        """Explicitly set the current phase."""
        self.current_phase = phase


def build_step_graphs(nodes: dict[str, OpNode]) -> dict[str, StepGraph]:
    """Bucket OpNodes into forward/backward/optimizer StepGraphs using each
    node's `annotations["phase"]` (defaults to `"forward"` if the tag is
    missing, e.g. a node captured without a `phase_provider`)."""
    buckets: dict[str, dict[str, OpNode]] = {phase: {} for phase in _PHASES}
    for op_id, node in nodes.items():
        phase = node.annotations.get("phase", "forward")
        buckets.setdefault(phase, {})[op_id] = node

    graphs: dict[str, StepGraph] = {}
    for phase, phase_nodes in buckets.items():
        if not phase_nodes:
            continue
        graphs[phase] = StepGraph(step_id=uuid.uuid4().hex[:12], step_type=phase, nodes=phase_nodes)
    return graphs
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_step_boundary.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker, build_step_graphs
from torchtitan_npu.simulator.ir.op_node import OpNode


def _node(op_id: str, phase: str) -> OpNode:
    return OpNode(
        op_id=op_id, op_type="x", inputs=[], outputs=[], attrs={},
        predecessors=[], successors=[], annotations={"phase": phase},
    )


def test_build_step_graphs_buckets_by_phase():
    nodes = {
        "f1": _node("f1", "forward"),
        "b1": _node("b1", "backward"),
        "o1": _node("o1", "optimizer"),
    }
    graphs = build_step_graphs(nodes)
    assert set(graphs.keys()) == {"forward", "backward", "optimizer"}
    assert graphs["forward"].step_type == "forward"
    assert "f1" in graphs["forward"].nodes
    assert "b1" in graphs["backward"].nodes
    assert "o1" in graphs["optimizer"].nodes


def test_build_step_graphs_defaults_missing_phase_to_forward():
    node_without_phase = OpNode(
        op_id="x", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[], annotations={},
    )
    graphs = build_step_graphs({"x": node_without_phase})
    assert "x" in graphs["forward"].nodes


def test_build_step_graphs_skips_empty_phases():
    nodes = {"f1": _node("f1", "forward")}
    graphs = build_step_graphs(nodes)
    assert "backward" not in graphs
    assert "optimizer" not in graphs


def test_step_boundary_tracker_flips_phase_on_backward_call():
    tracker = StepBoundaryTracker()
    with tracker:
        assert tracker.current_phase == "forward"
        x = torch.randn(4, device="meta", requires_grad=True)
        x.sum().backward()
        assert tracker.current_phase == "backward"


def test_step_boundary_tracker_restores_original_backward_on_exit():
    original = torch.Tensor.backward
    tracker = StepBoundaryTracker()
    with tracker:
        pass
    assert torch.Tensor.backward is original


def test_step_boundary_tracker_mark_sets_phase_explicitly():
    tracker = StepBoundaryTracker()
    with tracker:
        tracker.mark("optimizer")
        assert tracker.current_phase == "optimizer"


def test_step_boundary_tracker_integrates_with_dispatch_capture():
    tracker = StepBoundaryTracker()
    capture = OpDispatchCapture(phase_provider=lambda: tracker.current_phase)
    with tracker, capture:
        x = torch.randn(4, device="meta", requires_grad=True)
        y = x.relu()
        y.sum().backward()
    nodes = capture.build_nodes()
    graphs = build_step_graphs(nodes)
    assert "forward" in graphs
    assert "backward" in graphs
    relu_nodes = [n for n in graphs["forward"].nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert relu_nodes, "relu should have been captured during the forward phase"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_step_boundary.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.step_boundary'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_step_boundary.py`
Expected: `7 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/step_boundary.py tests/unit_tests/simulator/capture/test_step_boundary.py
git commit -m "feat(simulator): add step-boundary hooks and L1 StepGraph assembly"
git push origin master
```

---

### Task 10: Fake collectives interception layer + comm event recording

**Files:**
- Create: `torchtitan_npu/simulator/capture/comm_events.py`
- Test: `tests/unit_tests/simulator/capture/test_comm_events.py`

**Interfaces:**
- Consumes: `is_fake_process_group` from `torchtitan_npu.distributed.process_group` (existing repo module); `dtype_to_str`/`tensor_volume_bytes` from Task 5.
- Produces: `CommEvent(event_id, comm_primitive, group_name, world_size, tensor_shape, dtype, volume_bytes)`; `CommEventRecorder` with `.events: list[CommEvent]`; `capture_fake_collectives()` context manager yielding a `CommEventRecorder`.

This is the component that fixes the crash found empirically in
docs/superpowers/specs/2026-07-01-npu-simulator-design.md §2 finding #4:
`dist.all_reduce`/etc. raise `NotImplementedError` on meta tensors even
under a fake ProcessGroup, because c10d collective ops have no Meta kernel.

- [ ] **Step 1: Write `comm_events.py`**

```python
# torchtitan_npu/simulator/capture/comm_events.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Intercepts torch.distributed collective calls so they can run on
meta-device tensors under a FakeProcessGroup without touching the real
c10d dispatcher (see design doc §2 finding #4 and §5.2). Generalizes the
`is_fake_process_group` short-circuit pattern already used by
`torchtitan_npu.converters.kernels.moe_dispatch.NpuExpertParallel` to every
collective entry point (FSDP2 all-gather/reduce-scatter, TP all-reduce, DP
grad all-reduce, broadcast)."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from torchtitan_npu.distributed.process_group import is_fake_process_group
from torchtitan_npu.simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes

_event_counter = itertools.count()


@dataclass
class CommEvent:
    event_id: str
    comm_primitive: str
    group_name: str
    world_size: int
    tensor_shape: tuple[int, ...]
    dtype: str
    volume_bytes: int


class _NoOpWork:
    """Minimal stand-in for `torch.distributed.Work`, so callers that use
    the `async_op=True` idiom (call `.wait()` on the return value) do not
    crash when we skip the real collective."""

    def wait(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def is_completed(self) -> bool:
        return True


class CommEventRecorder:
    def __init__(self) -> None:
        self.events: list[CommEvent] = []

    def record(self, comm_primitive: str, group: object, tensor: torch.Tensor) -> None:
        dtype_str = dtype_to_str(tensor.dtype)
        world_size = dist.get_world_size(group) if dist.is_initialized() else 1  # type: ignore[arg-type]
        self.events.append(
            CommEvent(
                event_id=f"comm_{next(_event_counter)}",
                comm_primitive=comm_primitive,
                group_name=_group_name(group),
                world_size=world_size,
                tensor_shape=tuple(int(d) for d in tensor.shape),
                dtype=dtype_str,
                volume_bytes=tensor_volume_bytes(tuple(tensor.shape), dtype_str),
            )
        )


def _group_name(group: object) -> str:
    if group is None:
        return "default"
    name = getattr(group, "group_name", None)
    return str(name) if name is not None else "default"


@contextmanager
def capture_fake_collectives() -> Iterator[CommEventRecorder]:
    """Monkeypatch the legacy (`torch.distributed.*`) and functional
    (`torch.distributed._functional_collectives.*`) collective APIs for the
    duration of the context.

    Legacy APIs receive a real `ProcessGroup` (or `None`) as their `group`
    argument, so they defensively check `is_fake_process_group(group)` and
    fall back to the real implementation when it is not fake (keeps this
    module safe to import even outside a simulation run).

    Functional-collective APIs (used internally by DTensor/FSDP2) accept a
    `ProcessGroup`, `DeviceMesh`, list of ranks, or group-name string as
    `group` -- resolving all of those reliably is fragile (see design doc
    §2 finding #4 discussion). Because this context manager is only ever
    active for the full duration of one simulated training step, and the
    simulator always runs entirely under a fake backend (never a mix of
    real and fake groups), the functional-collective patches always treat
    calls made while the context is active as fake, unconditionally.
    """
    recorder = CommEventRecorder()

    orig_all_reduce = dist.all_reduce
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_all_to_all_single = dist.all_to_all_single
    orig_broadcast = dist.broadcast
    orig_barrier = dist.barrier

    orig_funcol_all_reduce = funcol.all_reduce
    orig_funcol_all_gather_tensor = funcol.all_gather_tensor
    orig_funcol_reduce_scatter_tensor = funcol.reduce_scatter_tensor
    orig_funcol_all_to_all_single = funcol.all_to_all_single

    def patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
        recorder.record("allreduce", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)
        recorder.record("allgather", group, input_tensor)
        return _NoOpWork() if async_op else None

    def patched_reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)
        recorder.record("reduce_scatter", group, input)
        return _NoOpWork() if async_op else None

    def patched_all_to_all_single(  # noqa: ANN001
        output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False
    ):
        if not is_fake_process_group(group):
            return orig_all_to_all_single(
                output, input, output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes, group=group, async_op=async_op,
            )
        recorder.record("all_to_all", group, input)
        return _NoOpWork() if async_op else None

    def patched_broadcast(tensor, src=0, group=None, async_op=False, group_src=None):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_broadcast(tensor, src=src, group=group, async_op=async_op, group_src=group_src)
        recorder.record("broadcast", group, tensor)
        return _NoOpWork() if async_op else None

    def patched_barrier(group=None, async_op=False, device_ids=None):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)
        return _NoOpWork() if async_op else None

    def patched_funcol_all_reduce(self_tensor, reduceOp, group, tag=""):  # noqa: ANN001, N803
        recorder.record("allreduce", None, self_tensor)
        return self_tensor.clone()

    def patched_funcol_all_gather_tensor(self_tensor, gather_dim, group, tag=""):  # noqa: ANN001
        recorder.record("allgather", None, self_tensor)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        out_shape = list(self_tensor.shape)
        out_shape[gather_dim] *= world_size
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    def patched_funcol_reduce_scatter_tensor(self_tensor, reduceOp, scatter_dim, group, tag=""):  # noqa: ANN001, N803
        recorder.record("reduce_scatter", None, self_tensor)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        out_shape = list(self_tensor.shape)
        out_shape[scatter_dim] //= world_size
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    def patched_funcol_all_to_all_single(self_tensor, output_split_sizes, input_split_sizes, group, tag=""):  # noqa: ANN001
        recorder.record("all_to_all", None, self_tensor)
        out_shape = list(self_tensor.shape)
        if output_split_sizes:
            out_shape[0] = int(sum(output_split_sizes))
        return torch.empty(out_shape, dtype=self_tensor.dtype, device=self_tensor.device)

    dist.all_reduce = patched_all_reduce
    dist.all_gather_into_tensor = patched_all_gather_into_tensor
    dist.reduce_scatter_tensor = patched_reduce_scatter_tensor
    dist.all_to_all_single = patched_all_to_all_single
    dist.broadcast = patched_broadcast
    dist.barrier = patched_barrier
    funcol.all_reduce = patched_funcol_all_reduce
    funcol.all_gather_tensor = patched_funcol_all_gather_tensor
    funcol.reduce_scatter_tensor = patched_funcol_reduce_scatter_tensor
    funcol.all_to_all_single = patched_funcol_all_to_all_single

    try:
        yield recorder
    finally:
        dist.all_reduce = orig_all_reduce
        dist.all_gather_into_tensor = orig_all_gather_into_tensor
        dist.reduce_scatter_tensor = orig_reduce_scatter_tensor
        dist.all_to_all_single = orig_all_to_all_single
        dist.broadcast = orig_broadcast
        dist.barrier = orig_barrier
        funcol.all_reduce = orig_funcol_all_reduce
        funcol.all_gather_tensor = orig_funcol_all_gather_tensor
        funcol.reduce_scatter_tensor = orig_funcol_reduce_scatter_tensor
        funcol.all_to_all_single = orig_funcol_all_to_all_single
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_comm_events.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives


@pytest.fixture(scope="module", autouse=True)
def _fake_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29713"
    dist.init_process_group("fake", rank=0, world_size=8)
    yield
    dist.destroy_process_group()


def test_all_reduce_on_meta_tensor_is_noop_and_recorded():
    t = torch.randn(16, 16, device="meta")
    with capture_fake_collectives() as recorder:
        result = dist.all_reduce(t)
    assert result is None
    assert len(recorder.events) == 1
    assert recorder.events[0].comm_primitive == "allreduce"
    assert recorder.events[0].tensor_shape == (16, 16)


def test_all_gather_into_tensor_on_meta_is_noop_and_recorded():
    input_t = torch.randn(4, device="meta")
    output_t = torch.empty(32, device="meta")
    with capture_fake_collectives() as recorder:
        dist.all_gather_into_tensor(output_t, input_t)
    assert output_t.shape == (32,)  # caller-preallocated shape untouched
    assert recorder.events[0].comm_primitive == "allgather"


def test_all_to_all_single_on_meta_is_noop_and_recorded():
    input_t = torch.randn(8, device="meta")
    output_t = torch.empty(8, device="meta")
    with capture_fake_collectives() as recorder:
        dist.all_to_all_single(output_t, input_t)
    assert recorder.events[0].comm_primitive == "all_to_all"


def test_funcol_all_gather_tensor_returns_correctly_shaped_new_tensor():
    t = torch.randn(4, 8, device="meta")
    with capture_fake_collectives() as recorder:
        out = funcol.all_gather_tensor(t, gather_dim=0, group=dist.group.WORLD)
    assert out.shape == (32, 8)  # 4 * world_size(8)
    assert recorder.events[0].comm_primitive == "allgather"


def test_funcol_all_to_all_single_respects_output_split_sizes():
    t = torch.randn(10, device="meta")
    with capture_fake_collectives() as recorder:
        out = funcol.all_to_all_single(t, [3, 4], [5, 5], group=dist.group.WORLD)
    assert out.shape == (7,)
    assert recorder.events[0].comm_primitive == "all_to_all"


def test_collectives_restored_after_context_exit():
    original_all_reduce = dist.all_reduce
    with capture_fake_collectives():
        assert dist.all_reduce is not original_all_reduce
    assert dist.all_reduce is original_all_reduce
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_comm_events.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.comm_events'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_comm_events.py`
Expected: `6 passed`

If `test_all_reduce_on_meta_tensor_is_noop_and_recorded` (or any legacy-API
test) fails with `NotImplementedError` instead of being intercepted, check
`torchtitan_npu.distributed.process_group.is_fake_process_group(dist.group.WORLD)`
directly in a REPL against the exact torch version in use --
`dist.get_backend(group)` must return `"fake"` for the default group when
initialized via `dist.init_process_group("fake", ...)` (this repo's own
`is_fake_process_group` implementation is the source of truth; do not
modify it from this task).

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/comm_events.py tests/unit_tests/simulator/capture/test_comm_events.py
git commit -m "feat(simulator): add fake-collective interception and comm event recording"
git push origin master
```

---

### Task 11: RankTable construction from ParallelDims/DeviceMesh

**Files:**
- Create: `torchtitan_npu/simulator/rank_table.py`
- Test: `tests/unit_tests/simulator/test_rank_table.py`

**Interfaces:**
- Consumes: a `torchtitan.distributed.parallel_dims.ParallelDims` instance (after `.build_mesh()` has run), via its public fields (`world_size`, `pp`, `dp_replicate`, `dp_shard`, `cp`, `tp`, `ep`) and its `_global_meshes: dict[str, DeviceMesh]` attribute (already read/written directly by this repo's own `torchtitan_npu/train.py::_patch_for_parallel_dims_build_mesh`, so treated as a stable extension point here too -- see design doc §5.6).
- Produces: `RankTable(world_size, dim_degrees: dict[str, int], rank_coordinates: dict[int, dict[str, int]], process_groups: dict[str, list[list[int]]], dim_by_group_name: dict[str, str])` with `.to_dict() -> dict` (JSON-serializable); `build_rank_table(parallel_dims) -> RankTable`.

This task's mesh-traversal approach is verified against a real
`ParallelDims.build_mesh()` run under a fake process group (see design doc
§5.6): reading the *composite* meshes (`_global_meshes["dense"/"sparse"/...]`)
is required because a single-axis mesh (`get_optional_mesh("ep")`) only
exposes the caller's own group, not every group.

- [ ] **Step 1: Write `rank_table.py`**

```python
# torchtitan_npu/simulator/rank_table.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expands ParallelDims/DeviceMesh into a communication-domain RankTable:
for every named mesh axis (pp, dp_replicate, fsdp/dp_shard, cp, tp, ep,
efsdp, and torchtitan_npu's own "etp" once
`_patch_for_parallel_dims_build_mesh` has run), which global ranks belong
to each communication group, and each rank's coordinate along every axis.
See design doc §5.6."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankTable:
    world_size: int
    dim_degrees: dict[str, int] = field(default_factory=dict)
    rank_coordinates: dict[int, dict[str, int]] = field(default_factory=dict)
    process_groups: dict[str, list[list[int]]] = field(default_factory=dict)
    dim_by_group_name: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "dim_degrees": dict(self.dim_degrees),
            "rank_coordinates": {str(rank): dict(coords) for rank, coords in self.rank_coordinates.items()},
            "process_groups": {dim: [list(group) for group in groups] for dim, groups in self.process_groups.items()},
        }


def _groups_along_axis(full_tensor: Any, axis_pos: int) -> list[list[int]]:
    """Every group of ranks that varies along `axis_pos`, with every other
    axis held fixed -- i.e. every "row" of the mesh along that axis."""
    other_dims = [d for d in range(full_tensor.dim()) if d != axis_pos]
    ranges = [range(full_tensor.shape[d]) for d in other_dims]
    groups: list[list[int]] = []
    for combo in itertools.product(*ranges) if ranges else [()]:
        index: list[Any] = [slice(None)] * full_tensor.dim()
        for dim, value in zip(other_dims, combo):
            index[dim] = value
        groups.append([int(r) for r in full_tensor[tuple(index)].flatten().tolist()])
    return groups


def build_rank_table(parallel_dims: Any) -> RankTable:
    """Expand `parallel_dims` (after `.build_mesh()`) into a RankTable."""
    world_size = int(parallel_dims.world_size)
    dim_degrees: dict[str, int] = {
        "pp": int(parallel_dims.pp),
        "dp_replicate": int(parallel_dims.dp_replicate),
        "dp_shard": int(parallel_dims.dp_shard),
        "cp": int(parallel_dims.cp),
        "tp": int(parallel_dims.tp),
        "ep": int(parallel_dims.ep),
    }

    process_groups: dict[str, list[list[int]]] = {}
    dim_by_group_name: dict[str, str] = {}

    composite_meshes = getattr(parallel_dims, "_global_meshes", {}) or {}
    for composite in composite_meshes.values():
        mesh_dim_names = getattr(composite, "mesh_dim_names", None)
        if not mesh_dim_names:
            continue
        full_tensor = composite.mesh
        for axis_pos, axis_name in enumerate(mesh_dim_names):
            if axis_name in process_groups:
                continue  # already captured from an earlier composite mesh
            groups = _groups_along_axis(full_tensor, axis_pos)
            process_groups[axis_name] = groups
            dim_degrees.setdefault(axis_name, int(full_tensor.shape[axis_pos]))
            try:
                group_name = str(composite[axis_name].get_group().group_name)
                dim_by_group_name[group_name] = axis_name
            except (ValueError, RuntimeError, AttributeError):
                pass  # single-axis view unavailable (e.g. degree-1 dim) -- harmless

    # Any dimension never discovered via a composite mesh (e.g. tp/cp
    # disabled, degree 1) still gets a trivial per-rank singleton group.
    for name, degree in list(dim_degrees.items()):
        if name not in process_groups:
            process_groups[name] = [[rank] for rank in range(world_size)]

    rank_coordinates: dict[int, dict[str, int]] = {rank: {} for rank in range(world_size)}
    for name, groups in process_groups.items():
        for group in groups:
            for idx, rank in enumerate(group):
                if 0 <= rank < world_size:
                    rank_coordinates[rank][name] = idx

    return RankTable(
        world_size=world_size,
        dim_degrees=dim_degrees,
        rank_coordinates=rank_coordinates,
        process_groups=process_groups,
        dim_by_group_name=dim_by_group_name,
    )
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/test_rank_table.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch.distributed as dist

from torchtitan_npu.simulator.rank_table import build_rank_table


@pytest.fixture
def fake_world():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29812"
    dist.init_process_group("fake", rank=0, world_size=16)
    yield
    dist.destroy_process_group()


def test_build_rank_table_matches_real_parallel_dims_mesh(fake_world):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)

    assert table.world_size == 16
    assert table.dim_degrees["ep"] == 8
    # verified by hand in design doc §5.6: ep groups are contiguous blocks
    ep_groups = sorted(table.process_groups["ep"], key=lambda g: g[0])
    assert ep_groups == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    assert table.rank_coordinates[0]["ep"] == 0
    assert table.rank_coordinates[8]["ep"] == 0
    assert table.rank_coordinates[9]["ep"] == 1


def test_build_rank_table_every_rank_has_coordinates_for_every_group_dim(fake_world):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)
    for rank in range(16):
        assert rank in table.rank_coordinates
        for dim_name in table.process_groups:
            assert dim_name in table.rank_coordinates[rank]


def test_rank_table_to_dict_is_json_serializable(fake_world):
    import json

    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=16, cp=1, tp=1, pp=1, ep=8, world_size=16)
    parallel_dims.build_mesh()

    table = build_rank_table(parallel_dims)
    serialized = json.dumps(table.to_dict())
    assert '"world_size": 16' in serialized
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_rank_table.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.rank_table'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_rank_table.py`
Expected: `3 passed`. This test genuinely exercises `torchtitan.distributed.parallel_dims.ParallelDims` (no `torch_npu` needed -- verified directly in this sandbox during design).

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/rank_table.py tests/unit_tests/simulator/test_rank_table.py
git commit -m "feat(simulator): add RankTable construction from ParallelDims/DeviceMesh"
git push origin master
```

---

### Task 12: L2 ScheduleGraph builder

**Files:**
- Create: `torchtitan_npu/simulator/capture/schedule_builder.py`
- Test: `tests/unit_tests/simulator/capture/test_schedule_builder.py`

**Interfaces:**
- Consumes: `StepGraph` (Task 2); `StepInstance`/`TensorSlot`/`DataPass`/`ScheduleGraph` (Task 3); `CommEvent` (Task 10); `RankTable` (Task 11).
- Produces: `build_schedule_graph(step_templates: dict[str, StepGraph], rank_table: RankTable, comm_events: list[CommEvent], pipeline_schedule: str = "none", num_micro_batches: int = 1, gradient_accumulation: int = 1) -> ScheduleGraph`.

- [ ] **Step 1: Write `schedule_builder.py`**

```python
# torchtitan_npu/simulator/capture/schedule_builder.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assembles the L2 ScheduleGraph from a captured L1 template, the
RankTable, and recorded communication events. See design doc §5.5."""

from __future__ import annotations

import uuid

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph, StepInstance, TensorSlot
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.rank_table import RankTable


def build_schedule_graph(
    *,
    step_templates: dict[str, StepGraph],
    rank_table: RankTable,
    comm_events: list[CommEvent],
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
) -> ScheduleGraph:
    """Build one StepInstance per logical rank -- all ranks share the
    single captured template when `pipeline_parallel_degree == 1` (the
    acceptance config's case; see design doc §5.5 for the general
    per-pipeline-stage template note, out of scope for this task) -- plus
    one DataPass per communication-group member pair for every recorded
    CommEvent whose `group_name` resolves to a known RankTable dimension.
    """
    template_id = next(iter(step_templates), "")
    template_step_type = step_templates[template_id].step_type if template_id else "forward"

    instances: list[StepInstance] = []
    for rank in range(rank_table.world_size):
        coords = rank_table.rank_coordinates.get(rank, {})
        instances.append(
            StepInstance(
                instance_id=f"rank{rank}",
                step_ref=template_id,
                step_type=template_step_type,
                micro_batch_idx=0,
                pipeline_stage=coords.get("pp", 0),
                device_ids=[rank],
                dp_group=coords.get("dp_replicate", 0),
            )
        )

    data_passes: list[DataPass] = []
    for event in comm_events:
        dim_name = rank_table.dim_by_group_name.get(event.group_name)
        groups = rank_table.process_groups.get(dim_name, []) if dim_name else []
        for group in groups:
            if len(group) < 2:
                continue
            slot = TensorSlot(
                name=f"{event.comm_primitive}_{event.event_id}",
                src_exit_op="",
                dst_entry_op="",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
            )
            for i, src_rank in enumerate(group):
                for dst_rank in group[i + 1 :]:
                    data_passes.append(
                        DataPass(
                            src_instance=f"rank{src_rank}",
                            dst_instance=f"rank{dst_rank}",
                            slots=[slot],
                            src_device=src_rank,
                            dst_device=dst_rank,
                            requires_communication=True,
                            comm_primitive=event.comm_primitive,
                        )
                    )

    dp_degree = rank_table.dim_degrees.get("dp_replicate", 1) * rank_table.dim_degrees.get(
        "fsdp", rank_table.dim_degrees.get("dp_shard", 1)
    )

    return ScheduleGraph(
        schedule_id=uuid.uuid4().hex[:12],
        workload_type="train",
        step_templates=step_templates,
        instances=instances,
        data_passes=data_passes,
        dp_degree=dp_degree,
        tp_degree=rank_table.dim_degrees.get("tp", 1),
        pp_degree=rank_table.dim_degrees.get("pp", 1),
        num_micro_batches=num_micro_batches,
        pipeline_schedule=pipeline_schedule,
        gradient_accumulation=gradient_accumulation,
        annotations={"rank_table": rank_table.to_dict()},
    )
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_schedule_builder.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_graph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.rank_table import RankTable


def _rank_table() -> RankTable:
    return RankTable(
        world_size=4,
        dim_degrees={"ep": 2, "tp": 1, "pp": 1, "dp_replicate": 1, "fsdp": 2},
        rank_coordinates={
            0: {"ep": 0, "pp": 0, "dp_replicate": 0},
            1: {"ep": 1, "pp": 0, "dp_replicate": 0},
            2: {"ep": 0, "pp": 0, "dp_replicate": 0},
            3: {"ep": 1, "pp": 0, "dp_replicate": 0},
        },
        process_groups={"ep": [[0, 1], [2, 3]], "fsdp": [[0, 2], [1, 3]]},
        dim_by_group_name={"grp_ep": "ep", "grp_fsdp": "fsdp"},
    )


def test_build_schedule_graph_creates_one_instance_per_rank():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert len(graph.instances) == 4
    assert {i.instance_id for i in graph.instances} == {"rank0", "rank1", "rank2", "rank3"}
    assert all(i.step_ref == "tmpl" for i in graph.instances)


def test_build_schedule_graph_expands_comm_event_into_data_passes():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    event = CommEvent(
        event_id="c1", comm_primitive="all_to_all", group_name="grp_ep",
        world_size=2, tensor_shape=(8, 16), dtype="bfloat16", volume_bytes=256,
    )
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[event],
    )
    # ep groups are [0,1] and [2,3]: exactly one pass per group (single pair each)
    assert len(graph.data_passes) == 2
    pass_pairs = {(p.src_instance, p.dst_instance) for p in graph.data_passes}
    assert pass_pairs == {("rank0", "rank1"), ("rank2", "rank3")}
    assert all(p.comm_primitive == "all_to_all" for p in graph.data_passes)
    assert all(p.requires_communication for p in graph.data_passes)


def test_build_schedule_graph_ignores_comm_event_with_unknown_group():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    event = CommEvent(
        event_id="c2", comm_primitive="allreduce", group_name="totally_unrecognized_group",
        world_size=4, tensor_shape=(4,), dtype="float32", volume_bytes=16,
    )
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[event],
    )
    assert graph.data_passes == []


def test_build_schedule_graph_carries_rank_table_in_annotations():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert graph.annotations["rank_table"]["world_size"] == 4


def test_build_schedule_graph_degrees_from_rank_table():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    graph = build_schedule_graph(
        step_templates={"tmpl": template}, rank_table=_rank_table(), comm_events=[],
    )
    assert graph.pp_degree == 1
    assert graph.tp_degree == 1
    assert graph.dp_degree == 2  # dp_replicate(1) * fsdp(2)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_schedule_builder.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.schedule_builder'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_schedule_builder.py`
Expected: `5 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/schedule_builder.py tests/unit_tests/simulator/capture/test_schedule_builder.py
git commit -m "feat(simulator): add L2 ScheduleGraph builder (StepInstance x N + DataPass)"
git push origin master
```

---

### Task 13: L3 WorkloadGraph builder

**Files:**
- Create: `torchtitan_npu/simulator/capture/workload_builder.py`
- Test: `tests/unit_tests/simulator/capture/test_workload_builder.py`

**Interfaces:**
- Consumes: `ScheduleGraph`, `StepGraph` (Tasks 2-3); `DataFlow`/`IterationSpec`/`WorkloadGraph` (Task 4).
- Produces: `build_workload_graph(*, schedule_graph: ScheduleGraph, step_templates: dict[str, StepGraph], local_batch_size: int, seq_len: int, num_micro_batches: int = 1) -> WorkloadGraph`.

- [ ] **Step 1: Write `workload_builder.py`**

```python
# torchtitan_npu/simulator/capture/workload_builder.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wraps a captured L2 ScheduleGraph into the top-level L3 WorkloadGraph:
iteration semantics + dataloader-derived data flow. See design doc §5.7 and
spec/L3-WorkloadGraph.md."""

from __future__ import annotations

import uuid

from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph


def build_workload_graph(
    *,
    schedule_graph: ScheduleGraph,
    step_templates: dict[str, StepGraph],
    local_batch_size: int,
    seq_len: int,
    num_micro_batches: int = 1,
) -> WorkloadGraph:
    """One captured training step, wrapped as a single-iteration
    `WorkloadGraph`. `num_iterations` is always 1 -- the simulator captures
    exactly one train step, per design doc §1."""
    input_flow = DataFlow(
        source="dataloader",
        tensor_shape=(local_batch_size, seq_len),
        dtype="int64",
        volume_per_iter=local_batch_size * seq_len * 8,  # int64 = 8 bytes/token
        is_streaming=True,
        interleave_strategy="synced",
    )
    output_flow = DataFlow(
        source="labels",
        tensor_shape=(local_batch_size, seq_len),
        dtype="int64",
        volume_per_iter=local_batch_size * seq_len * 8,
        is_streaming=True,
        interleave_strategy="synced",
    )

    iteration = IterationSpec(schedule=schedule_graph, microbatch_count=num_micro_batches)

    return WorkloadGraph(
        workload_id=uuid.uuid4().hex[:12],
        workload_type="train",
        step_templates=step_templates,
        iteration=iteration,
        num_iterations=1,
        warmup_iterations=0,
        data_inputs=[input_flow],
        data_outputs=[output_flow],
        cross_iter_passes=[],
    )
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/capture/test_workload_builder.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.workload_builder import build_workload_graph
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def test_build_workload_graph_wraps_single_iteration():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template}, local_batch_size=2, seq_len=4096,
    )
    assert graph.num_iterations == 1
    assert graph.warmup_iterations == 0
    assert graph.iteration.schedule is schedule
    assert graph.iteration.microbatch_count == 1


def test_build_workload_graph_data_flow_shapes_and_volume():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template}, local_batch_size=2, seq_len=4096,
    )
    assert graph.data_inputs[0].tensor_shape == (2, 4096)
    assert graph.data_inputs[0].volume_per_iter == 2 * 4096 * 8
    assert graph.data_inputs[0].is_streaming is True
    assert graph.data_outputs[0].source == "labels"


def test_build_workload_graph_respects_microbatch_count():
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    graph = build_workload_graph(
        schedule_graph=schedule, step_templates={"tmpl": template},
        local_batch_size=1, seq_len=128, num_micro_batches=4,
    )
    assert graph.iteration.microbatch_count == 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_workload_builder.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.capture.workload_builder'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/capture/test_workload_builder.py`
Expected: `3 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/capture/workload_builder.py tests/unit_tests/simulator/capture/test_workload_builder.py
git commit -m "feat(simulator): add L3 WorkloadGraph builder"
git push origin master
```

---

### Task 14: JSON + DOT + text-summary exporters

**Files:**
- Create: `torchtitan_npu/simulator/viz/__init__.py`
- Create: `torchtitan_npu/simulator/viz/json_export.py`
- Create: `torchtitan_npu/simulator/viz/dot_export.py`
- Create: `torchtitan_npu/simulator/viz/text_summary.py`
- Test: `tests/unit_tests/simulator/viz/__init__.py`
- Test: `tests/unit_tests/simulator/viz/test_json_export.py`
- Test: `tests/unit_tests/simulator/viz/test_dot_export.py`
- Test: `tests/unit_tests/simulator/viz/test_text_summary.py`

**Interfaces:**
- Consumes: `WorkloadGraph` (Task 4).
- Produces: `export_json(workload_graph: WorkloadGraph, path: str) -> None`; `export_dot(workload_graph: WorkloadGraph, path: str) -> None`; `export_text_summary(workload_graph: WorkloadGraph) -> str` (also usable via `write_text_summary(workload_graph, path)`).

- [ ] **Step 1: Create the viz package**

```python
# torchtitan_npu/simulator/viz/__init__.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exporters for the captured four-layer IR: JSON, Graphviz DOT, plain-text
summary, and self-contained HTML (Task 15)."""
```

```python
# tests/unit_tests/simulator/viz/__init__.py
```

- [ ] **Step 2: Write `json_export.py`**

```python
# torchtitan_npu/simulator/viz/json_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serializes a WorkloadGraph (and everything it recursively contains) to
JSON. `dataclasses.asdict()` recursively converts every nested dataclass
(including dataclasses stored as dict/list values, e.g. `StepGraph.nodes:
dict[str, OpNode]`) into plain dicts/lists, which `json.dumps` can then
serialize directly (tuples become JSON arrays)."""

from __future__ import annotations

import dataclasses
import json

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def workload_graph_to_dict(workload_graph: WorkloadGraph) -> dict:
    return dataclasses.asdict(workload_graph)


def export_json(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workload_graph_to_dict(workload_graph), f, indent=2, ensure_ascii=False)
```

- [ ] **Step 3: Write `dot_export.py`**

```python
# torchtitan_npu/simulator/viz/dot_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Renders every L1 StepGraph template's L0 operator DAG as Graphviz DOT.
Nodes are colored by op_type category (compute=lightblue, communication
=gold, data-move/memory=plum), matching the color scheme convention already
used by comparable trace tooling in this ecosystem (see design doc §5.9)."""

from __future__ import annotations

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph

_COMM_OP_TYPES = {"allreduce", "allgather", "reduce_scatter", "all_to_all"}
_DATA_MOVE_OP_TYPES = {"moe_token_permute", "moe_token_unpermute", "moe_re_routing", "view", "reshape", "transpose", "cat", "split"}


def _node_color(op_type: str) -> str:
    if op_type in _COMM_OP_TYPES:
        return "gold"
    if op_type in _DATA_MOVE_OP_TYPES:
        return "plum"
    return "lightblue"


def export_dot(workload_graph: WorkloadGraph, path: str) -> None:
    lines = ["digraph ComputeGraph {", '  rankdir="LR";']
    for step_id, step_graph in workload_graph.step_templates.items():
        lines.append(f'  subgraph "cluster_{step_id}" {{')
        lines.append(f'    label="{step_graph.step_type}";')
        for op_id, node in step_graph.nodes.items():
            label = f"{node.op_type}"
            if node.annotations.get("repeat_count", 1) > 1:
                label += f" (x{node.annotations['repeat_count']})"
            lines.append(f'    "{op_id}" [label="{label}", style=filled, fillcolor={_node_color(node.op_type)}];')
        for op_id, node in step_graph.nodes.items():
            for succ in node.successors:
                lines.append(f'    "{op_id}" -> "{succ}";')
        lines.append("  }")
    lines.append("}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Write `text_summary.py`**

```python
# torchtitan_npu/simulator/viz/text_summary.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Human-readable plain-text summary of a captured WorkloadGraph: op
counts per step, FLOPs/memory/communication totals, and an explicit list
of "unrecognized" op types (never silently hidden -- see design doc §5.8
and §9's note about the sibling project's MockCostModel coverage gap)."""

from __future__ import annotations

from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph


def export_text_summary(workload_graph: WorkloadGraph) -> str:
    lines: list[str] = []
    lines.append(f"Workload: {workload_graph.workload_id} ({workload_graph.workload_type})")
    lines.append(f"Iterations: {workload_graph.num_iterations} (warmup={workload_graph.warmup_iterations})")
    lines.append("")

    unknown_op_types: set[str] = set()
    for step_id, step_graph in workload_graph.step_templates.items():
        total_flops = sum(node.flops for node in step_graph.nodes.values())
        total_comm_bytes = sum(node.comm_bytes for node in step_graph.nodes.values())
        total_peak_mem = sum(node.peak_mem for node in step_graph.nodes.values())
        lines.append(f"[{step_graph.step_type}] step={step_id} nodes={len(step_graph.nodes)}")
        lines.append(f"  total_flops={total_flops}  total_peak_mem_bytes={total_peak_mem}  total_comm_bytes={total_comm_bytes}")
        lines.append(f"  is_acyclic={step_graph.is_acyclic}")
        for node in step_graph.nodes.values():
            if node.annotations.get("cost_unknown"):
                unknown_op_types.add(node.op_type if node.op_type != "unknown" else node.annotations.get("raw_op_type", "unknown"))
        lines.append("")

    schedule = workload_graph.iteration.schedule
    lines.append(f"Schedule: {len(schedule.instances)} instances, {len(schedule.data_passes)} data passes")
    lines.append(
        f"  dp_degree={schedule.dp_degree} tp_degree={schedule.tp_degree} pp_degree={schedule.pp_degree} "
        f"pipeline_schedule={schedule.pipeline_schedule}"
    )
    comm_bytes_by_primitive: dict[str, int] = {}
    for data_pass in schedule.data_passes:
        volume = sum(slot.volume_bytes for slot in data_pass.slots)
        comm_bytes_by_primitive[data_pass.comm_primitive] = comm_bytes_by_primitive.get(data_pass.comm_primitive, 0) + volume
    for primitive, total_bytes in sorted(comm_bytes_by_primitive.items()):
        lines.append(f"  comm[{primitive}] total_bytes={total_bytes}")
    lines.append("")

    if unknown_op_types:
        lines.append(f"Unrecognized op types ({len(unknown_op_types)}) -- cost not estimated for these:")
        for op_type in sorted(unknown_op_types):
            lines.append(f"  - {op_type}")
    else:
        lines.append("All captured op types were recognized by the cost model.")

    return "\n".join(lines) + "\n"


def write_text_summary(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(export_text_summary(workload_graph))
```

- [ ] **Step 5: Write the failing tests**

```python
# tests/unit_tests/simulator/viz/test_json_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.json_export import export_json, workload_graph_to_dict


def _tiny_workload() -> WorkloadGraph:
    node = OpNode(op_id="op1", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"op1": node})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    return WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )


def test_workload_graph_to_dict_is_plain_dict():
    d = workload_graph_to_dict(_tiny_workload())
    assert isinstance(d, dict)
    assert d["workload_id"] == "wl1"
    assert d["step_templates"]["tmpl"]["nodes"]["op1"]["op_type"] == "matmul"


def test_export_json_writes_valid_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        export_json(_tiny_workload(), path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["workload_id"] == "wl1"
```

```python
# tests/unit_tests/simulator/viz/test_dot_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.dot_export import export_dot


def test_export_dot_writes_valid_digraph_with_edges():
    node_a = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=["b"])
    node_b = OpNode(op_id="b", op_type="allreduce", inputs=[], outputs=[], attrs={}, predecessors=["a"], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": node_a, "b": node_b})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.dot")
        export_dot(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert content.startswith("digraph ComputeGraph {")
    assert '"a" -> "b"' in content
    assert "fillcolor=gold" in content  # allreduce is a comm op
    assert "fillcolor=lightblue" in content  # matmul is a compute op
```

```python
# tests/unit_tests/simulator/viz/test_text_summary.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph, StepInstance
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.text_summary import export_text_summary


def test_export_text_summary_reports_flops_and_unknown_ops():
    known = OpNode(
        op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[], flops=1000,
    )
    unknown = OpNode(
        op_id="b", op_type="unknown", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[],
        annotations={"cost_unknown": True, "raw_op_type": "aten.mystery_op.default"},
    )
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": known, "b": unknown})
    instance = StepInstance(
        instance_id="rank0", step_ref="tmpl", step_type="forward", micro_batch_idx=0,
        pipeline_stage=0, device_ids=[0], dp_group=0,
    )
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[instance],
    )
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    summary = export_text_summary(workload)
    assert "total_flops=1000" in summary
    assert "aten.mystery_op.default" in summary
    assert "Unrecognized op types" in summary


def test_export_text_summary_reports_no_unknown_ops_when_all_recognized():
    known = OpNode(op_id="a", op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    template = StepGraph(step_id="tmpl", step_type="forward", nodes={"a": known})
    schedule = ScheduleGraph(schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[])
    workload = WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
    )
    summary = export_text_summary(workload)
    assert "All captured op types were recognized" in summary
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/viz/`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.viz'`

- [ ] **Step 7: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/viz/`
Expected: `5 passed`

- [ ] **Step 8: Commit and push**

```bash
git add torchtitan_npu/simulator/viz/__init__.py torchtitan_npu/simulator/viz/json_export.py \
        torchtitan_npu/simulator/viz/dot_export.py torchtitan_npu/simulator/viz/text_summary.py \
        tests/unit_tests/simulator/viz/
git commit -m "feat(simulator): add JSON/DOT/text-summary exporters"
git push origin master
```

---

### Task 15: Self-contained HTML exporter with layer folding

**Files:**
- Create: `torchtitan_npu/simulator/viz/html_export.py`
- Test: `tests/unit_tests/simulator/viz/test_html_export.py`

**Interfaces:**
- Consumes: `WorkloadGraph` (Task 4).
- Produces: `export_html(workload_graph: WorkloadGraph, path: str) -> None`; `normalize_module_path(path: str) -> str` (helper, exported for testing).

Uses native HTML5 `<details>`/`<summary>` for expand/collapse (no external
JS framework, no CDN -- keeps the file self-contained and viewable offline,
matching design doc §5.9). Repeated layers (e.g. 61 `TransformerBlock`s)
are grouped by *normalized* module path (numeric `ModuleList` indices
replaced with `N`) so the L0 DAG view shows one representative layer plus
an occurrence count instead of rendering all 61 in full.

- [ ] **Step 1: Write `html_export.py`**

```python
# torchtitan_npu/simulator/viz/html_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained HTML visualization of the captured four-layer IR: L3
workload card, L2 RankTable + schedule summary, L1 step cards, and a
foldable L0 operator listing per step (see design doc §5.9)."""

from __future__ import annotations

import html
import re

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph

_NUMERIC_SEGMENT = re.compile(r"\.\d+(?=\.|$)")


def normalize_module_path(path: str) -> str:
    """Replace numeric `ModuleList` indices with `N` so repeated layers
    (e.g. `layers.0.attention`, `layers.1.attention`, ...) collapse to the
    same group key (`layers.N.attention`)."""
    if not path:
        return path
    return _NUMERIC_SEGMENT.sub(".N", path)


def _group_nodes_by_normalized_path(nodes: dict[str, OpNode]) -> dict[str, list[tuple[str, OpNode]]]:
    groups: dict[str, list[tuple[str, OpNode]]] = {}
    for op_id, node in nodes.items():
        raw_path = node.annotations.get("module_path", "")
        key = normalize_module_path(raw_path) or "(root)"
        groups.setdefault(key, []).append((op_id, node))
    return groups


def _distinct_raw_paths(entries: list[tuple[str, OpNode]]) -> set[str]:
    return {node.annotations.get("module_path", "") for _, node in entries}


def _render_op_row(op_id: str, node: OpNode) -> str:
    repeat = node.annotations.get("repeat_count", 1)
    repeat_suffix = f" (dedup x{repeat})" if repeat > 1 else ""
    unknown_suffix = " [cost unknown]" if node.annotations.get("cost_unknown") else ""
    return (
        f"<li><code>{html.escape(op_id)}</code> "
        f"<strong>{html.escape(node.op_type)}</strong>{repeat_suffix}{unknown_suffix} "
        f"flops={node.flops} peak_mem={node.peak_mem} comm_bytes={node.comm_bytes}</li>"
    )


def _render_step_graph_section(step_graph: StepGraph) -> str:
    groups = _group_nodes_by_normalized_path(step_graph.nodes)
    parts = [
        f"<h3>{html.escape(step_graph.step_type)} "
        f"(step_id={html.escape(step_graph.step_id)}, {len(step_graph.nodes)} ops, "
        f"is_acyclic={step_graph.is_acyclic})</h3>"
    ]
    for group_key in sorted(groups):
        entries = groups[group_key]
        distinct_paths = _distinct_raw_paths(entries)
        occurrence_count = max(len(distinct_paths), 1)
        summary_label = html.escape(group_key)
        if occurrence_count > 1:
            summary_label += f" &times; {occurrence_count} layers"
        # Render only the first occurrence's ops in full when the group is
        # a repeated layer; always render every op for non-repeated groups.
        if occurrence_count > 1:
            first_path = sorted(distinct_paths)[0]
            rows = [_render_op_row(op_id, node) for op_id, node in entries if node.annotations.get("module_path") == first_path]
        else:
            rows = [_render_op_row(op_id, node) for op_id, node in entries]
        parts.append(
            f"<details><summary>{summary_label} ({len(rows)} ops shown"
            f"{' for representative layer' if occurrence_count > 1 else ''})</summary>"
            f"<ul>{''.join(rows)}</ul></details>"
        )
    return "\n".join(parts)


def _render_rank_table_section(workload_graph: WorkloadGraph) -> str:
    schedule = workload_graph.iteration.schedule
    rank_table = schedule.annotations.get("rank_table", {}) if schedule.annotations else {}
    dim_degrees = rank_table.get("dim_degrees", {})
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(dim_degrees.items()))
    return (
        "<h2>L2: RankTable / Schedule</h2>"
        f"<p>world_size={rank_table.get('world_size', '?')}, "
        f"instances={len(schedule.instances)}, data_passes={len(schedule.data_passes)}, "
        f"dp_degree={schedule.dp_degree}, tp_degree={schedule.tp_degree}, pp_degree={schedule.pp_degree}, "
        f"pipeline_schedule={html.escape(schedule.pipeline_schedule)}</p>"
        f"<table border='1'><tr><th>dimension</th><th>degree</th></tr>{rows}</table>"
    )


def _render_workload_section(workload_graph: WorkloadGraph) -> str:
    inputs = "".join(
        f"<li>{html.escape(f.source)}: shape={f.tensor_shape} dtype={f.dtype} "
        f"volume_per_iter={f.volume_per_iter}</li>"
        for f in workload_graph.data_inputs
    )
    return (
        "<h2>L3: WorkloadGraph</h2>"
        f"<p>workload_id={html.escape(workload_graph.workload_id)} "
        f"type={html.escape(workload_graph.workload_type)} "
        f"num_iterations={workload_graph.num_iterations}</p>"
        f"<ul>{inputs}</ul>"
    )


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>torchtitan_npu simulator trace</title>
<style>
body {{ font-family: monospace; margin: 2em; }}
table {{ border-collapse: collapse; margin-bottom: 1em; }}
td, th {{ padding: 4px 8px; }}
details {{ margin: 4px 0; }}
summary {{ cursor: pointer; font-weight: bold; }}
</style>
</head>
<body>
<h1>torchtitan_npu Simulator Trace</h1>
{workload_section}
{rank_table_section}
<h2>L1/L0: Step Graphs</h2>
{step_sections}
</body>
</html>
"""


def render_html(workload_graph: WorkloadGraph) -> str:
    step_sections = "\n".join(_render_step_graph_section(sg) for sg in workload_graph.step_templates.values())
    return _PAGE_TEMPLATE.format(
        workload_section=_render_workload_section(workload_graph),
        rank_table_section=_render_rank_table_section(workload_graph),
        step_sections=step_sections,
    )


def export_html(workload_graph: WorkloadGraph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(workload_graph))
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/viz/test_html_export.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.schedule_graph import ScheduleGraph
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.ir.workload_graph import DataFlow, IterationSpec, WorkloadGraph
from torchtitan_npu.simulator.viz.html_export import export_html, normalize_module_path, render_html


def test_normalize_module_path_strips_numeric_modulelist_indices():
    assert normalize_module_path("layers.0.attention.wq") == "layers.N.attention.wq"
    assert normalize_module_path("layers.60.mlp.w1") == "layers.N.mlp.w1"
    assert normalize_module_path("gate") == "gate"
    assert normalize_module_path("") == ""


def _workload_with_repeated_layers(num_layers: int) -> WorkloadGraph:
    nodes: dict[str, OpNode] = {}
    for layer_idx in range(num_layers):
        op_id = f"op_{layer_idx}"
        nodes[op_id] = OpNode(
            op_id=op_id, op_type="matmul", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[],
            annotations={"module_path": f"layers.{layer_idx}.attention.wq"},
        )
    template = StepGraph(step_id="tmpl", step_type="forward", nodes=nodes)
    schedule = ScheduleGraph(
        schedule_id="sched", workload_type="train", step_templates={"tmpl": template}, instances=[],
        annotations={"rank_table": {"world_size": 384, "dim_degrees": {"ep": 192}}},
    )
    return WorkloadGraph(
        workload_id="wl1", workload_type="train", step_templates={"tmpl": template},
        iteration=IterationSpec(schedule=schedule, microbatch_count=1), num_iterations=1,
        data_inputs=[DataFlow(source="dataloader", tensor_shape=(1, 4096), dtype="int64", volume_per_iter=32768)],
    )


def test_render_html_folds_repeated_layers_into_one_group():
    workload = _workload_with_repeated_layers(61)
    page = render_html(workload)
    assert "layers.N.attention.wq" in page
    assert "&times; 61 layers" in page
    # only the representative layer's op should actually be listed
    assert page.count("<li><code>op_") == 1


def test_render_html_includes_rank_table_world_size():
    workload = _workload_with_repeated_layers(2)
    page = render_html(workload)
    assert "world_size=384" in page


def test_export_html_writes_a_file():
    workload = _workload_with_repeated_layers(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.html")
        export_html(workload, path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
    assert content.startswith("<!DOCTYPE html>")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/viz/test_html_export.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.viz.html_export'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/viz/test_html_export.py`
Expected: `4 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/viz/html_export.py tests/unit_tests/simulator/viz/test_html_export.py
git commit -m "feat(simulator): add self-contained HTML exporter with repeated-layer folding"
git push origin master
```

---

### Task 16: `meta_env.py` -- patch device_type to meta

**Files:**
- Create: `torchtitan_npu/simulator/meta_env.py`
- Test: `tests/unit_tests/simulator/test_meta_env.py`

**Interfaces:**
- Produces: `patch_device_type_to_meta() -> None`; `unpatch_device_type_to_meta() -> None`.

This task's approach is verified against the exact pinned torchtitan commit
(`ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`, see design doc §5.1):
`torch.device("meta:0")` behaves identically to `torch.device("meta")`, and
both `Module.to_empty(device="meta:0")` and `nn.init.*` functions run
without error on meta tensors -- so `Trainer.__init__` needs **zero** code
changes once `device_type`/`device_module` are patched to `"meta"` in the
three modules that import them by value at load time.

- [ ] **Step 1: Write `meta_env.py`**

```python
# torchtitan_npu/simulator/meta_env.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forces torchtitan's device_type/device_module globals to `"meta"` so
`Trainer.__init__` builds, materializes (`to_empty`), and initializes
(`init_weights`) its model entirely on the meta device -- no real memory is
ever allocated. See design doc §5.1 for the verification this relies on."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any


class _MetaDeviceModule:
    """Minimal stand-in for `torch.cuda`/`torch_npu`, covering every method
    actually called on `device_module` by `torchtitan.trainer`,
    `torchtitan.components.metrics`, and `torchtitan.distributed.utils`
    (verified against the pinned torchtitan commit -- see design doc §5.1)."""

    name = "Meta_Simulator"

    def set_device(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def reset_peak_memory_stats(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def get_device_name(self, *_args: Any, **_kwargs: Any) -> str:
        return self.name

    def get_device_properties(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(total_memory=0, name=self.name)

    def memory_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {
            "active_bytes.all.peak": 0,
            "reserved_bytes.all.peak": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }


_PATCHED_MODULE_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchtitan.tools.utils", ("device_type", "device_module")),
    ("torchtitan.components.metrics", ("device_type", "device_module")),
    ("torchtitan.distributed.parallel_dims", ("device_type",)),
    ("torchtitan.distributed.utils", ("device_type", "device_module")),
)

_original_values: dict[tuple[str, str], Any] = {}
_patched = False


def patch_device_type_to_meta() -> None:
    """Idempotently rebind `device_type="meta"` / `device_module=<stub>`
    across every module that imported them by value at load time."""
    global _patched
    if _patched:
        return

    stub = _MetaDeviceModule()
    for module_path, attr_names in _PATCHED_MODULE_ATTRS:
        module = importlib.import_module(module_path)
        for attr_name in attr_names:
            if not hasattr(module, attr_name):
                continue
            _original_values[(module_path, attr_name)] = getattr(module, attr_name)
            value: Any = "meta" if attr_name == "device_type" else stub
            setattr(module, attr_name, value)
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched
    for (module_path, attr_name), original in _original_values.items():
        module = importlib.import_module(module_path)
        setattr(module, attr_name, original)
    _original_values.clear()
    _patched = False
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/test_meta_env.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta, unpatch_device_type_to_meta


def test_patch_device_type_to_meta_rebinds_all_dependent_modules():
    import torchtitan.components.metrics as metrics_mod
    import torchtitan.distributed.parallel_dims as parallel_dims_mod
    import torchtitan.distributed.utils as dist_utils_mod
    import torchtitan.tools.utils as utils_mod

    try:
        patch_device_type_to_meta()
        assert utils_mod.device_type == "meta"
        assert metrics_mod.device_type == "meta"
        assert parallel_dims_mod.device_type == "meta"
        assert dist_utils_mod.device_type == "meta"
        assert utils_mod.device_module.get_device_name() == "Meta_Simulator"
    finally:
        unpatch_device_type_to_meta()


def test_unpatch_restores_original_device_type():
    import torchtitan.tools.utils as utils_mod

    original = utils_mod.device_type
    patch_device_type_to_meta()
    unpatch_device_type_to_meta()
    assert utils_mod.device_type == original


def test_patch_is_idempotent():
    import torchtitan.tools.utils as utils_mod

    try:
        patch_device_type_to_meta()
        patch_device_type_to_meta()  # must not raise, must not double-save originals
        assert utils_mod.device_type == "meta"
    finally:
        unpatch_device_type_to_meta()


def test_stub_device_module_methods_used_by_trainer_and_metrics_do_not_raise():
    try:
        patch_device_type_to_meta()
        import torchtitan.tools.utils as utils_mod

        stub = utils_mod.device_module
        stub.set_device(torch.device("meta:0"))
        assert stub.current_device() == 0
        assert stub.device_count() == 1
        stub.synchronize()
        stub.empty_cache()
        stub.reset_peak_memory_stats()
        props = stub.get_device_properties(torch.device("meta:0"))
        assert props.total_memory == 0
        stats = stub.memory_stats(torch.device("meta:0"))
        assert stats["active_bytes.all.peak"] == 0
    finally:
        unpatch_device_type_to_meta()


def test_meta_device_materialization_pattern_used_by_trainer_init_weights():
    # Mirrors Trainer.__init__'s `model.to_empty(device=init_device)` +
    # `nn.init.*` calls (trainer.py:407-411 in the pinned commit) -- this
    # must never raise once device_type/device_module are patched to meta.
    module = nn.Linear(4, 8, device="meta")
    module.to_empty(device="meta:0")
    with torch.no_grad():
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
    assert module.weight.device.type == "meta"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_meta_env.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.meta_env'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_meta_env.py`
Expected: `5 passed`. This test genuinely exercises `torchtitan.tools.utils`/`torchtitan.components.metrics`/`torchtitan.distributed.parallel_dims`/`torchtitan.distributed.utils` -- no `torch_npu` needed (verified directly in this sandbox during design).

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/meta_env.py tests/unit_tests/simulator/test_meta_env.py
git commit -m "feat(simulator): add meta-device patching for zero-memory model construction"
git push origin master
```

---

### Task 17: `moe_force_balance.py` -- force deterministic MoE routing and seed

**Files:**
- Create: `torchtitan_npu/simulator/moe_force_balance.py`
- Test: `tests/unit_tests/simulator/test_moe_force_balance.py`

**Interfaces:**
- Produces: `force_moe_load_balance(config: Any) -> None`; `force_deterministic_seed(config: Any, seed: int = 42) -> None`.

`force_deterministic_seed` fixes a second concrete crash found by reading
the pinned `torchtitan.distributed.utils.set_determinism()`: when
`world_size > 1` and `debug.seed is None`, it executes
`seed_tensor.to("cpu").view(torch.uint64).item()` on a device tensor,
which raises `NotImplementedError: Cannot copy out of meta tensor; no
data!` under meta-device execution (reproduced during design, see design
doc §9). Providing an explicit seed short-circuits that code path entirely.

- [ ] **Step 1: Write `moe_force_balance.py`**

```python
# torchtitan_npu/simulator/moe_force_balance.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forces deterministic (data-independent) behavior needed for meta-device
capture: MoE round-robin load balancing (reusing the existing
`debug.moe_force_load_balance` flag already wired through
`torchtitan_npu.models.deepseek_v4.moe.TokenChoiceTopKRouter`) and a fixed
RNG seed (see design doc §5.3 and §9)."""

from __future__ import annotations

import warnings
from typing import Any

DEFAULT_SIMULATION_SEED = 42


def force_moe_load_balance(config: Any) -> None:
    """Force `config.debug.moe_force_load_balance = True`, warning if the
    caller's config had it disabled. The simulator always needs
    deterministic MoE routing so the captured compute graph (in particular
    `num_tokens_per_expert`, and therefore the EP all-to-all split sizes)
    does not depend on real token data -- see design doc §5.3 for the proof
    that this makes every rank's routing decision identical."""
    debug_config = config.debug
    if not getattr(debug_config, "moe_force_load_balance", False):
        warnings.warn(
            "torchtitan_npu.simulator: forcing debug.moe_force_load_balance=True "
            "(config had it disabled). The simulator always uses deterministic "
            "round-robin MoE routing so the captured compute graph does not "
            "depend on real token data.",
            stacklevel=2,
        )
    debug_config.moe_force_load_balance = True


def force_deterministic_seed(config: Any, seed: int = DEFAULT_SIMULATION_SEED) -> None:
    """Force a fixed `config.debug.seed` if the caller left it unset.

    `torchtitan.distributed.utils.set_determinism()` broadcasts a seed
    derived from `torch.get_rng_state()` whenever `world_size > 1` and
    `debug.seed is None`; that code path calls `.to("cpu")` on a tensor
    living on the trainer's device and then `.item()`, which raises
    `NotImplementedError: Cannot copy out of meta tensor; no data!` under
    the simulator's meta-device execution. Supplying an explicit seed
    means `set_determinism()` takes the "already have a seed" branch and
    never touches that code path.
    """
    if config.debug.seed is None:
        config.debug.seed = seed
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/test_moe_force_balance.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan_npu.simulator.moe_force_balance import (
    DEFAULT_SIMULATION_SEED,
    force_deterministic_seed,
    force_moe_load_balance,
)


def _config(moe_force_load_balance: bool = False, seed=None) -> SimpleNamespace:
    return SimpleNamespace(debug=SimpleNamespace(moe_force_load_balance=moe_force_load_balance, seed=seed))


def test_force_moe_load_balance_sets_true_and_warns_when_disabled():
    config = _config(moe_force_load_balance=False)
    with pytest.warns(UserWarning, match="moe_force_load_balance"):
        force_moe_load_balance(config)
    assert config.debug.moe_force_load_balance is True


def test_force_moe_load_balance_no_warning_when_already_enabled():
    config = _config(moe_force_load_balance=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        force_moe_load_balance(config)  # must not raise/warn
    assert config.debug.moe_force_load_balance is True


def test_force_deterministic_seed_sets_default_when_none():
    config = _config(seed=None)
    force_deterministic_seed(config)
    assert config.debug.seed == DEFAULT_SIMULATION_SEED


def test_force_deterministic_seed_respects_existing_seed():
    config = _config(seed=123)
    force_deterministic_seed(config)
    assert config.debug.seed == 123


def test_force_deterministic_seed_accepts_custom_seed():
    config = _config(seed=None)
    force_deterministic_seed(config, seed=7)
    assert config.debug.seed == 7
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_moe_force_balance.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.moe_force_balance'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_moe_force_balance.py`
Expected: `5 passed`

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/moe_force_balance.py tests/unit_tests/simulator/test_moe_force_balance.py
git commit -m "feat(simulator): force deterministic MoE routing and RNG seed for meta-device capture"
git push origin master
```

---

### Task 18: `SimulationTrainer` -- wire everything together

**Files:**
- Create: `torchtitan_npu/simulator/trainer.py`
- Test: `tests/unit_tests/simulator/test_trainer.py`

**Interfaces:**
- Consumes: everything from Tasks 5-17 (`OpDispatchCapture`/`ModulePathTracker`/`StepBoundaryTracker` from capture; `capture_fake_collectives` from Task 10; `build_rank_table` from Task 11; `build_schedule_graph` from Task 12; `build_workload_graph` from Task 13; `export_json`/`export_dot`/`write_text_summary`/`export_html` from Tasks 14-15; `patch_device_type_to_meta` from Task 16; `force_moe_load_balance`/`force_deterministic_seed` from Task 17); `torchtitan.trainer.Trainer`.
- Produces: `SimulationConfig(output_dir: str = "./simulator_output", output_formats: list[str] = [...])`; `SimulationTrainerConfig(Trainer.Config)` with a `simulation: SimulationConfig` field; `run_simulation_step(*, model_parts, parallel_dims, forward_backward_step, input_dict, labels, optimizer_step, lr_scheduler_step, pipeline_schedule="none", num_micro_batches=1, gradient_accumulation=1, local_batch_size, seq_len) -> WorkloadGraph` (the core, Trainer-independent logic); `SimulationTrainer(Trainer)` with `Config = SimulationTrainerConfig`, overriding `__init__` and `train()`.

**Why `run_simulation_step` is factored out of `SimulationTrainer.train()`:**
constructing a real `Trainer` end-to-end (tokenizer, dataloader,
`parallelize_fn`, optimizer, ...) is exactly what Task 20 exercises against
the real DeepSeek-V4-Pro config inside the CANN container; it cannot run
in this sandbox (no `torch_npu`). Factoring the capture-and-build logic
into a standalone function that accepts plain callables/model parts lets
this task's tests exercise the *entire* capture-through-WorkloadGraph
pipeline with a tiny `nn.Linear` "model" and a real fake `ParallelDims`
mesh, fully runnably in this sandbox -- de-risking the wiring before Task
20's real-model run.

- [ ] **Step 1: Write `trainer.py`**

```python
# torchtitan_npu/simulator/trainer.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimulationTrainer: a Trainer subclass that captures one training step's
four-layer IR (L0-L3) instead of running a full multi-step training loop,
with zero real NPU hardware and zero real memory allocation. See
docs/superpowers/specs/2026-07-01-npu-simulator-design.md."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torchtitan.trainer import Trainer

from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.module_path import ModulePathTracker
from torchtitan_npu.simulator.capture.schedule_builder import build_schedule_graph
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker, build_step_graphs
from torchtitan_npu.simulator.capture.workload_builder import build_workload_graph
from torchtitan_npu.simulator.ir.workload_graph import WorkloadGraph
from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta
from torchtitan_npu.simulator.moe_force_balance import force_deterministic_seed, force_moe_load_balance
from torchtitan_npu.simulator.rank_table import build_rank_table
from torchtitan_npu.simulator.viz.dot_export import export_dot
from torchtitan_npu.simulator.viz.html_export import export_html
from torchtitan_npu.simulator.viz.json_export import export_json
from torchtitan_npu.simulator.viz.text_summary import write_text_summary


@dataclass(kw_only=True, slots=True)
class SimulationConfig:
    output_dir: str = "./simulator_output"
    output_formats: list[str] = field(default_factory=lambda: ["json", "dot", "text", "html"])


@dataclass(kw_only=True, slots=True)
class SimulationTrainerConfig(Trainer.Config):
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def run_simulation_step(
    *,
    model_parts: list[nn.Module],
    parallel_dims: Any,
    forward_backward_step: Callable[..., torch.Tensor],
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    optimizer_step: Callable[[], None],
    lr_scheduler_step: Callable[[], None],
    local_batch_size: int,
    seq_len: int,
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
) -> WorkloadGraph:
    """Run one forward+backward+optimizer step under full capture and
    return the resulting four-layer WorkloadGraph. Bypasses
    `Trainer.train_step()` deliberately: that method's `dist_sum`-based
    token counting and loss/grad-norm logging both call `.item()` on
    device tensors, which raises under meta-device execution (see design
    doc §9) -- `global_valid_tokens` is instead supplied here as a plain
    Python float derived from the static input shape.
    """
    global_valid_tokens = float(labels.numel())

    boundary = StepBoundaryTracker()
    module_path_tracker = ModulePathTracker(model_parts[0])
    capture = OpDispatchCapture(module_path_tracker=module_path_tracker, phase_provider=lambda: boundary.current_phase)

    with capture_fake_collectives() as comm_recorder, boundary, module_path_tracker, capture:
        boundary.mark("forward")
        forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            global_valid_tokens=global_valid_tokens,
        )
        boundary.mark("optimizer")
        optimizer_step()
        lr_scheduler_step()

    nodes = capture.build_nodes()
    step_templates = build_step_graphs(nodes)
    rank_table = build_rank_table(parallel_dims)
    schedule_graph = build_schedule_graph(
        step_templates=step_templates,
        rank_table=rank_table,
        comm_events=comm_recorder.events,
        pipeline_schedule=pipeline_schedule,
        num_micro_batches=num_micro_batches,
        gradient_accumulation=gradient_accumulation,
    )
    return build_workload_graph(
        schedule_graph=schedule_graph,
        step_templates=step_templates,
        local_batch_size=local_batch_size,
        seq_len=seq_len,
        num_micro_batches=num_micro_batches,
    )


class SimulationTrainer(Trainer):
    """Drop-in replacement for `Trainer` that captures the four-layer IR of
    one training step instead of training for `config.training.steps`
    steps. See design doc §6 for the end-to-end data flow.

    `Config = SimulationTrainerConfig` (an attribute assignment, not nested
    class syntax) is enough for `torchtitan.config.configurable.
    Configurable.__init_subclass__` to auto-wire `SimulationTrainerConfig.
    _owner = SimulationTrainer` -- verified directly against the pinned
    torchtitan commit during design: any name bound in a class body
    (including plain assignment) lands in that class's own `__dict__`,
    which is exactly what `__init_subclass__` checks for. This makes
    `some_simulation_trainer_config.build()` correctly return a
    `SimulationTrainer` instance (not a plain `Trainer`), the same pattern
    the sibling project's docs describe for their own `SimulationTrainer`.
    """

    Config = SimulationTrainerConfig

    def __init__(self, config: SimulationTrainerConfig) -> None:
        force_moe_load_balance(config)
        force_deterministic_seed(config)
        config.compile.enable = False  # tracing needs eager dispatch, not a compiled graph
        config.comm.mode = "fake_backend"

        patch_device_type_to_meta()
        super().__init__(config)
        self.simulation_config = config.simulation
        self.workload_graph: WorkloadGraph | None = None

    def train(self) -> None:
        data_iterator = iter(self.dataloader)
        input_dict, labels = next(data_iterator)
        for key, value in list(input_dict.items()):
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)
        labels = labels.to(self.device)

        self.workload_graph = run_simulation_step(
            model_parts=self.model_parts,
            parallel_dims=self.parallel_dims,
            forward_backward_step=lambda **kwargs: self.forward_backward_step(**kwargs),
            input_dict=input_dict,
            labels=labels,
            optimizer_step=self.optimizers.step,
            lr_scheduler_step=self.lr_schedulers.step,
            local_batch_size=self.config.training.local_batch_size,
            seq_len=self.config.training.seq_len,
            pipeline_schedule=self.config.parallelism.pipeline_parallel_schedule,
            num_micro_batches=self.gradient_accumulation_steps,
            gradient_accumulation=self.gradient_accumulation_steps,
        )
        self._export()

    def _export(self) -> None:
        assert self.workload_graph is not None
        out_dir = self.simulation_config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        formats = self.simulation_config.output_formats
        if "json" in formats:
            export_json(self.workload_graph, os.path.join(out_dir, "simulation_result.json"))
        if "dot" in formats:
            export_dot(self.workload_graph, os.path.join(out_dir, "compute_graph.dot"))
        if "text" in formats:
            write_text_summary(self.workload_graph, os.path.join(out_dir, "summary.txt"))
        if "html" in formats:
            export_html(self.workload_graph, os.path.join(out_dir, "trace.html"))
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit_tests/simulator/test_trainer.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan_npu.simulator.trainer import run_simulation_step


@pytest.fixture
def fake_world():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29913"
    dist.init_process_group("fake", rank=0, world_size=4)
    yield
    dist.destroy_process_group()


def _build_parallel_dims(world_size: int):
    from torchtitan.distributed.parallel_dims import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=world_size, cp=1, tp=1, pp=1, ep=1, world_size=world_size)
    parallel_dims.build_mesh()
    return parallel_dims


def test_run_simulation_step_produces_complete_workload_graph(fake_world):
    parallel_dims = _build_parallel_dims(4)

    model = nn.Linear(8, 8, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        pred = model(input_dict["input"])
        loss = pred.sum() / global_valid_tokens
        loss.backward()
        return loss

    input_dict = {"input": torch.randn(2, 8, device="meta")}
    labels = torch.randint(0, 10, (2, 8), device="meta")

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict=input_dict,
        labels=labels,
        optimizer_step=optimizer.step,
        lr_scheduler_step=lr_scheduler.step,
        local_batch_size=2,
        seq_len=8,
    )

    assert graph.num_iterations == 1
    assert "forward" in graph.step_templates
    assert "backward" in graph.step_templates
    schedule = graph.iteration.schedule
    assert len(schedule.instances) == 4  # world_size
    assert schedule.annotations["rank_table"]["world_size"] == 4


def test_run_simulation_step_captures_optimizer_phase(fake_world):
    parallel_dims = _build_parallel_dims(4)
    model = nn.Linear(4, 4, device="meta")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def forward_backward_step(*, input_dict, labels, global_valid_tokens):
        loss = model(input_dict["input"]).sum() / global_valid_tokens
        loss.backward()
        return loss

    graph = run_simulation_step(
        model_parts=[model],
        parallel_dims=parallel_dims,
        forward_backward_step=forward_backward_step,
        input_dict={"input": torch.randn(2, 4, device="meta")},
        labels=torch.randint(0, 10, (2, 4), device="meta"),
        optimizer_step=optimizer.step,
        lr_scheduler_step=lambda: None,
        local_batch_size=2,
        seq_len=4,
    )
    assert "optimizer" in graph.step_templates
    optimizer_ops = graph.step_templates["optimizer"].nodes
    assert len(optimizer_ops) > 0


def test_simulation_trainer_config_build_dispatches_to_simulation_trainer():
    # Regression test for the Configurable._owner auto-wiring mechanism
    # this design relies on (verified against the pinned torchtitan
    # source): `SimulationTrainerConfig().build()` must construct a
    # SimulationTrainer, not a plain Trainer, even though
    # `SimulationTrainer.Config = SimulationTrainerConfig` uses simple
    # attribute assignment rather than nested `class Config:` syntax.
    from torchtitan_npu.simulator.trainer import SimulationTrainerConfig

    assert SimulationTrainerConfig._owner.__name__ == "SimulationTrainer"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_trainer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.trainer'`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest -v --tb=short tests/unit_tests/simulator/test_trainer.py`
Expected: `3 passed`. Verified feasible in this sandbox during design: forward -> backward -> `AdamW.step()` -> `LambdaLR.step()` all run without error on meta tensors, and the `Configurable._owner` dispatch was confirmed directly against the pinned torchtitan source (see design doc §5.1-adjacent verification notes).

If `SimulationTrainer` itself (not just `run_simulation_step`) needs a
smoke check before Task 20, do it in a CANN-enabled environment (Task 20)
with a *real* `torchtitan_npu` config -- constructing a real `Trainer`
subclass requires a full `--module`/`--config` `ConfigManager` resolution
and (for any torchtitan_npu model) `torch_npu` being importable, neither of
which is available in this sandbox.

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/trainer.py tests/unit_tests/simulator/test_trainer.py
git commit -m "feat(simulator): add SimulationTrainer wiring capture, meta-env, and export together"
git push origin master
```

---

### Task 19: `config_registry.py` -- simulation-config factory functions

**Files:**
- Create: `torchtitan_npu/simulator/config_registry.py`
- Test: `tests/smoke_tests/simulator/__init__.py`
- Test: `tests/smoke_tests/simulator/test_config_registry.py`

**Interfaces:**
- Consumes: `SimulationTrainerConfig`/`SimulationConfig` (Task 18); `torchtitan_npu.models.deepseek_v4.config_registry.deepseek_v4_pro_debug_61_layers_4k_384die` and `deepseek_v4_pro_debug_16_layers` (existing repo functions, unmodified).
- Produces: `deepseek_v4_pro_simulate_61_layers() -> SimulationTrainerConfig`; `deepseek_v4_pro_simulate_16_layers() -> SimulationTrainerConfig` (a faster-iterating variant for local smoke testing before the full 61-layer run).

This task (and its test) require `torch_npu` to be importable, because
importing anything under `torchtitan_npu.models.deepseek_v4` triggers
`torchtitan_npu/__init__.py`'s `_apply_patches()`, which imports `torch_npu`
transitively. Run this task's test inside the CANN container (see Task 20
for exact setup commands) -- it cannot run in a plain-CPU sandbox.

- [ ] **Step 1: Write `config_registry.py`**

```python
# torchtitan_npu/simulator/config_registry.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulation-config factory functions, resolved via
`--module torchtitan_npu.simulator --config <name>` (mirrors how every
other torchtitan_npu model's `config_registry.py` is resolved by
`ConfigManager`). Each function takes the *exact* existing
`torchtitan_npu.models.deepseek_v4.config_registry` factory output and
copies every field into a `SimulationTrainerConfig` -- the model_spec,
parallelism degrees, and every NPU-specific sub-config value (e.g.
`optimizer.swap_optimizer`) are reused unchanged; see design doc §7."""

from __future__ import annotations

import dataclasses

from torchtitan_npu.models.deepseek_v4.config_registry import (
    deepseek_v4_pro_debug_16_layers,
    deepseek_v4_pro_debug_61_layers_4k_384die,
)
from torchtitan_npu.simulator.trainer import SimulationConfig, SimulationTrainerConfig


def _to_simulation_config(base_config: object, output_dir: str) -> SimulationTrainerConfig:
    base_fields = {f.name: getattr(base_config, f.name) for f in dataclasses.fields(base_config)}
    return SimulationTrainerConfig(**base_fields, simulation=SimulationConfig(output_dir=output_dir))


def deepseek_v4_pro_simulate_61_layers() -> SimulationTrainerConfig:
    """Acceptance-target config: 61 layers, 384 experts,
    `expert_parallel_degree=192`, `384die` world size -- see
    docs/superpowers/specs/2026-07-01-npu-simulator-design.md."""
    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    return _to_simulation_config(base_config, output_dir="./simulator_output/deepseek_v4_pro_61_layers")


def deepseek_v4_pro_simulate_16_layers() -> SimulationTrainerConfig:
    """Smaller/faster variant for local smoke testing before running the
    full 61-layer acceptance config (Task 20)."""
    base_config = deepseek_v4_pro_debug_16_layers()
    return _to_simulation_config(base_config, output_dir="./simulator_output/deepseek_v4_pro_16_layers")
```

- [ ] **Step 2: Write the smoke test**

```python
# tests/smoke_tests/simulator/__init__.py
```

```python
# tests/smoke_tests/simulator/test_config_registry.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="requires torch_npu + CANN (see Task 20 container setup)")

from torchtitan_npu.simulator.config_registry import (  # noqa: E402
    deepseek_v4_pro_simulate_16_layers,
    deepseek_v4_pro_simulate_61_layers,
)


def test_simulate_61_layers_config_matches_acceptance_target():
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_pro_debug_61_layers_4k_384die

    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    sim_config = deepseek_v4_pro_simulate_61_layers()

    assert sim_config.model_spec.name == base_config.model_spec.name
    assert sim_config.model_spec.flavor == base_config.model_spec.flavor
    assert sim_config.parallelism.expert_parallel_degree == 192
    assert sim_config.debug.moe_force_load_balance is True  # already True on the acceptance config
    assert sim_config.optimizer.swap_optimizer == base_config.optimizer.swap_optimizer
    assert sim_config.simulation.output_dir == "./simulator_output/deepseek_v4_pro_61_layers"


def test_simulate_16_layers_config_is_a_smaller_variant():
    sim_config = deepseek_v4_pro_simulate_16_layers()
    assert sim_config.model_spec.flavor == "v4_pro_debug_16_layers"
```

- [ ] **Step 3: Run tests to verify they fail**

Run (inside the CANN container -- see Task 20): `python3 -m pytest -v --tb=short tests/smoke_tests/simulator/test_config_registry.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.config_registry'`

- [ ] **Step 4: Run tests to verify they pass**

Run (inside the CANN container): `python3 -m pytest -v --tb=short tests/smoke_tests/simulator/test_config_registry.py`
Expected: `2 passed` (or `2 skipped` if run outside a `torch_npu` environment -- that is the correct, intentional behavior of the `importorskip` guard, not a failure).

- [ ] **Step 5: Commit and push**

```bash
git add torchtitan_npu/simulator/config_registry.py tests/smoke_tests/simulator/
git commit -m "feat(simulator): add config_registry factory functions for DeepSeek-V4-Pro simulation"
git push origin master
```

---

### Task 20: End-to-end validation against the acceptance config

This task has no new source files -- it validates Tasks 1-19 together
against real `torch_npu`/CANN and the actual 61-layer/384-die acceptance
config. Run everything inside a CANN container (a fresh one is safest;
reusing `titan-npu-sim-validate`, set up during design with `torch==2.10.0+cpu`
+ `torch_npu==2.10.0` already installed, is fine for this validation pass --
re-verify with the exact `requirements.txt`-pinned versions in the team's
real CI/dev environment before treating this as final production sign-off).

- [ ] **Step 1: Start the container and confirm the repo is visible inside it**

```bash
sudo docker start titan-npu-sim-validate
sudo docker exec titan-npu-sim-validate bash -c '
  ls /workspace/torchtitan-npu-simulator/torchtitan_npu/simulator
'
```
Expected: lists every file created by Tasks 1-19 (the container bind-mounts
the repo, so files created on the host during Tasks 1-19 are automatically
visible here -- no copying needed). If the container does not exist anymore,
recreate it:

```bash
sudo docker run -d --name titan-npu-sim-validate --network host \
  -v /mnt/c/Users/admin/Documents/torchtitan-npu-simulator:/workspace/torchtitan-npu-simulator \
  --workdir /workspace/torchtitan-npu-simulator \
  quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12 sleep infinity
# then install torch/torch_npu as described in design doc §2 before continuing.
```

- [ ] **Step 2: Install this repo's dependencies inside the container**

```bash
sudo docker exec titan-npu-sim-validate bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
  cd /workspace/torchtitan-npu-simulator
  pip3 install --quiet -e .
  python3 -c "import torchtitan_npu; print(\"torchtitan_npu import OK\")"
'
```
Expected: `torchtitan_npu import OK` (this alone validates that every new
`torchtitan_npu/simulator/*` file is at minimum import-clean under real
`torch_npu`). If `torchtitan` itself is not yet installed in the container,
install it first per `requirements.txt`:
`pip3 install "torchtitan @ git+https://gitcode.com/GitHub_Trending/to/torchtitan.git@ac13e536c84e7f6647b14fa9375c3c8a8a2b8578"`.

- [ ] **Step 3: Run every simulator unit + smoke test inside the container**

```bash
sudo docker exec titan-npu-sim-validate bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
  cd /workspace/torchtitan-npu-simulator
  python3 -m pytest -v --tb=short tests/unit_tests/simulator/ tests/smoke_tests/simulator/
'
```
Expected: every test from Tasks 1-19 passes (including the ones gated by
`torch_npu`, which now actually run instead of skipping).

- [ ] **Step 4: Run the small (16-layer) config through `SimulationTrainer` first**

```bash
sudo docker exec titan-npu-sim-validate bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
  cd /workspace/torchtitan-npu-simulator
  NGPU=16 LOCAL_RANK=0 WORLD_SIZE=16 RANK=0 \
  python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_pro_simulate_16_layers \
    --comm.mode=fake_backend --training.steps=1
  ls -la simulator_output/deepseek_v4_pro_16_layers/
'
```
Expected: no crash; `simulator_output/deepseek_v4_pro_16_layers/` contains
`simulation_result.json`, `compute_graph.dot`, `summary.txt`, `trace.html`.
If this step fails, fix the failure here (smaller model = faster iteration,
easier stack traces) before attempting Step 5. Common expected fix points,
per design doc §9's risk table: an NPU custom op used by DeepSeek-V4 that
was not among the 4 ops sampled during the design spike may need a targeted
meta-kernel fallback -- add it to `torchtitan_npu/simulator/capture/op_mapping.py`'s
`OP_MAPPING` and, if the op itself errors (not just miscategorizes) on meta
tensors, that is a torch_npu-version-specific gap outside this plan's scope
to fix generically; note it and move on if it does not block the 61-layer run.

- [ ] **Step 5: Run the full 61-layer/384-die acceptance config**

```bash
sudo docker exec titan-npu-sim-validate bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH="/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
  cd /workspace/torchtitan-npu-simulator
  NGPU=384 LOCAL_RANK=0 WORLD_SIZE=384 RANK=0 \
  timeout 1800 python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_pro_simulate_61_layers \
    --comm.mode=fake_backend --training.steps=1
  ls -la simulator_output/deepseek_v4_pro_61_layers/
  cat simulator_output/deepseek_v4_pro_61_layers/summary.txt
'
```
Expected: no crash within the timeout; the same four output files are
produced. This is the literal acceptance criterion from the user's
original request ("DeepSeek-v4-pro 61 layers 的用例可以以 simulator 方式
运行起来").

- [ ] **Step 6: Verify the RankTable and communication statistics are correct**

```bash
sudo docker exec titan-npu-sim-validate bash -c '
  cd /workspace/torchtitan-npu-simulator
  python3 -c "
import json
with open(\"simulator_output/deepseek_v4_pro_61_layers/simulation_result.json\") as f:
    result = json.load(f)
rank_table = result[\"iteration\"][\"schedule\"][\"annotations\"][\"rank_table\"]
assert rank_table[\"world_size\"] == 384, rank_table[\"world_size\"]
assert rank_table[\"dim_degrees\"][\"ep\"] == 192, rank_table[\"dim_degrees\"]
data_passes = result[\"iteration\"][\"schedule\"][\"data_passes\"]
comm_primitives = {p[\"comm_primitive\"] for p in data_passes}
print(\"world_size:\", rank_table[\"world_size\"])
print(\"ep degree:\", rank_table[\"dim_degrees\"][\"ep\"])
print(\"comm primitives observed:\", comm_primitives)
assert \"all_to_all\" in comm_primitives or len(comm_primitives) > 0, \"expected at least one communication primitive\"
print(\"=== ACCEPTANCE CHECKS PASSED ===\")
"
'
```
Expected: `=== ACCEPTANCE CHECKS PASSED ===`, `world_size: 384`,
`ep degree: 192`.

- [ ] **Step 7: Commit and push (documentation only -- no source changes in this task)**

If Steps 4-6 required any fixes to earlier tasks' files (expected per the
design doc's risk table -- e.g. a newly-discovered unsupported op), those
fixes were already committed as part of returning to the relevant task. For
this task itself, record the validated command sequence by updating
`docs/superpowers/specs/2026-07-01-npu-simulator-design.md`'s §8 with the
final, confirmed-working container commands if they differ from what is
written there, then:

```bash
git add docs/superpowers/specs/2026-07-01-npu-simulator-design.md
git commit -m "docs(simulator): confirm end-to-end validation against deepseek_v4_pro_debug_61_layers_4k_384die"
git push origin master
```

---

## Post-Plan Notes

- **Non-goal reminder:** numerical correctness was never in scope (meta
  tensors carry no real data, and several NPU custom ops emit "autograd
  kernel not registered" warnings under this torch_npu version -- see
  design doc §2 finding #5). Only shape/structure/dependency/communication
  capture is validated.
- **`pipeline_parallel_degree > 1`** is explicitly out of scope (design doc
  §5.5/§9): the acceptance config uses `pp=1`. Extending to multiple
  pipeline stages requires capturing one StepGraph template per stage,
  left as documented future work.
- **Exact pinned versions:** this plan's CANN-container tasks were
  developed and verified against `torch==2.10.0+cpu` / `torch_npu==2.10.0`
  (the closest publicly-downloadable versions), not the
  `requirements.txt`-pinned `torch==2.12.0+cpu` / `torch_npu==2.12.0rc1`.
  Re-run Task 20 with the exact pinned versions in the team's real
  CI/dev environment before considering this production-validated.
