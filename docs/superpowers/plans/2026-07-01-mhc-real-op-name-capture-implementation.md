# MHC Real-Op-Name Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current blanket strip-out of the `npu_mhc_pre`/`npu_mhc_post` model
converters with simulator-only shim classes, so the captured L0 op graph for `HcPre`/`HcHead`/
`HcPost` carries the *real* op names that would run in production (Triton kernel names / real
`torch_npu` op names), instead of the pure-PyTorch base classes' unrelated op sequence.

**Architecture:** A new `OpDispatchCapture.record_synthetic_op()` method lets code manually
register an L0 `OpNode` with a chosen `raw_op_type` + analytically-known output shapes, without
running any real computation. New `torch.autograd.Function` subclasses (`_SimHcPreFn`/
`_SimHcHeadFn`/`_SimHcPostFn`) call this for every real sub-step of MHC's forward/backward
(covering both the already-dispatcher-visible ops like `npu_rms_norm`/`matmul` *and* the raw
Triton-kernel-only steps, uniformly, for full sandbox testability with zero `torch_npu`
dependency), wrapped by drop-in `nn.Module` replacements (`SimHcPre`/`SimHcHead`/`SimHcPost`) that
`SimMHCPreConverter`/`SimMHCPostConverter` install in place of the production converters --
*only* under simulation, via a reversible class-attribute patch on the existing registered
`MHCPrePostModelConfig`/`MHCPostModelConfig` classes (same pattern as `meta_env.py`'s patches).
Zero modification to any existing production file.

**Tech Stack:** Python 3.12, PyTorch 2.12 (`torch.autograd.Function`, `torch.empty(device="meta")`), pytest.

## Global Constraints

- Zero modification to any file under `torchtitan_npu/converters/`, `torchtitan_npu/ops/`, or
  `torchtitan_npu/models/` -- this feature is purely additive, consistent with this repo's
  side-loaded-package convention (verified: `git diff master --stat` on the existing simulator
  work shows 57 files changed, 0 deletions).
- All new simulator code lives under `torchtitan_npu/simulator/`.
- Every new/modified test must pass in this sandbox (no `torch_npu` installed) **except** tests
  that import `torchtitan_npu.models.deepseek_v4.model` (Tasks 2-5's tests, which construct real
  `HcPre`/`HcHead`/`HcPost` parent instances): this sandbox's plain `torchtitan` package resolves
  to a sibling-repo checkout, not the exact pinned commit, and transitively fails on an unrelated
  `AttributeError: module 'torchtitan.models.common.moe' has no attribute 'TokenReorderer'` when
  importing `model.py` (verified empirically while writing this plan) -- the *exact same*
  root cause already documented for `test_rank_table.py`/`test_trainer.py`'s pre-existing `etp`
  mismatch, not a new issue introduced by this plan. These tests are expected to fail with this
  specific `ImportError`/`AttributeError` in this sandbox and pass in the real CANN container
  (Task 7 is their first real signal). Task 1's tests (pure IR/capture, no model imports) remain
  fully sandbox-testable with no caveats.
- Full container re-validation (16-layer + 61-layer configs) is Task 8's job -- do not skip it.
- Follow this repo's TDD convention established across all 20 prior simulator tasks: write the
  failing test first, run it, then implement.
- Commit after each task passes its tests (per this repo's established workflow convention:
  commit + push after each verified step).

---

## Reference: exact shapes used throughout this plan

`n` = `hc_mult` (4 for this repo's DeepSeek-V4-Pro config), `D` = per-stream head dim,
`nD = n * D`, `total = n*n + 2*n` (the `hc_fn`/`hc_scale`/`hc_base` sizing used by
`torchtitan_npu/converters/kernels/mhc_prepost.py` and `torchtitan_npu/ops/triton/mhc_triton.py`,
verified against their docstrings and shape-assertion code):

| real function (production file) | inputs | outputs |
|---|---|---|
| `torch_npu.npu_rms_norm` | `x_flat[BS,nD]`, `gamma[nD]` | `x_norm_flat[BS,nD]`, `rstd[BS,1]` |
| `torch.matmul` (proj) | `x_norm_mat[B,S,nD]`, `weight[nD,total]` | `x_proj[B,S,total]` |
| `hc_pre_fwd` (`ops/triton/prepost_sinkhorn.py:591`) | `mixes=x_proj[B,S,total]`, `hc_scale[3]`, `hc_base[total]` | `h_pre[B,S,n]`, `h_post[B,S,n]`, `h_res[B,S,n,n]` |
| `hc_pre_bwd` (`ops/triton/prepost_sinkhorn.py:665`) | `grad_pre[B,S,n]`, `grad_post[B,S,n]`, `grad_comb[B,S,n,n]`, `mixes[B,S,total]`, `hc_scale[3]`, `hc_base[total]` | `grad_x_proj[B,S,total]`, `grad_branch_alpha[3]`, `grad_branch_beta[total]` |
| `hc_pre_bmm_forward` (`ops/triton/pre_bmm.py`) | `H_pre[B,S,n]`, `x[B,S,n,D]` | `y[B,S,D]` |
| `hc_pre_bmm_backward` (`ops/triton/pre_bmm.py`) | `H_pre[B,S,n]`, `x[B,S,n,D]`, `grad_out[B,S,D]` | `grad_H_pre[B,S,n]`, `grad_x[B,S,n,D]` |
| `hc_pre_only_fwd` (`ops/triton/prepost_sinkhorn.py:815`) | `mixes[B,S,total]`, `hc_scale[3]`, `hc_base[total]` | `h_pre[B,S,n]` |
| `hc_pre_only_bwd` (`ops/triton/prepost_sinkhorn.py:856`) | `grad_pre[B,S,n]`, `mixes[B,S,total]`, `hc_scale[3]`, `hc_base[total]` | `grad_x_proj[B,S,total]`, `grad_branch_alpha[3]`, `grad_branch_beta[total]` |
| `hc_post_bmm1_forward` (`ops/triton/post_bmm1.py:176`) | `h_out[B,S,C]`, `H_post[B,S,n]` | `[B,S,n,C]` |
| `hc_post_bmm1_backward` (`ops/triton/post_bmm1.py:224`) | `h_out[B,S,C]`, `H_post[B,S,n]`, `grad_out[B,S,n,C]` | `grad_h_out[B,S,C]`, `grad_H_post[B,S,n]` |
| `hc_post_bmm2_forward` (`ops/triton/post_bmm2.py:249`) | `H_res[B,S,n,n]`, `x[B,S,n,C]` | `[B,S,n,C]` |
| `hc_post_bmm2_backward` (`ops/triton/post_bmm2.py:295`) | `H_res[B,S,n,n]`, `x[B,S,n,C]`, `dY[B,S,n,C]` | `dH_res[B,S,n,n]`, `dX[B,S,n,C]` |
| `add_fwd` (`ops/triton/add.py:72`) | `A[M,N]`, `B[M,N]` | `[M,N]` |

**Design refinement vs. the design doc** (`docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md`):
the design doc's table marks `npu_rms_norm`/`matmul` as "not needing a shim" (since they're
already real, dispatcher-visible, meta-safe ops). This plan shims **all** of MHC's sub-steps
uniformly through `record_synthetic_op`, including `npu_rms_norm`/`matmul` -- this keeps
`mhc_shim.py` free of any `torch_npu` import, so every shim unit test in this plan runs in this
sandbox with zero `torch_npu` dependency, while producing the *exact same* real op names in the
final captured graph (the observable requirement from the design doc is unchanged: node labels
match production op names).

---

### Task 1: `record_synthetic_op` capture infrastructure

**Files:**
- Modify: `torchtitan_npu/simulator/capture/dispatch_capture.py`
- Test: `tests/unit_tests/simulator/capture/test_dispatch_capture.py`

**Interfaces:**
- Produces: `OpDispatchCapture.record_synthetic_op(self, raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str = "") -> None`
- Produces: `torchtitan_npu.simulator.capture.dispatch_capture.get_active_capture() -> OpDispatchCapture | None`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit_tests/simulator/capture/test_dispatch_capture.py`:

```python
def test_record_synthetic_op_creates_a_node_with_given_raw_op_type():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(2, 4, device="meta")
        b = torch.empty(2, 4, device="meta")
        capture.record_synthetic_op("triton.hc_pre_bmm_forward", inputs=[a], outputs=[b])
    nodes = capture.build_nodes()
    synthetic = [n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.hc_pre_bmm_forward"]
    assert len(synthetic) == 1
    assert synthetic[0].op_type == "unknown"  # not in OP_MAPPING -- expected, display_op_label handles it
    assert [o.shape for o in synthetic[0].outputs] == [(2, 4)]


def test_record_synthetic_op_wires_producer_consumer_edges():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(2, 4, device="meta")
        mid = torch.empty(2, 4, device="meta")
        capture.record_synthetic_op("triton.step_one", inputs=[a], outputs=[mid])
        out = mid.relu()  # a REAL dispatched op consuming the synthetic op's output
    nodes = capture.build_nodes()
    synthetic_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.step_one")
    relu_node = next(n for n in nodes.values() if "relu" in n.annotations["raw_op_type"])
    assert relu_node.predecessors == [synthetic_node.op_id]
    assert synthetic_node.op_id in relu_node.predecessors
    assert relu_node.op_id in synthetic_node.successors


def test_record_synthetic_op_respects_phase_provider():
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        a = torch.randn(2, device="meta")
        b = torch.empty(2, device="meta")
        capture.record_synthetic_op("triton.fwd_step", inputs=[a], outputs=[b])
        phase_box["value"] = "backward"
        c = torch.empty(2, device="meta")
        capture.record_synthetic_op("triton.bwd_step", inputs=[b], outputs=[c])
    nodes = capture.build_nodes()
    fwd_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.fwd_step")
    bwd_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.bwd_step")
    assert fwd_node.annotations["phase"] == "forward"
    assert bwd_node.annotations["phase"] == "backward"


def test_get_active_capture_returns_none_outside_context():
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    assert get_active_capture() is None


def test_get_active_capture_returns_the_entered_instance():
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    capture = OpDispatchCapture()
    with capture:
        assert get_active_capture() is capture
    assert get_active_capture() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit_tests/simulator/capture/test_dispatch_capture.py -v -k "synthetic or active_capture"`
Expected: FAIL with `AttributeError: 'OpDispatchCapture' object has no attribute 'record_synthetic_op'` (first 3 tests) and `ImportError: cannot import name 'get_active_capture'` (last 2 tests).

- [ ] **Step 3: Implement `record_synthetic_op` + `get_active_capture`**

In `torchtitan_npu/simulator/capture/dispatch_capture.py`, replace the whole
`__torch_dispatch__` method plus everything below it with (this extracts the shared
event-recording logic into `_record_event`, used by both the real dispatch path and the new
synthetic path):

```python
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN201
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        flat_inputs = _flatten_tensors(args) + _flatten_tensors(tuple(kwargs.values()))
        flat_outputs = _flatten_tensors(result if isinstance(result, (tuple, list)) else (result,))
        module_path = self.module_path_tracker.current_path() if self.module_path_tracker else ""
        self._record_event(str(func), flat_inputs, flat_outputs, module_path)

        return result

    def record_synthetic_op(
        self,
        raw_op_type: str,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        module_path: str = "",
    ) -> None:
        """Manually register one synthetic L0 event, as if `raw_op_type` had
        gone through __torch_dispatch__ normally. Used by
        torchtitan_npu.simulator.hardware_shims for ops that cannot execute
        for real (raw Triton kernels / JIT-compiled extensions) but whose
        real op name + output shape are known analytically. Participates in
        the same producer/consumer id(tensor) wiring, repeat_count dedup,
        and phase tagging as real dispatched events."""
        self._record_event(raw_op_type, inputs, outputs, module_path)

    def _record_event(
        self,
        raw_op_type: str,
        flat_inputs: list[torch.Tensor],
        flat_outputs: list[torch.Tensor],
        module_path: str,
    ) -> None:
        predecessors = sorted({self._producer[id(t)] for t in flat_inputs if id(t) in self._producer})
        input_metas = [to_tensor_meta(t, name=f"in_{i}") for i, t in enumerate(flat_inputs)]
        output_metas = [to_tensor_meta(t, name=f"out_{i}") for i, t in enumerate(flat_outputs)]

        op_type = to_canonical_op_type(raw_op_type)
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

        for t in flat_outputs:
            self._producer[id(t)] = op_id

    def __enter__(self) -> "OpDispatchCapture":
        super().__enter__()
        global _active_capture
        self._previous_active_capture = _active_capture
        _active_capture = self
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        global _active_capture
        _active_capture = self._previous_active_capture
        super().__exit__(exc_type, exc_val, exc_tb)

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


_active_capture: "OpDispatchCapture | None" = None


def get_active_capture() -> "OpDispatchCapture | None":
    """Returns the `OpDispatchCapture` instance currently inside its `with`
    block (there is at most one active at a time -- one step is captured at
    a time), or `None` if no capture is active. Lets code that has no
    direct reference to the capture instance (e.g. hardware_shims'
    nn.Module replacements, which run deep inside a model's forward/backward
    with no capture parameter threaded through) reach it to call
    `record_synthetic_op`."""
    return _active_capture
```

Also add `self._previous_active_capture: OpDispatchCapture | None = None` to `__init__`'s body
(right after `self._last_signature: tuple | None = None`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit_tests/simulator/capture/test_dispatch_capture.py -v`
Expected: all PASS (previous tests + 5 new ones).

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/capture/dispatch_capture.py tests/unit_tests/simulator/capture/test_dispatch_capture.py
git commit -m "feat(simulator): add OpDispatchCapture.record_synthetic_op for manual L0 node injection"
```

---

### Task 2: `hardware_shims` package + `SimHcPre`

**Files:**
- Create: `torchtitan_npu/simulator/hardware_shims/__init__.py`
- Create: `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`
- Test: `tests/unit_tests/simulator/hardware_shims/__init__.py` (empty)
- Test: `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`

**Interfaces:**
- Consumes: `torchtitan_npu.simulator.capture.dispatch_capture.get_active_capture()` (Task 1)
- Produces: `SimHcPre(nn.Module)` with `__init__(self, parent: HcPre)` (mirrors the real
  `NpuHcPre.__init__(self, parent: HcPre)` pattern from `mhc_prepost.py` exactly: shallow-copies
  `parent.__dict__` so `hc_mult`/`hc_sinkhorn_iters`/`hc_eps`/`norm_eps` are reused verbatim, no
  separate constructor-argument list to keep in sync with `HcPre.Config`) and
  `forward(self, x, hc_fn, hc_scale, hc_base) -> tuple[Tensor, Tensor, Tensor]` (matches
  `NpuHcPre.forward`'s signature exactly).

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/simulator/hardware_shims/__init__.py` (empty file).

Create `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.deepseek_v4.model import HcPre
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcPre


def _build_sim_hc_pre(n: int = 4, D: int = 8) -> tuple[SimHcPre, dict]:
    parent = HcPre(HcPre.Config(hc_mult=n, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6))
    shim = SimHcPre(parent)
    total = n * n + 2 * n
    tensors = {
        "x": torch.randn(2, 3, n * D, requires_grad=True),
        "hc_fn": torch.randn(total, n * D, requires_grad=True),
        "hc_scale": torch.randn(3, requires_grad=True),
        "hc_base": torch.randn(total, requires_grad=True),
    }
    return shim, tensors


def test_sim_hc_pre_forward_returns_correct_shapes():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    assert y.shape == (2, 3, 8)  # [B,S,D]
    assert h_post.shape == (2, 3, 4)  # [B,S,n]
    assert h_res.shape == (2, 3, 4, 4)  # [B,S,n,n]


def test_sim_hc_pre_records_real_op_names_in_active_capture():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    capture = OpDispatchCapture()
    with capture:
        shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "npu.npu_rms_norm.default" in raw_names
    assert "aten.matmul.default" in raw_names
    assert "triton.hc_pre_fwd" in raw_names
    assert "triton.hc_pre_bmm_forward" in raw_names


def test_sim_hc_pre_backward_propagates_gradient_to_input():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
    (y.sum() + h_post.sum() + h_res.sum()).backward()
    assert t["x"].grad is not None
    assert t["x"].grad.shape == t["x"].shape
    assert t["hc_fn"].grad is not None
    assert t["hc_scale"].grad is not None
    assert t["hc_base"].grad is not None


def test_sim_hc_pre_records_backward_op_names_only_during_backward_phase():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        y, h_post, h_res = shim(t["x"], t["hc_fn"], t["hc_scale"], t["hc_base"])
        phase_box["value"] = "backward"
        (y.sum() + h_post.sum() + h_res.sum()).backward()
    nodes = capture.build_nodes()
    bwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "backward"}
    assert "triton.hc_pre_bwd" in bwd_names
    assert "triton.hc_pre_bmm_backward" in bwd_names
    fwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "forward"}
    assert "triton.hc_pre_fwd" in fwd_names


def test_sim_hc_pre_works_on_meta_device():
    shim, t = _build_sim_hc_pre(n=4, D=8)
    meta_tensors = {k: v.detach().to("meta").requires_grad_(True) for k, v in t.items()}
    y, h_post, h_res = shim(meta_tensors["x"], meta_tensors["hc_fn"], meta_tensors["hc_scale"], meta_tensors["hc_base"])
    assert y.device.type == "meta"
    (y.sum() + h_post.sum() + h_res.sum()).backward()
    assert meta_tensors["x"].grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.hardware_shims'`

- [ ] **Step 3: Implement `SimHcPre`**

Create `torchtitan_npu/simulator/hardware_shims/__init__.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator-only replacements for model submodules whose real production
implementation requires actual NPU hardware (raw Triton kernels, JIT-
compiled aclnn extensions) and therefore cannot execute under meta-device
simulation. Each shim preserves the *real* op name that would run in
production (via OpDispatchCapture.record_synthetic_op) and the real output
shape (computed analytically), without invoking any real kernel. See
docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md."""
```

Create `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shim for MHC's HcPre (`torchtitan_npu.converters.kernels.mhc_prepost.NpuHcPre` /
`torchtitan_npu.ops.triton.mhc_triton.MHCPreTriton` in production). Records the real production
op names (`npu_rms_norm`, `matmul`, and the Triton kernels `hc_pre_fwd`/`hc_pre_bmm_forward`, or
their backward counterparts) into the active OpDispatchCapture, with analytically-derived shapes
-- never invoking torch_npu or Triton for real. See design doc for the exact shape formulas."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


def _empty_like_shape(shape: tuple[int, ...], ref: torch.Tensor) -> torch.Tensor:
    return torch.empty(shape, dtype=ref.dtype, device=ref.device)


class _SimHcPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hc_fn, hc_scale, hc_base, hc_mult, module_path):  # noqa: ANN001
        B, S, nD = x.shape
        D = nD // hc_mult
        total = hc_fn.shape[0]
        dtype = x.dtype

        x_flat = x.reshape(B * S, nD)
        x_norm_flat = _empty_like_shape((B * S, nD), x_flat)
        rstd = _empty_like_shape((B * S, 1), x_flat)
        _record("npu.npu_rms_norm.default", [x_flat], [x_norm_flat, rstd], module_path)

        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = _empty_like_shape((B, S, total), x_flat)
        _record("aten.matmul.default", [x_norm_mat, hc_fn], [x_proj], module_path)

        h_pre = _empty_like_shape((B, S, hc_mult), x_flat)
        h_post = _empty_like_shape((B, S, hc_mult), x_flat)
        h_res = _empty_like_shape((B, S, hc_mult, hc_mult), x_flat)
        _record("triton.hc_pre_fwd", [x_proj, hc_scale, hc_base], [h_pre, h_post, h_res], module_path)

        x_unflatten = x.unflatten(dim=-1, sizes=(hc_mult, D))
        y = torch.empty((B, S, D), dtype=dtype, device=x.device)
        _record("triton.hc_pre_bmm_forward", [h_pre, x_unflatten], [y], module_path)

        ctx.save_for_backward(x, hc_fn, hc_scale, hc_base, h_pre)
        ctx.hc_mult, ctx.module_path = hc_mult, module_path
        ctx.B, ctx.S, ctx.nD, ctx.D, ctx.total = B, S, nD, D, total
        return y, h_post, h_res

    @staticmethod
    def backward(ctx, grad_y, grad_h_post, grad_h_res):  # noqa: ANN001
        x, hc_fn, hc_scale, hc_base, h_pre = ctx.saved_tensors
        B, S, nD, D, hc_mult, total = ctx.B, ctx.S, ctx.nD, ctx.D, ctx.hc_mult, ctx.total

        x_unflatten_shape = (B, S, hc_mult, D)
        grad_h_pre = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_x_direct = torch.empty(x_unflatten_shape, dtype=x.dtype, device=x.device)
        _record(
            "triton.hc_pre_bmm_backward",
            [h_pre, x.unflatten(dim=-1, sizes=(hc_mult, D)), grad_y],
            [grad_h_pre, grad_x_direct],
            ctx.module_path,
        )

        grad_x_proj = _empty_like_shape((B, S, total), h_pre)
        grad_branch_alpha = torch.empty_like(hc_scale)
        grad_branch_beta = torch.empty_like(hc_base)
        _record(
            "triton.hc_pre_bwd",
            [grad_h_pre, grad_h_post, grad_h_res, x, hc_scale, hc_base],
            [grad_x_proj, grad_branch_alpha, grad_branch_beta],
            ctx.module_path,
        )

        grad_x = torch.empty((B, S, nD), dtype=x.dtype, device=x.device)
        grad_hc_fn = torch.empty_like(hc_fn)
        return grad_x, grad_hc_fn, grad_branch_alpha, grad_branch_beta, None, None


class SimHcPre(nn.Module):
    """Drop-in simulator replacement for `NpuHcPre`/`NpuHcPreFused`
    (`torchtitan_npu.converters.kernels.mhc_prepost`). Same forward()
    signature; never runs real Triton/torch_npu, only records the real op
    names + analytically-correct shapes.

    `__init__` mirrors `NpuHcPre.__init__(self, parent: HcPre)` exactly
    (shallow `__dict__` copy, no `super().__init__()` call): `parent` is
    already a fully-initialized `nn.Module` instance, so its `__dict__`
    already carries every internal `nn.Module` bookkeeping attribute
    (`_parameters`, `_buffers`, `_modules`, hook registries, ...) alongside
    `hc_mult`/`hc_sinkhorn_iters`/`hc_eps`/`norm_eps` -- copying it in one
    step reproduces both correctly without hardcoding `HcPre.Config`'s
    field names here."""

    def __init__(self, parent: "HcPre") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        x = x.flatten(2)
        module_path = self.__class__.__name__
        return _SimHcPreFn.apply(x, hc_fn, hc_scale, hc_base, self.hc_mult, module_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/ tests/unit_tests/simulator/hardware_shims/
git commit -m "feat(simulator): add SimHcPre hardware shim for MHC pre-mapping stage"
```

---

### Task 3: `SimHcHead` (MHC head-only variant)

**Files:**
- Modify: `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`
- Test: `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`

**Interfaces:**
- Consumes: `_record`, `_empty_like_shape` (Task 2, same file)
- Produces: `SimHcHead(nn.Module)` with `__init__(self, parent: "HcHead")` (mirrors
  `NpuHcHead.__init__(self, parent: HcHead)`'s exact `__dict__` shallow-copy pattern -- this is
  **required**, not just a style choice: `HcHead` owns real `nn.Parameter`s
  (`hc_head_fn`/`hc_head_base`/`hc_head_scale`, sized by `HcHead.Config.dim`/`hc_mult` in
  `torchtitan_npu/models/deepseek_v4/model.py:1064-1084`) that must be reused verbatim, not
  recreated with a guessed shape) and `forward(self, x) -> Tensor` (matches `NpuHcHead.forward`'s
  signature from `mhc_prepost.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`:

```python
from torchtitan_npu.models.deepseek_v4.model import HcHead  # add to existing import line
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcHead  # add to existing import line


def _build_sim_hc_head(n: int = 4, D: int = 8) -> tuple["SimHcHead", dict]:
    parent = HcHead(HcHead.Config(norm_eps=1e-6, hc_eps=1e-6, hc_mult=n, dim=D))
    shim = SimHcHead(parent)
    tensors = {"x": torch.randn(2, 3, n, D, requires_grad=True)}
    return shim, tensors


def test_sim_hc_head_forward_returns_correct_shape():
    shim, t = _build_sim_hc_head(n=4, D=8)
    y = shim(t["x"])
    assert y.shape == (2, 3, 8)  # [B,S,D]


def test_sim_hc_head_records_real_op_names():
    shim, t = _build_sim_hc_head(n=4, D=8)
    capture = OpDispatchCapture()
    with capture:
        shim(t["x"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "triton.hc_pre_only_fwd" in raw_names
    assert "triton.hc_pre_bmm_forward" in raw_names


def test_sim_hc_head_backward_propagates_gradient():
    shim, t = _build_sim_hc_head(n=4, D=8)
    y = shim(t["x"])
    y.sum().backward()
    assert t["x"].grad is not None
    assert t["x"].grad.shape == t["x"].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v -k head`
Expected: FAIL with `ImportError: cannot import name 'SimHcHead'`

- [ ] **Step 3: Implement `SimHcHead`**

Append to `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`:

```python
class _SimHcHeadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hc_head_fn, hc_head_scale, hc_head_base, hc_mult, module_path):  # noqa: ANN001
        # x arrives pre-flattened to [B,S,nD] by SimHcHead.forward, matching NpuHcHead.
        B, S, nD = x.shape
        D = nD // hc_mult
        dtype = x.dtype

        # NpuHcHead's real forward calls MHCPreOnlyTriton.apply(x, hc_head_fn, hc_head_scale,
        # hc_head_base, ...) directly on the already-flattened x. Tracing MHCPreOnlyTriton.forward
        # (ops/triton/mhc_triton.py): `weight = hc_head_fn.t()` transposes hc_head_fn's real shape
        # `[hc_mult, hc_dim]` (HcHead.__init__, model.py:1064-1084) to `[hc_dim, hc_mult]`, so
        # `x_proj = matmul(x_norm_mat[B,S,nD], weight[nD,hc_mult]) = [B,S,hc_mult]` -- the "mixes"
        # last dim here is `hc_mult`, NOT `n*n+2*n` (that larger size is specific to the main
        # HcPre/MHCPreTriton path in SimHcPre, which uses a differently-shaped hc_fn weight).
        # hc_head_scale/hc_head_base are real HcHead parameters too: shape [1] and [hc_mult]
        # respectively (NOT [3]/[n*n+2*n] like HcPre's hc_scale/hc_base) -- confirmed from
        # HcHead.__init__'s actual nn.Parameter allocations.
        mixes = _empty_like_shape((B, S, hc_mult), x)
        h_pre = _empty_like_shape((B, S, hc_mult), x)
        _record("triton.hc_pre_only_fwd", [mixes, hc_head_scale, hc_head_base], [h_pre], module_path)

        x_unflatten = x.unflatten(dim=-1, sizes=(hc_mult, D))
        y = torch.empty((B, S, D), dtype=dtype, device=x.device)
        _record("triton.hc_pre_bmm_forward", [h_pre, x_unflatten], [y], module_path)

        ctx.save_for_backward(x, hc_head_fn, hc_head_scale, hc_head_base, h_pre)
        ctx.hc_mult, ctx.module_path = hc_mult, module_path
        ctx.B, ctx.S, ctx.nD, ctx.D = B, S, nD, D
        return y

    @staticmethod
    def backward(ctx, grad_y):  # noqa: ANN001
        x, hc_head_fn, hc_head_scale, hc_head_base, h_pre = ctx.saved_tensors
        B, S, nD, D, hc_mult = ctx.B, ctx.S, ctx.nD, ctx.D, ctx.hc_mult

        grad_h_pre = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_x_direct = torch.empty((B, S, hc_mult, D), dtype=x.dtype, device=x.device)
        _record(
            "triton.hc_pre_bmm_backward",
            [h_pre, x.unflatten(dim=-1, sizes=(hc_mult, D)), grad_y],
            [grad_h_pre, grad_x_direct],
            ctx.module_path,
        )

        grad_x_proj = _empty_like_shape((B, S, hc_mult), h_pre)
        grad_branch_alpha = torch.empty_like(hc_head_scale)
        grad_branch_beta = torch.empty_like(hc_head_base)
        _record(
            "triton.hc_pre_only_bwd",
            [grad_h_pre, x, hc_head_scale, hc_head_base],
            [grad_x_proj, grad_branch_alpha, grad_branch_beta],
            ctx.module_path,
        )

        grad_x = torch.empty((B, S, nD), dtype=x.dtype, device=x.device)
        grad_hc_head_fn = torch.empty_like(hc_head_fn)
        return grad_x, grad_hc_head_fn, grad_branch_alpha, grad_branch_beta, None, None


class SimHcHead(nn.Module):
    """Drop-in simulator replacement for `NpuHcHead`
    (`torchtitan_npu.converters.kernels.mhc_prepost`).

    `__init__` mirrors `NpuHcHead.__init__(self, parent: HcHead)`'s exact
    `__dict__` shallow-copy pattern -- **required**, not just style: unlike
    `HcPre`/`HcPost`, `HcHead` owns real `nn.Parameter`s
    (`hc_head_fn: [hc_mult,hc_dim]`, `hc_head_base: [hc_mult]`,
    `hc_head_scale: [1]`, per `HcHead.__init__` in
    `torchtitan_npu/models/deepseek_v4/model.py:1064-1084`) that must be
    reused verbatim from `parent`, not recreated with a guessed shape.

    **Important, found via real-container validation (Task 7):** unlike
    `HcPre`, `HcHead` does NOT store `hc_mult` as an instance attribute --
    `HcHead.__init__` only uses it as a local variable to size
    `hc_head_fn`/`hc_head_base` (`hc_dim = hc_mult * config.dim`), never
    doing `self.hc_mult = ...`. So `self.__dict__.update(parent.__dict__)`
    correctly carries over `hc_head_fn`/`hc_head_base`/`hc_head_scale`/
    `hc_eps`/`norm_eps`, but there is no `self.hc_mult` to copy -- `forward`
    must derive it from `self.hc_head_fn.shape[0]` (`hc_head_fn`'s real
    shape is `[hc_mult, hc_dim]`), not reference a nonexistent
    `self.hc_mult`."""

    def __init__(self, parent: "HcHead") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor):
        x = x.flatten(2)
        module_path = self.__class__.__name__
        hc_mult = self.hc_head_fn.shape[0]
        return _SimHcHeadFn.apply(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, hc_mult, module_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v`
Expected: all tests (Task 2's 6 + Task 3's 3) PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/mhc_shim.py tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py
git commit -m "feat(simulator): add SimHcHead hardware shim for MHC head-only aggregation"
```

---

### Task 4: `SimHcPost`

**Files:**
- Modify: `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`
- Test: `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`

**Interfaces:**
- Consumes: `_record`, `_empty_like_shape` (Task 2, same file)
- Produces: `SimHcPost(nn.Module)` with `__init__(self, parent: "HcPost")` (mirrors
  `NpuHcPost.__init__(self, parent: HcPost)`'s `__dict__` shallow-copy pattern, for consistency
  with `SimHcPre`/`SimHcHead` even though `HcPost` owns no extra parameters/attributes beyond
  base `Module` -- keeps all three shim constructors uniform) and
  `forward(self, x, residual, post, comb) -> Tensor` (matches `NpuHcPost.forward` from
  `mhc_prepost.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py`:

```python
from torchtitan_npu.models.deepseek_v4.model import HcPost  # add to existing import line
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcPost  # add to existing import line


def _build_sim_hc_post(n: int = 4, D: int = 8) -> tuple["SimHcPost", dict]:
    parent = HcPost(HcPost.Config())
    shim = SimHcPost(parent)
    tensors = {
        "x": torch.randn(2, 3, D, requires_grad=True),
        "residual": torch.randn(2, 3, n, D, requires_grad=True),
        "post": torch.randn(2, 3, n, requires_grad=True),
        "comb": torch.randn(2, 3, n, n, requires_grad=True),
    }
    return shim, tensors


def test_sim_hc_post_forward_returns_correct_shape():
    shim, t = _build_sim_hc_post(n=4, D=8)
    y = shim(t["x"], t["residual"], t["post"], t["comb"])
    assert y.shape == (2, 3, 4, 8)  # [B,S,N,D] (matches production NpuHcPost.forward's return -- it
    # reshapes MHCPostTriton's flat [B,S,N*D] output back to 4D before returning, mhc_prepost.py:277)


def test_sim_hc_post_records_real_op_names():
    shim, t = _build_sim_hc_post(n=4, D=8)
    capture = OpDispatchCapture()
    with capture:
        shim(t["x"], t["residual"], t["post"], t["comb"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "triton.hc_post_bmm1_forward" in raw_names
    assert "triton.hc_post_bmm2_forward" in raw_names
    assert "triton.add_fwd" in raw_names


def test_sim_hc_post_backward_propagates_gradient_to_all_inputs():
    shim, t = _build_sim_hc_post(n=4, D=8)
    y = shim(t["x"], t["residual"], t["post"], t["comb"])
    y.sum().backward()
    for key in ("x", "residual", "post", "comb"):
        assert t[key].grad is not None, f"{key} did not receive a gradient"
        assert t[key].grad.shape == t[key].shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v -k post`
Expected: FAIL with `ImportError: cannot import name 'SimHcPost'`

- [ ] **Step 3: Implement `SimHcPost`**

Append to `torchtitan_npu/simulator/hardware_shims/mhc_shim.py`:

```python
class _SimHcPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, h_post, h_res, module_path):  # noqa: ANN001
        B, S, D = x.shape
        N = h_post.shape[-1]
        dtype = x.dtype

        bmm1 = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm1_forward", [x, h_post], [bmm1], module_path)

        residual_unflat = residual.view(B, S, N, D)
        bmm2 = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm2_forward", [h_res, residual_unflat], [bmm2], module_path)

        result_flat = torch.empty((B * S, N * D), dtype=torch.float32, device=x.device)
        _record(
            "triton.add_fwd",
            [bmm1.reshape(B * S, -1), bmm2.reshape(B * S, -1)],
            [result_flat],
            module_path,
        )

        ctx.save_for_backward(x, residual, h_post, h_res)
        ctx.module_path = module_path
        ctx.B, ctx.S, ctx.D, ctx.N = B, S, D, N
        return result_flat.view(B, S, N * D).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        x, residual, h_post, h_res = ctx.saved_tensors
        B, S, D, N = ctx.B, ctx.S, ctx.D, ctx.N

        grad_out_4d = grad_output.view(B, S, N, D).float()

        grad_x = torch.empty((B, S, D), dtype=torch.float32, device=x.device)
        grad_h_post = torch.empty((B, S, N), dtype=torch.float32, device=x.device)
        _record("triton.hc_post_bmm1_backward", [x, h_post, grad_out_4d], [grad_x, grad_h_post], ctx.module_path)

        residual_unflat = residual.view(B, S, N, D)
        grad_h_res = torch.empty((B, S, N, N), dtype=torch.float32, device=x.device)
        grad_residual_unflat = torch.empty((B, S, N, D), dtype=torch.float32, device=x.device)
        _record(
            "triton.hc_post_bmm2_backward",
            [h_res, residual_unflat, grad_out_4d],
            [grad_h_res, grad_residual_unflat],
            ctx.module_path,
        )
        grad_residual = grad_residual_unflat.flatten(-2)

        return grad_x, grad_residual, grad_h_post, grad_h_res


class SimHcPost(nn.Module):
    """Drop-in simulator replacement for `NpuHcPost`
    (`torchtitan_npu.converters.kernels.mhc_prepost`). `__init__` mirrors
    `NpuHcPost.__init__(self, parent: HcPost)`'s `__dict__` shallow-copy
    pattern for consistency with `SimHcPre`/`SimHcHead`, even though
    `HcPost` owns no extra parameters beyond base `Module`.

    `forward` mirrors `NpuHcPost.forward`'s exact wrapping (mhc_prepost.py:236-278),
    which is a THIN WRAPPER around `MHCPostTriton.apply(...)` (mirrored here by
    `_SimHcPostFn`): it takes `residual` as 4D `[B,S,N,D]`, flattens it to
    `[B,S,N*D]` before calling the lower-level function (matching
    `_SimHcPostFn`'s own `residual.view(B,S,N,D)` internal-unflatten
    convention -- `_SimHcPostFn` expects an already-flattened `residual`,
    exactly like `MHCPostTriton.forward` does), then reshapes the `[B,S,N*D]`
    result back to 4D `[B,S,N,D]` before returning -- `NpuHcPost.forward`
    does the identical `y = y.view(dim_b, dim_s, dim_n, dim_d)` reshape
    at its very end (mhc_prepost.py:277) before returning to its caller."""

    def __init__(self, parent: "HcPost") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        dim_b, dim_s, dim_n, dim_d = residual.shape
        residual_flat = residual.flatten(2)
        module_path = self.__class__.__name__
        y = _SimHcPostFn.apply(x, residual_flat, post, comb, module_path)
        return y.view(dim_b, dim_s, dim_n, dim_d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py -v`
Expected: all tests (Task 2's 6 + Task 3's 3 + Task 4's 3 = 12) PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/mhc_shim.py tests/unit_tests/simulator/hardware_shims/test_mhc_shim.py
git commit -m "feat(simulator): add SimHcPost hardware shim for MHC post-mapping stage"
```

---

### Task 5: `SimMHCPreConverter`/`SimMHCPostConverter` + apply/unapply wiring

**Files:**
- Create: `torchtitan_npu/simulator/hardware_shims/mhc_converter.py`
- Test: `tests/unit_tests/simulator/hardware_shims/test_mhc_converter.py`

**Interfaces:**
- Consumes: `SimHcPre`, `SimHcHead`, `SimHcPost` (Tasks 2-4); `HcPre`, `HcHead`, `HcPost` from `torchtitan_npu.models.deepseek_v4.model`; `MHCPrePostModelConfig`, `MHCPostModelConfig` from `torchtitan_npu.converters.kernels.mhc_prepost`; `replace_module_with_name` from `torchtitan_npu.converters.convert_utils`.
- Produces: `apply_mhc_shims() -> None`, `unapply_mhc_shims() -> None` (module-level functions in `mhc_converter.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/simulator/hardware_shims/test_mhc_converter.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.converters.kernels.mhc_prepost import MHCPostModelConfig, MHCPrePostModelConfig
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.simulator.hardware_shims.mhc_converter import apply_mhc_shims, unapply_mhc_shims
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcHead, SimHcPost, SimHcPre


class _FakeModelSpec:
    name = "deepseek_v4"


def _make_hc_pre() -> HcPre:
    config = HcPre.Config(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-6)
    return HcPre(config)


def test_apply_mhc_shims_replaces_converter_target_classes():
    original_pre = MHCPrePostModelConfig.model_converter
    original_post = MHCPostModelConfig.model_converter
    try:
        apply_mhc_shims()
        assert MHCPrePostModelConfig.model_converter is not original_pre
        assert MHCPostModelConfig.model_converter is not original_post
    finally:
        unapply_mhc_shims()
        assert MHCPrePostModelConfig.model_converter is original_pre
        assert MHCPostModelConfig.model_converter is original_post


def test_applied_mhc_pre_converter_replaces_hc_pre_with_sim_hc_pre():
    apply_mhc_shims()
    try:
        model = nn.Sequential()
        model.add_module("hc_pre", _make_hc_pre())
        converter = MHCPrePostModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.hc_pre, SimHcPre)
        assert model.hc_pre.hc_mult == 4
    finally:
        unapply_mhc_shims()


def test_applied_mhc_post_converter_replaces_hc_post_and_hc_head():
    apply_mhc_shims()
    try:
        model = nn.Sequential()
        model.add_module("hc_post", HcPost(HcPost.Config()))
        model.add_module("hc_head", HcHead(HcHead.Config(norm_eps=1e-6, hc_eps=1e-6, hc_mult=4, dim=8)))
        converter = MHCPostModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.hc_post, SimHcPost)
        assert isinstance(model.hc_head, SimHcHead)
    finally:
        unapply_mhc_shims()


def test_unapply_is_idempotent_when_not_applied():
    unapply_mhc_shims()  # must not raise even if apply_mhc_shims was never called
    unapply_mhc_shims()  # calling twice must also not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_converter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.hardware_shims.mhc_converter'`

- [ ] **Step 3: Implement `mhc_converter.py`**

First inspect the exact `HcHead.Config`/`HcPre.Config`/`HcPost.Config` field names to confirm
(already verified in this plan's investigation): `HcPre.Config(hc_mult, hc_sinkhorn_iters,
hc_eps, norm_eps)`, `HcHead.Config(norm_eps, hc_eps, hc_mult, dim)`, `HcPost.Config()` (no
fields beyond base `Module.Config`).

Create `torchtitan_npu/simulator/hardware_shims/mhc_converter.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reversible class-attribute patch that swaps the `npu_mhc_pre`/`npu_mhc_post` model
converters' target implementation classes for the simulator's shape-only shims
(SimHcPre/SimHcHead/SimHcPost), instead of SimulationTrainer stripping these converters out
entirely. Mirrors meta_env.py's established "patch a class attribute, track the original for a
symmetric unpatch" pattern -- MHCPrePostModelConfig/MHCPostModelConfig are the real, singleton,
already-registered converter-config classes from torchtitan_npu.converters.kernels.mhc_prepost
(zero modification to that file itself: this module only reassigns their `model_converter`
class attribute at runtime, under simulation)."""

from __future__ import annotations

import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.mhc_prepost import MHCPostModelConfig, MHCPrePostModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.deepseek_v4.model import HcHead, HcPost, HcPre
from torchtitan_npu.simulator.hardware_shims.mhc_shim import SimHcHead, SimHcPost, SimHcPre

_original_mhc_pre_converter: type | None = None
_original_mhc_post_converter: type | None = None


class SimMHCPreConverter(ModelCustomConverter):
    """Replaces every `HcPre` submodule with `SimHcPre` -- never selects the
    real fused/Triton implementation (see design doc §2: neither path can
    execute under simulation)."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPre):
                replace_module_with_name(model, name, SimHcPre(module))


class SimMHCPostConverter(ModelCustomConverter):
    """Replaces every `HcPost` submodule with `SimHcPost` and every
    `HcHead` submodule with `SimHcHead`."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, HcPost):
                replace_module_with_name(model, name, SimHcPost(module))
            if isinstance(module, HcHead):
                replace_module_with_name(model, name, SimHcHead(module))


def apply_mhc_shims() -> None:
    """Patch MHCPrePostModelConfig.model_converter / MHCPostModelConfig.model_converter to
    point at the Sim* converters above. Idempotent: calling twice in a row is safe (the second
    call just re-saves the already-patched value as "original", which unapply_mhc_shims still
    correctly restores to the value active before the *first* call, since SimulationTrainer only
    ever calls apply once per process)."""
    global _original_mhc_pre_converter, _original_mhc_post_converter
    if _original_mhc_pre_converter is None:
        _original_mhc_pre_converter = MHCPrePostModelConfig.model_converter
    if _original_mhc_post_converter is None:
        _original_mhc_post_converter = MHCPostModelConfig.model_converter
    MHCPrePostModelConfig.model_converter = SimMHCPreConverter
    MHCPostModelConfig.model_converter = SimMHCPostConverter


def unapply_mhc_shims() -> None:
    """Restore the original converter classes. Safe to call even if
    apply_mhc_shims() was never called (no-op), and safe to call more than
    once (idempotent)."""
    global _original_mhc_pre_converter, _original_mhc_post_converter
    if _original_mhc_pre_converter is not None:
        MHCPrePostModelConfig.model_converter = _original_mhc_pre_converter
        _original_mhc_pre_converter = None
    if _original_mhc_post_converter is not None:
        MHCPostModelConfig.model_converter = _original_mhc_post_converter
        _original_mhc_post_converter = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_mhc_converter.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/mhc_converter.py tests/unit_tests/simulator/hardware_shims/test_mhc_converter.py
git commit -m "feat(simulator): add reversible MHC converter patch (SimMHCPreConverter/SimMHCPostConverter)"
```

---

### Task 6: Wire into `SimulationTrainer` + `SimulationConfig`

**Files:**
- Modify: `torchtitan_npu/simulator/trainer.py`
- Modify: `tests/unit_tests/simulator/test_trainer.py`

**Interfaces:**
- Consumes: `apply_mhc_shims` (Task 5)
- Produces: `SimulationConfig.target_npu_device_type: str` field; narrows `_HARDWARE_DEPENDENT_CONVERTER_NAMES` to `frozenset({"npu_smla"})`.

- [ ] **Step 1: Write the failing test**

In `tests/unit_tests/simulator/test_trainer.py`, replace the existing
`test_strip_hardware_dependent_model_converters_removes_mhc_converters` test (MHC is no longer
stripped -- this is a required update, not a regression: Task 5 replaced strip-and-drop with
patch-and-keep for these two names) with:

```python
def test_strip_hardware_dependent_model_converters_only_removes_smla():
    # Updated expectation (was: also strips npu_mhc_pre/npu_mhc_post). MHC is no longer
    # stripped -- SimulationTrainer now installs SimMHCPreConverter/SimMHCPostConverter via
    # apply_mhc_shims() instead (Task 5), so npu_mhc_pre/npu_mhc_post stay in the converters
    # list and get a real (shim) implementation rather than being dropped to the base class.
    config = SimpleNamespace(
        model_converters=SimpleNamespace(
            converters=[
                _fake_converter_config("npu_rms_norm"),
                _fake_converter_config("npu_mhc_pre"),
                _fake_converter_config("npu_mhc_post"),
                _fake_converter_config("npu_smla"),
                _fake_converter_config("npu_gmm"),
            ]
        )
    )
    _strip_hardware_dependent_model_converters(config)
    remaining_names = {c._owner._model_config.name for c in config.model_converters.converters}
    assert remaining_names == {"npu_rms_norm", "npu_mhc_pre", "npu_mhc_post", "npu_gmm"}
```

Also add:

```python
def test_simulation_config_defaults_target_npu_device_type_to_non_a5():
    from torchtitan_npu.simulator.trainer import SimulationConfig

    config = SimulationConfig(output_dir="./out")
    assert config.target_npu_device_type == "non_a5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit_tests/simulator/test_trainer.py -v -k "only_removes_smla or target_npu_device_type"`
Expected: FAIL -- `only_removes_smla` fails on the `assert remaining_names == ...` line (current
code still strips MHC); `target_npu_device_type` fails with `TypeError: __init__() got an
unexpected keyword argument` (field doesn't exist yet).

- [ ] **Step 3: Modify `trainer.py`**

View the current file first to find the exact locations:

```bash
grep -n "_HARDWARE_DEPENDENT_CONVERTER_NAMES\|class SimulationConfig\|class SimulationTrainer\|def __init__" torchtitan_npu/simulator/trainer.py
```

Make these three edits to `torchtitan_npu/simulator/trainer.py`:

1. Add the import near the other simulator imports (after the `RankTable`/`build_rank_table` import line):

```python
from torchtitan_npu.simulator.hardware_shims.mhc_converter import apply_mhc_shims
```

2. Change the `_HARDWARE_DEPENDENT_CONVERTER_NAMES` constant:

```python
_HARDWARE_DEPENDENT_CONVERTER_NAMES = frozenset({"npu_smla"})
```

(was `frozenset({"npu_mhc_pre", "npu_mhc_post", "npu_smla"})` -- update the explanatory comment
block directly above it to remove the two bullet points about `npu_mhc_pre`/`npu_mhc_post`,
replacing them with: `# npu_mhc_pre/npu_mhc_post: no longer stripped -- SimulationTrainer.__init__`
`# now calls apply_mhc_shims() (torchtitan_npu.simulator.hardware_shims.mhc_converter) to install`
`# SimHcPre/SimHcHead/SimHcPost instead, preserving the real op names in the captured graph. See`
`# docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md.`)

3. Add the `target_npu_device_type` field to `SimulationConfig` (find its `@dataclass` definition
   and add the field with a default, after `output_dir`):

```python
    target_npu_device_type: str = "non_a5"
```

4. In `SimulationTrainer.__init__`, find the line that calls
   `_strip_hardware_dependent_model_converters(config)` and add a call to `apply_mhc_shims()`
   immediately before it:

```python
        apply_mhc_shims()
        _strip_hardware_dependent_model_converters(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit_tests/simulator/test_trainer.py -v`
Expected: all tests PASS.

Then run the full simulator suite to check for regressions:

Run: `python3 -m pytest tests/unit_tests/simulator/ tests/smoke_tests/simulator/ -q`
Expected: all PASS (previous 108 + this task's new tests; some may report `ModuleNotFoundError`
for `torch_npu` if run outside the container per this repo's existing sandbox limitation --
acceptable, matches the documented convention for `test_rank_table.py`/pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/trainer.py tests/unit_tests/simulator/test_trainer.py
git commit -m "feat(simulator): wire apply_mhc_shims into SimulationTrainer, narrow strip-out to npu_smla only"
```

---

### Task 7: Small-scale container validation spike

**Files:** none (validation only, no code changes)

**Interfaces:** none

- [ ] **Step 1: Run the full simulator test suite in the CANN container**

```bash
sudo docker exec titan-npu-sim-e2e bash -c "
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=\"/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:\${LD_LIBRARY_PATH}\"
cd /mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator
python3 -m pytest -v tests/unit_tests/simulator/ tests/smoke_tests/simulator/ 2>&1 | tail -40
"
```

Expected: all tests pass (previous count + this plan's new tests, no `torch_npu`-related
skips this time since the container has real `torch_npu`).

- [ ] **Step 2: Run the 16-layer smoke config and confirm MHC's real op names appear**

```bash
sudo docker exec titan-npu-sim-e2e bash -c "
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=\"/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:\${LD_LIBRARY_PATH}\"
cd /mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator
NGPU=16 LOCAL_RANK=0 WORLD_SIZE=16 RANK=0 COMM_MODE=fake_backend \
python3 -m torchtitan_npu.entry \
  --module torchtitan_npu.simulator \
  --config deepseek_v4_pro_simulate_16_layers \
  --comm.mode=fake_backend --training.steps=1 \
  --hf_assets_path=./tests/assets/tokenizer/deepseekv3_tokenizer
echo EXIT_CODE=\$?
grep -c 'triton.hc_pre_fwd\|triton.hc_pre_bmm_forward\|triton.hc_pre_bwd\|triton.hc_pre_bmm_backward' \
  simulator_output/deepseek_v4_pro_16_layers/compute_graph.dot
"
```

Expected: `EXIT_CODE=0`, and the `grep -c` count is > 0 (confirms `SimHcPre`'s real op names now
appear in the captured/visualized graph where the base-class `HcPre`'s unrelated op sequence --
`aten.rsqrt`/`aten.mean`, etc. -- used to appear instead).

- [ ] **Step 3: If Step 2 fails, debug before proceeding**

If the run crashes: read the stack trace, identify the exact failing call (most likely a shape
mismatch between what `SimHcPre`/`SimHcHead`/`SimHcPost` produce and what the surrounding real
model code -- e.g. the `TransformerBlock`'s residual-add logic -- expects downstream). Fix the
shape formula in `mhc_shim.py` (re-verify against the exact production call site in
`torchtitan_npu/models/deepseek_v4/model.py` and `torchtitan_npu/converters/kernels/mhc_prepost.py`
that invokes `HcPre`/`HcHead`/`HcPost`), re-run this task's Step 1 (full test suite) to catch any
regressions, then retry Step 2. Do not proceed to Task 8 until Step 2 passes cleanly.

- [ ] **Step 4: Commit any fixes made during debugging**

```bash
git add -A
git commit -m "fix(simulator): correct MHC shim shape formula found via 16-layer container validation"
```

(Skip this step if Step 2 passed on the first try with no code changes needed.)

---

### Task 8: Full container re-validation (16-layer + 61-layer)

**Files:** none (validation only)

**Interfaces:** none

- [ ] **Step 1: Re-run the 16-layer smoke config, save `summary.txt`**

```bash
sudo docker exec titan-npu-sim-e2e bash -c "
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=\"/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:\${LD_LIBRARY_PATH}\"
cd /mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator
rm -rf simulator_output/deepseek_v4_pro_16_layers
NGPU=16 LOCAL_RANK=0 WORLD_SIZE=16 RANK=0 COMM_MODE=fake_backend \
python3 -m torchtitan_npu.entry \
  --module torchtitan_npu.simulator \
  --config deepseek_v4_pro_simulate_16_layers \
  --comm.mode=fake_backend --training.steps=1 \
  --hf_assets_path=./tests/assets/tokenizer/deepseekv3_tokenizer
echo EXIT_CODE=\$?
cat simulator_output/deepseek_v4_pro_16_layers/summary.txt
"
```

Expected: `EXIT_CODE=0`. Compare the "Unrecognized op types" list against the pre-this-plan
baseline (94 entries) -- confirm it no longer contains any base-`HcPre`/`HcHead`/`HcPost`-only
ops that MHC's real path wouldn't use (e.g. if `aten.rsqrt.default`/`aten.mean.default` were only
reachable via the base class's `forward`, and are absent from other model code paths, they should
now disappear from this list; if they're also used elsewhere in the model they may still appear
-- that's expected and fine, this check is about confirming the *shim* op names appear, not about
the unrecognized-list shrinking to a specific count).

- [ ] **Step 2: Re-run the 61-layer acceptance config**

```bash
sudo docker exec titan-npu-sim-e2e bash -c "
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=\"/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:\${LD_LIBRARY_PATH}\"
cd /mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator
rm -rf simulator_output/deepseek_v4_pro_61_layers
NGPU=384 LOCAL_RANK=0 WORLD_SIZE=384 RANK=0 COMM_MODE=fake_backend \
python3 -m torchtitan_npu.entry \
  --module torchtitan_npu.simulator \
  --config deepseek_v4_pro_simulate_61_layers \
  --comm.mode=fake_backend --training.steps=1 \
  --hf_assets_path=./tests/assets/tokenizer/deepseekv3_tokenizer
echo EXIT_CODE=\$?
cat simulator_output/deepseek_v4_pro_61_layers/summary.txt
"
```

Expected: `EXIT_CODE=0` (takes ~8-9 minutes, matching the original acceptance run's timing).
Confirm `RankTable`: `world_size=384`, `dim_degrees["ep"]=192`, `pp=tp=cp=1` (same as the
original acceptance result -- MHC's shim should not change parallelism/RankTable at all, only
the op-name granularity of HcPre/HcHead/HcPost nodes). Confirm forward/backward/optimizer node
counts are in the same order of magnitude as the original acceptance run (50,027 / 137,299 /
3,508) -- some difference is expected (MHC's real op sequence has a different node count than
the base class's), but a wildly different order of magnitude would indicate a bug.

- [ ] **Step 3: Copy the updated output out of the container for review**

```bash
HOST_DIR="/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/simulator_review_output/deepseek_v4_pro_61_layers_post_mhc_shim"
mkdir -p "$HOST_DIR"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/compute_graph.dot" "$HOST_DIR/"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/summary.txt" "$HOST_DIR/"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/trace.html" "$HOST_DIR/"
sudo chown -R $(id -u):$(id -g) "$HOST_DIR"
```

- [ ] **Step 4: Update the design doc with final validated results**

Add a short "§8. 验证结果" section to
`docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md` recording: the exact
container commands run, exit codes, and the confirmed presence of `triton.hc_pre_fwd`/
`triton.hc_pre_bmm_forward`/etc. in the 61-layer output, plus the forward/backward/optimizer
node counts observed (for future comparison if this shim is extended or modified).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md
git commit -m "docs(simulator): record MHC shim validation results (16-layer + 61-layer re-run)"
git push origin feat/npu-simulator  # or the current feature branch
```

---

## Explicitly out of scope for this plan

- **SMLA** (`npu_smla`): remains stripped, per the design doc's §6. A follow-up plan should
  repeat this same pattern (`record_synthetic_op` + shim classes + reversible converter patch)
  for `SparseAttention`/`LiCompute`/`LiLoss`, after a dedicated shape-formula verification pass
  against `torchtitan_npu/converters/kernels/npu_smla.py`'s `build_op(...)`-based call sites.
- **`target_npu_device_type == "A5"`**: the field exists (Task 6) but `SimHcPre`/`SimHcHead`/
  `SimHcPost` do not yet branch on it -- they always record the non-A5 (Triton) op names
  regardless of the config value, since the "A5" path's real shapes
  (`hc_before_norm`/`inv_rms`/`sum_out`/`norm_out`) were never verified (design doc §6). A
  follow-up task should either implement the A5 branch properly or make `SimHcPre.__init__`
  raise/warn clearly when `target_npu_device_type == "A5"` is requested but unimplemented.
