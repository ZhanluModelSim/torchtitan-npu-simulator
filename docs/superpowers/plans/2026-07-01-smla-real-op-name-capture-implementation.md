# SMLA Real-Op-Name Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current blanket strip-out of the `npu_smla` model converter with simulator-only
shim classes for `SparseAttention`/`LiCompute`/`LiLoss`, so the captured L0 op graph shows the *real*
ACLNN op names that would run in production (`aclnn.npu_sparse_attn_sharedkv` etc.) instead of the
base classes' manually-decomposed `matmul`/`softmax`/`einsum`/`topk` op sequence.

**Architecture:** Reuses `OpDispatchCapture.record_synthetic_op()` (already built for MHC, zero
changes needed) via `torch.autograd.Function` shim classes (`_SimSparseAttnFn`/
`_SimLightningIndexerFn`/`_SimLiLossFn`) that record the real op names + analytically-derived shapes
(verified directly against the real production `ops/aclnn/*/binding.cpp` C++ sources) without ever
invoking the real ACLNN extensions. A reversible class-attribute patch
(`apply_smla_shims()`/`unapply_smla_shims()`, mirroring the already-built `apply_mhc_shims()`)
installs the shims in place of the real `NpuSMLAConverter`, only under simulation.

**Tech Stack:** Python 3.12, PyTorch 2.12 (`torch.autograd.Function`, `torch.empty(device="meta")`), pytest.

## Global Constraints

- Zero modification to any existing production file (`torchtitan_npu/converters/`,
  `torchtitan_npu/models/`, `torchtitan_npu/ops/` all stay byte-for-byte unchanged) -- this feature
  is purely additive, consistent with this repo's side-loaded-package convention.
- All new simulator code lives under `torchtitan_npu/simulator/`.
- Every new/modified test must pass in this sandbox (no `torch_npu` installed) **except** tests that
  import `torchtitan_npu.models.deepseek_v4.model` (this sandbox's plain `torchtitan` package is not
  the exact pinned commit and transitively fails with an unrelated `AttributeError` about
  `TokenReorderer` -- the exact same, already-documented root cause as the MHC plan's tests). These
  tests are expected to fail with that specific error in this sandbox and pass in the real CANN
  container (Task 6 is their first real signal). Use the scratch-verification workaround (a
  throwaway, NOT-committed script using fake stand-in parent objects) to gain real confidence before
  committing the real test file -- see the MHC plan
  (`docs/superpowers/plans/2026-07-01-mhc-real-op-name-capture-implementation.md`) for the exact
  established pattern.
- STRICT SCOPE BOUNDARY per task: modify/create ONLY the files each task's brief lists. Never touch
  `torchtitan_npu/__init__.py` or any file outside `torchtitan_npu/simulator/` -- two earlier tasks in
  the MHC plan's execution had implementers violate this to work around the sandbox limitation above,
  and both had to be reverted after review caught them. Use the scratch-verification workaround
  instead.
- Follow this repo's TDD convention: write the failing test first, run it, then implement.
- Commit after each task passes its tests.
- All shim classes MUST subclass their real base class (`SimNpuSparseAttention(SparseAttention)`, not
  `SimNpuSparseAttention(nn.Module)`) -- this is a hard requirement, not style: `torchtitan`'s
  `BaseModel.verify_module_protocol()` asserts `isinstance(mod, torchtitan.protocols.module.Module)`
  for every submodule, and only subclassing the real base class satisfies this with zero extra code.
  This was a real bug found via container validation during the MHC plan's Task 7 -- avoid repeating
  it here by building it into every task from the start.
- All shim `__init__` methods use `self.__dict__.update(parent.__dict__)` (no `super().__init__()`
  call) -- matches the real `NpuSparseAttention`/`NpuLiCompute`/`NpuLiLoss` classes' own
  `__init__` pattern exactly (`torchtitan_npu/converters/kernels/npu_smla.py:1330-1338`, `1362-1370`,
  and mirrored for `NpuLiLoss` at `1534-1543`).

---

## Reference: exact shapes and real op names (verified against real production C++ sources)

`B`=batch, `S`=seqlen, `N`=`n_heads`, `D`=`head_dim`, `N_i`=`index_n_heads`, `D_i`=`index_head_dim`,
`K`=`index_topk`, `R`=`compress_ratios[layer_id]` (one of `1`, `4`, `128`).

### `SparseAttention` → `SimNpuSparseAttention`

Fixed public contract (`SparseAttention.forward`, `torchtitan_npu/models/deepseek_v4/model.py:475-523`
-- any replacement class must keep this exact I/O):
```
forward(query_states[B,S,N,D], kv_states[B,S,D], attn_sink[N],
        kv_compress[B,S//R,D]|None, compress_topk_idxs[B,S,K]|None) -> attn_output[B,S,N,D]
```

Real call chain (`NpuSparseAttention.forward` → `npu_sparse_attn_shared_kv` wrapper →
`SparseAttnSharedKV(torch.autograd.Function)`, `torchtitan_npu/converters/kernels/npu_smla.py:
1330-1359`, `537-588`, `378-485` forward / `487-534` backward):

| step | real op name | when | inputs | outputs |
|---|---|---|---|---|
| A | `aclnn.npu_sparse_attn_sharedkv_metadata` | forward | `query` (used only to derive shape) | `metadata[1024]` int32 |
| B | `aclnn.npu_sparse_attn_sharedkv` | forward | `query[B,S,N,D]` (bf16), `ori_kv[B,S,1,D]` (bf16, = `kv_states.unsqueeze(2)`), `cmp_kv[B,S//R,1,D]`\|omitted (bf16, = `kv_compress.unsqueeze(2)`, only when not None), `cmp_sparse_indices[B,S,1,K]`\|omitted (int32, = `compress_topk_idxs.unsqueeze(2)`, only when `R==4`), `sinks[N]` (fp32), `metadata[1024]` | `result[B,S,N,D]` (bf16), `softmax_lse[B,S,N,1]` (fp32) |
| C (bwd) | `aclnn.npu_sparse_attn_sharedkv_grad` | backward | `query`, `ori_kv`, `cmp_kv`\|omitted, `result`, `softmax_lse`, `sinks`, `grad_result[B,S,N,D]` | `dquery[B,S,N,D]`, `dori_kv[B,S,1,D]`, `dcmp_kv[B,S//R,1,D]`\|omitted, `dsinks[N]` |

Shape sources verified directly against `torchtitan_npu/ops/aclnn/sparse_attn_sharedkv/binding.cpp`:
`attnOutput = at::empty(query.sizes(), ...)` (line 42); `lse_sizes.back()=1` on `query.sizes()`
(lines 45-47); `metadata = at::empty(1024, ...)` (line 20); `dQuery/dOriKv/dSinks = at::empty(
query/oriKv/sinks.sizes(), ...)` (lines 70-72); `dCmpKv = at::empty(cmpKv.sizes(), ...)` only when
`cmpRatio > 1` (lines 74-78).

`SparseAttnSharedKV.forward` returns **only `result`** (a single tensor, not a tuple) --
`softmax_lse` is saved via `ctx.save_for_backward` for internal backward use, never returned to the
caller (`npu_smla.py:483-485`: `return result`, confirmed directly; `npu_sparse_attn_shared_kv`
wrapper does `return output.contiguous()` on this single tensor, `npu_smla.py:588`).

### `LiCompute` → `SimNpuLiCompute`

Fixed public contract (`LiCompute.forward`, `model.py:537-556`; `LiCompute`/`LiLoss` only exist on
`R==4` layers, `InnerAttention.__init__`, `model.py:682-691`):
```
forward(q_indexer[B,S,N_i,D_i], k_indexer[B,S//4,D_i], weights[B,S,N_i],
        seqlen:int, offset:int) -> (compress_topk_idxs[B,S,K] int32, index_score[B,S,K])
```

Real call (`NpuLiCompute.forward`, `npu_smla.py:1372-1406`): calls `_li_op.npu_lightning_indexer(...)`
**directly, with no `torch.autograd.Function` wrapper** (confirmed: no autograd.Function anywhere in
this call path, unlike `SparseAttention`'s `SparseAttnSharedKV`). This never crashes in real training
because `compress_topk_idxs` is int32 (non-differentiable) and `index_score`'s gradient flows through
the *separate* `LiLoss` path via detached tensors (`InnerAttention.forward:714-726`).

| step | real op name | inputs | outputs |
|---|---|---|---|
| D | `aclnn.npu_lightning_indexer` | `query[B,S,N_i,D_i]` (bf16, = `q_indexer.to(bf16)`), `key[B,S//4,1,D_i]` (bf16, = `k_indexer.to(bf16).unsqueeze(2)`), `weights[B,S,N_i]` (bf16) | `sparse_indices[B,S,1,K]` int32, `sparse_values[B,S,1,K]` bf16 → each `.squeeze(2)` → `[B,S,K]` |

Shape source: `torchtitan_npu/ops/aclnn/lightning_indexer/binding.cpp:19-26`:
`sparse_indices/sparse_values = at::empty({B, S1, N2=1, sparse_count=K}, ...)`.

**The shim must wrap this in its own `torch.autograd.Function`** (the real implementation has none)
purely for gradient-graph connectivity and consistent backward-phase tagging -- backward returns
`None` for every input (mirrors the A5-path `LightningIndexer.backward`'s `_none_grads(4)`,
`npu_smla.py:1059-1061`).

`_add_offset_to_valid_sparse_indices` (`npu_smla.py:301-310`, applied to `compress_topk_idxs` after
the real op call) is pure PyTorch (`torch.where`/`+`/`.eq`), has no hardware dependency, and is
imported directly (read-only) rather than reimplemented.

### `LiLoss` → `SimNpuLiLoss`

Fixed public contract (`LiLoss.forward`, `model.py:308-337`; `NpuLiLoss.forward` same positional
order, `npu_smla.py:1545-1573`):
```
forward(q[B,S,N,D] detached, kv[B,S,D] detached, kv_compress[B,S//4,D] detached,
        attn_sink[N] (unused), q_indexer[B,S,N_i,D_i], k_indexer[B,S//4,D_i],
        weights[B,S,N_i], sparse_indices[B,S,K], indexer_score[B,S,K] (unused),
        attention_masks (unused), offset:int (unused)) -> loss (scalar fp32)
```

Real call chain is a **deferred-computation pattern**: `NpuLiLoss.forward` → `npu_sparse_lightning_
indexer_grad_kl_loss` → `SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function)`
(`npu_smla.py:1545-1573`, `1499-1531`, `1409-1495`):

| step | real op name | when | inputs | outputs |
|---|---|---|---|---|
| (none) | -- | forward | -- | `loss = torch.zeros(1, dtype=torch.float32, device=query.device)[0]` -- **no hardware call in forward**, confirmed directly at `npu_smla.py:1441`: `return torch.zeros(1, dtype=torch.float32, device=query.device)[0]` |
| E | `aclnn.npu_sparse_lightning_indexer_grad_kl_loss` | backward | `query[B,S,N,D]` (= `q`), `key[B,S//4,1,D]` (= `kv_compress.unsqueeze(2)`), `query_index[B,S,N_i,D_i]` (= `q_indexer`), `key_index[B,S//4,1,D_i]` (= `k_indexer.unsqueeze(2)`), `weights[B,S,N_i]`, `sparse_indices[B,S,1,K]` (= `sparse_indices.unsqueeze(2)`) | `d_query_index[B,S,N_i,D_i]`, `d_key_index[B,S//4,1,D_i]`, `d_weight[B,S,N_i]`, `loss[1]` fp32 |

Shape source: `torchtitan_npu/ops/aclnn/sparse_lightning_indexer_grad_kl_loss/binding.cpp:23-26`:
`d_query_index/d_key_index/d_weight = at::zeros(query_index/key_index/weight.sizes(), ...)`,
`loss = at::zeros({1}, ...)`.

`SparseLightningIndexerGradKLLossWrapper.forward` (`npu_smla.py:1412-1441`) takes 14 positional args
and does `ctx.save_for_backward(query, key, query_index, key_index, weights, sparse_indices)`
(6 tensors); `backward` (`npu_smla.py:1444-1495`) returns 14 grad values:
`(None, None, d_query_index, d_key_index, d_weights, None, None, None, None, None, None, None, None)`
(`query`/`key` are detached inputs → `None`; `sparse_indices` int32 → `None`; remaining 8 are scalar
config args → `None`). The shim simplifies this to a 7-arg autograd.Function (dropping the scalar
config args that don't affect shape): `(query, key, query_index, key_index, weights, sparse_indices,
module_path)` → backward returns 7 values: `(None, None, d_query_index, d_key_index, d_weights, None,
None)`.

**Interaction with the existing `meta_env._patch_li_loss_to_skip_buggy_einsum` patch**: that patch
replaces `LiLoss.forward` (the base class method) to work around a real pre-existing shape bug in
`LiLoss._current_selected_attn_dist`, never hit in real production (which always uses `NpuLiLoss`).
Since `SimNpuLiLoss(LiLoss)` defines its **own** `forward`, Python's MRO means instances of
`SimNpuLiLoss` always resolve `.forward()` to `SimNpuLiLoss.forward`, never to the patched
`LiLoss.forward` -- **no change to `meta_env.py` is needed or wanted.**

---

### Task 1: `SimNpuSparseAttention` hardware shim

**Files:**
- Create: `torchtitan_npu/simulator/hardware_shims/smla_shim.py`
- Test: `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py`

**Interfaces:**
- Consumes: `torchtitan_npu.simulator.capture.dispatch_capture.get_active_capture()` (already exists
  from the MHC plan's Task 1, zero changes needed).
- Produces: `SimNpuSparseAttention(SparseAttention)` with `__init__(self, parent: SparseAttention)`
  and `forward(self, query_states, kv_states, attn_sink, kv_compress=None, compress_topk_idxs=None) ->
  Tensor` (matches `SparseAttention.forward`'s exact signature).

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model, SparseAttention
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuSparseAttention


def _build_sim_sparse_attention(B=2, S=3, N=4, D=8, R=4, K=5):
    args = DeepSeekV4Model.Config(n_heads=N, head_dim=D, compress_ratios=(R,), window_size=2, n_layers=1)
    parent = SparseAttention(SparseAttention.Config(layer_id=0, args=args))
    shim = SimNpuSparseAttention(parent)
    tensors = {
        "query_states": torch.randn(B, S, N, D, requires_grad=True),
        "kv_states": torch.randn(B, S, D, requires_grad=True),
        "attn_sink": torch.randn(N, requires_grad=True),
    }
    if R != 1:
        tensors["kv_compress"] = torch.randn(B, S // R, D, requires_grad=True)
    if R == 4:
        tensors["compress_topk_idxs"] = torch.randint(0, S, (B, S, K), dtype=torch.int32)
    return shim, tensors


def test_sim_sparse_attention_forward_returns_correct_shape_r1():
    shim, t = _build_sim_sparse_attention(B=2, S=3, N=4, D=8, R=1)
    y = shim(t["query_states"], t["kv_states"], t["attn_sink"])
    assert y.shape == (2, 3, 4, 8)


def test_sim_sparse_attention_forward_returns_correct_shape_r4():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    y = shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
    assert y.shape == (2, 8, 4, 8)


def test_sim_sparse_attention_records_real_op_names():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    capture = OpDispatchCapture()
    with capture:
        shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "aclnn.npu_sparse_attn_sharedkv_metadata" in raw_names
    assert "aclnn.npu_sparse_attn_sharedkv" in raw_names


def test_sim_sparse_attention_backward_propagates_gradient_and_records_grad_op():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        y = shim(t["query_states"], t["kv_states"], t["attn_sink"], t["kv_compress"], t["compress_topk_idxs"])
        phase_box["value"] = "backward"
        y.sum().backward()
    assert t["query_states"].grad is not None
    assert t["query_states"].grad.shape == t["query_states"].shape
    assert t["kv_states"].grad is not None
    assert t["kv_states"].grad.shape == t["kv_states"].shape
    assert t["attn_sink"].grad is not None
    assert t["attn_sink"].grad.shape == t["attn_sink"].shape
    assert t["kv_compress"].grad is not None
    assert t["kv_compress"].grad.shape == t["kv_compress"].shape
    nodes = capture.build_nodes()
    bwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "backward"}
    assert "aclnn.npu_sparse_attn_sharedkv_grad" in bwd_names


def test_sim_sparse_attention_works_on_meta_device():
    shim, t = _build_sim_sparse_attention(B=2, S=8, N=4, D=8, R=4, K=5)
    meta_t = {k: (v.detach().to("meta").requires_grad_(True) if v.dtype != torch.int32 else v.detach().to("meta")) for k, v in t.items()}
    y = shim(meta_t["query_states"], meta_t["kv_states"], meta_t["attn_sink"], meta_t["kv_compress"], meta_t["compress_topk_idxs"])
    assert y.device.type == "meta"
    y.sum().backward()
    assert meta_t["query_states"].grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.hardware_shims.smla_shim'`

- [ ] **Step 3: Implement `smla_shim.py`**

Create `torchtitan_npu/simulator/hardware_shims/smla_shim.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-only shims for SMLA's SparseAttention/LiCompute/LiLoss (`torchtitan_npu.converters.
kernels.npu_smla`'s NpuSparseAttention/NpuLiCompute/NpuLiLoss in production). Records the real
production ACLNN op names (`aclnn.npu_sparse_attn_sharedkv` etc., verified against the real
ops/aclnn/*/binding.cpp C++ sources) into the active OpDispatchCapture, with analytically-derived
shapes -- never invoking the real JIT-compiled ACLNN extensions. See design doc for exact shape
formulas: docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md."""

from __future__ import annotations

import torch

from torchtitan_npu.converters.kernels.npu_smla import _add_offset_to_valid_sparse_indices
from torchtitan_npu.models.deepseek_v4.model import LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


def _record(raw_op_type: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], module_path: str) -> None:
    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(raw_op_type, inputs=inputs, outputs=outputs, module_path=module_path)


class _SimSparseAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, module_path):  # noqa: ANN001
        B, S, N, D = query.shape
        dtype = query.dtype

        metadata = torch.empty(1024, dtype=torch.int32, device=query.device)
        _record("aclnn.npu_sparse_attn_sharedkv_metadata", [query], [metadata], module_path)

        result = torch.empty((B, S, N, D), dtype=dtype, device=query.device)
        softmax_lse = torch.empty((B, S, N, 1), dtype=torch.float32, device=query.device)
        fwd_inputs = [query, ori_kv, sinks, metadata]
        if cmp_kv is not None:
            fwd_inputs.append(cmp_kv)
        if cmp_sparse_indices is not None:
            fwd_inputs.append(cmp_sparse_indices)
        _record("aclnn.npu_sparse_attn_sharedkv", fwd_inputs, [result, softmax_lse], module_path)

        ctx.save_for_backward(query, ori_kv, cmp_kv, result, softmax_lse, sinks)
        ctx.module_path = module_path
        return result

    @staticmethod
    def backward(ctx, grad_result):  # noqa: ANN001
        query, ori_kv, cmp_kv, result, softmax_lse, sinks = ctx.saved_tensors
        dquery = torch.empty_like(query)
        dori_kv = torch.empty_like(ori_kv)
        dsinks = torch.empty_like(sinks)
        dcmp_kv = torch.empty_like(cmp_kv) if cmp_kv is not None else None

        bwd_inputs = [query, ori_kv, result, softmax_lse, sinks, grad_result]
        if cmp_kv is not None:
            bwd_inputs.append(cmp_kv)
        bwd_outputs = [dquery, dori_kv, dsinks]
        if dcmp_kv is not None:
            bwd_outputs.append(dcmp_kv)
        _record("aclnn.npu_sparse_attn_sharedkv_grad", bwd_inputs, bwd_outputs, ctx.module_path)

        return dquery, dori_kv, dcmp_kv, None, dsinks, None


class SimNpuSparseAttention(SparseAttention):
    """Drop-in simulator replacement for `NpuSparseAttention`
    (`torchtitan_npu.converters.kernels.npu_smla`). Same forward() signature as base
    `SparseAttention`; never runs the real JIT-compiled ACLNN extension, only records the real
    op names + analytically-correct shapes.

    `__init__` mirrors `NpuSparseAttention.__init__(self, parent: SparseAttention)`'s exact
    `__dict__` shallow-copy pattern (`npu_smla.py:1330-1338`) -- required both for consistency
    and to satisfy torchtitan's `verify_module_protocol()` (subclassing `SparseAttention`
    transitively satisfies the `isinstance(mod, Module)` check with zero extra code)."""

    def __init__(self, parent: "SparseAttention") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, query_states, kv_states, attn_sink, kv_compress=None, compress_topk_idxs=None):
        if compress_topk_idxs is not None and compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)
        ori_kv = kv_states.unsqueeze(2).contiguous()
        cmp_kv = kv_compress.unsqueeze(2).contiguous() if kv_compress is not None else None
        cmp_sparse_indices = compress_topk_idxs.unsqueeze(2).contiguous() if self.compress_ratio == 4 else None
        module_path = self.__class__.__name__
        result = _SimSparseAttnFn.apply(
            query_states.contiguous(), ori_kv, cmp_kv, cmp_sparse_indices, attn_sink.float(), module_path
        )
        return result.contiguous()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/smla_shim.py tests/unit_tests/simulator/hardware_shims/test_smla_shim.py
git commit -m "feat(simulator): add SimNpuSparseAttention hardware shim for SMLA sparse attention"
```

---

### Task 2: `SimNpuLiCompute` hardware shim

**Files:**
- Modify: `torchtitan_npu/simulator/hardware_shims/smla_shim.py` (append)
- Test: `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py` (append)

**Interfaces:**
- Consumes: `_record` (Task 1, same file)
- Produces: `SimNpuLiCompute(LiCompute)` with `__init__(self, parent: LiCompute)` and
  `forward(self, q_indexer, k_indexer, weights, seqlen, offset) -> tuple[Tensor, Tensor]` (matches
  `LiCompute.forward`'s exact signature).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py`:

```python
from torchtitan_npu.models.deepseek_v4.model import LiCompute  # add to existing import line
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute  # add to existing import line


def _build_sim_li_compute(B=2, S=8, N_i=4, D_i=8, K=5, ratio=4):
    parent = LiCompute(LiCompute.Config(ratio=ratio, index_topk=K))
    shim = SimNpuLiCompute(parent)
    tensors = {
        "q_indexer": torch.randn(B, S, N_i, D_i, requires_grad=True),
        "k_indexer": torch.randn(B, S // ratio, D_i, requires_grad=True),
        "weights": torch.randn(B, S, N_i, requires_grad=True),
    }
    return shim, tensors


def test_sim_li_compute_forward_returns_correct_shapes():
    shim, t = _build_sim_li_compute(B=2, S=8, N_i=4, D_i=8, K=5, ratio=4)
    compress_topk_idxs, index_score = shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    assert compress_topk_idxs.shape == (2, 8, 5)
    assert index_score.shape == (2, 8, 5)


def test_sim_li_compute_records_real_op_name():
    shim, t = _build_sim_li_compute()
    capture = OpDispatchCapture()
    with capture:
        shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "aclnn.npu_lightning_indexer" in raw_names


def test_sim_li_compute_backward_does_not_raise_and_returns_none_grad():
    shim, t = _build_sim_li_compute()
    compress_topk_idxs, index_score = shim(t["q_indexer"], t["k_indexer"], t["weights"], seqlen=8, offset=0)
    # index_score is a real (non-detached) autograd node output -- summing and backpropagating
    # through it must not raise, even though the real op has no gradient (mirrors production,
    # where index_score's actual gradient flows through the separate LiLoss path instead).
    index_score.float().sum().backward()
    assert t["q_indexer"].grad is None  # real non-A5 implementation has no gradient for this path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v -k li_compute`
Expected: FAIL with `ImportError: cannot import name 'SimNpuLiCompute'`

- [ ] **Step 3: Implement `SimNpuLiCompute`**

Append to `torchtitan_npu/simulator/hardware_shims/smla_shim.py`:

```python
class _SimLightningIndexerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, weights, index_topk, module_path):  # noqa: ANN001
        B, S, N_i, D_i = query.shape
        K = index_topk
        sparse_indices = torch.empty((B, S, 1, K), dtype=torch.int32, device=query.device)
        sparse_values = torch.empty((B, S, 1, K), dtype=query.dtype, device=query.device)
        _record("aclnn.npu_lightning_indexer", [query, key, weights], [sparse_indices, sparse_values], module_path)
        return sparse_indices.squeeze(2), sparse_values.squeeze(2)

    @staticmethod
    def backward(ctx, grad_topk_idxs, grad_index_score):  # noqa: ANN001
        # The real non-A5 npu_lightning_indexer call has no autograd kernel (see design doc
        # §4.2): compress_topk_idxs is non-differentiable (int32), and index_score's real
        # gradient flows through the separate LiLoss path via detached tensors, never back
        # through LiCompute. Mirrors the A5 LightningIndexer.backward's all-None return.
        return None, None, None, None, None


class SimNpuLiCompute(LiCompute):
    """Drop-in simulator replacement for `NpuLiCompute`
    (`torchtitan_npu.converters.kernels.npu_smla`). The real non-A5 implementation calls
    `_li_op.npu_lightning_indexer` directly with no `torch.autograd.Function` wrapper --
    `_SimLightningIndexerFn` adds one here purely for gradient-graph connectivity and
    consistent backward-phase tagging, matching the pattern established for `SimHcHead` in
    the MHC plan."""

    def __init__(self, parent: "LiCompute") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(self, q_indexer: torch.Tensor, k_indexer: torch.Tensor, weights: torch.Tensor, seqlen: int, offset: int):
        q_indexer = q_indexer.to(torch.bfloat16)
        k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
        weights = weights.to(torch.bfloat16)
        module_path = self.__class__.__name__
        compress_topk_idxs, index_score = _SimLightningIndexerFn.apply(q_indexer, k_indexer, weights, self.index_topk, module_path)
        compress_topk_idxs = _add_offset_to_valid_sparse_indices(compress_topk_idxs, offset)
        return compress_topk_idxs, index_score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v`
Expected: all tests (Task 1's 5 + Task 2's 3 = 8) PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/smla_shim.py tests/unit_tests/simulator/hardware_shims/test_smla_shim.py
git commit -m "feat(simulator): add SimNpuLiCompute hardware shim for SMLA lightning-indexer"
```

---

### Task 3: `SimNpuLiLoss` hardware shim

**Files:**
- Modify: `torchtitan_npu/simulator/hardware_shims/smla_shim.py` (append)
- Test: `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py` (append)

**Interfaces:**
- Consumes: `_record` (Task 1, same file)
- Produces: `SimNpuLiLoss(LiLoss)` with `__init__(self, parent: LiLoss)` and
  `forward(self, q, kv, kv_compress, attn_sink, q_indexer, k_indexer, weights, sparse_indices,
  indexer_score, attention_masks, offset) -> Tensor` (matches `LiLoss.forward`'s exact signature).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_tests/simulator/hardware_shims/test_smla_shim.py`:

```python
from torchtitan_npu.models.deepseek_v4.model import LiLoss  # add to existing import line
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiLoss  # add to existing import line


def _build_sim_li_loss(B=2, S=8, N=4, D=8, N_i=4, D_i=8, K=5, ratio=4):
    parent = LiLoss(LiLoss.Config(n_heads=N, softmax_scale=0.1, compress_ratio=ratio, window_size=2, layer_id=0, n_layers=1))
    shim = SimNpuLiLoss(parent)
    tensors = {
        "q": torch.randn(B, S, N, D),
        "kv": torch.randn(B, S, D),
        "kv_compress": torch.randn(B, S // ratio, D),
        "attn_sink": torch.randn(N),
        "q_indexer": torch.randn(B, S, N_i, D_i, requires_grad=True),
        "k_indexer": torch.randn(B, S // ratio, D_i, requires_grad=True),
        "weights": torch.randn(B, S, N_i, requires_grad=True),
        "sparse_indices": torch.randint(0, S, (B, S, K), dtype=torch.int32),
        "indexer_score": torch.randn(B, S, K),
    }
    return shim, tensors


def test_sim_li_loss_forward_returns_zero_scalar_with_no_recorded_op():
    shim, t = _build_sim_li_loss()
    capture = OpDispatchCapture()
    with capture:
        loss = shim(
            t["q"], t["kv"], t["kv_compress"], t["attn_sink"], t["q_indexer"], t["k_indexer"],
            t["weights"], t["sparse_indices"], t["indexer_score"], None, 0,
        )
    assert loss.shape == ()
    assert loss.item() == 0.0
    nodes = capture.build_nodes()
    raw_names = {n.annotations.get("raw_op_type") for n in nodes.values()}
    assert "aclnn.npu_sparse_lightning_indexer_grad_kl_loss" not in raw_names  # not fired during forward


def test_sim_li_loss_backward_records_real_op_and_propagates_gradient():
    shim, t = _build_sim_li_loss()
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        loss = shim(
            t["q"], t["kv"], t["kv_compress"], t["attn_sink"], t["q_indexer"], t["k_indexer"],
            t["weights"], t["sparse_indices"], t["indexer_score"], None, 0,
        )
        phase_box["value"] = "backward"
        loss.backward()
    assert t["q_indexer"].grad is not None
    assert t["q_indexer"].grad.shape == t["q_indexer"].shape
    assert t["k_indexer"].grad is not None
    assert t["k_indexer"].grad.shape == t["k_indexer"].shape
    assert t["weights"].grad is not None
    assert t["weights"].grad.shape == t["weights"].shape
    nodes = capture.build_nodes()
    bwd_names = {n.annotations["raw_op_type"] for n in nodes.values() if n.annotations["phase"] == "backward"}
    assert "aclnn.npu_sparse_lightning_indexer_grad_kl_loss" in bwd_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v -k li_loss`
Expected: FAIL with `ImportError: cannot import name 'SimNpuLiLoss'`

- [ ] **Step 3: Implement `SimNpuLiLoss`**

Append to `torchtitan_npu/simulator/hardware_shims/smla_shim.py`:

```python
class _SimLiLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, query_index, key_index, weights, sparse_indices, module_path):  # noqa: ANN001
        ctx.save_for_backward(query, key, query_index, key_index, weights, sparse_indices)
        ctx.module_path = module_path
        # Mirrors the real SparseLightningIndexerGradKLLossWrapper.forward exactly: no
        # hardware call here, just a deferred zero placeholder -- the real kernel only fires
        # in backward (npu_smla.py:1441: "Return dummy loss during fwd, real operation will
        # be postponed to bwd, to avoid redundant computation ... in case where activation
        # checkpointing is enabled").
        return torch.zeros((), dtype=torch.float32, device=query.device)

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001
        query, key, query_index, key_index, weights, sparse_indices = ctx.saved_tensors
        d_query_index = torch.empty_like(query_index)
        d_key_index = torch.empty_like(key_index)
        d_weights = torch.empty_like(weights)
        loss = torch.empty((1,), dtype=torch.float32, device=query.device)
        _record(
            "aclnn.npu_sparse_lightning_indexer_grad_kl_loss",
            [query, key, query_index, key_index, weights, sparse_indices],
            [d_query_index, d_key_index, d_weights, loss],
            ctx.module_path,
        )
        return None, None, d_query_index, d_key_index, d_weights, None, None


class SimNpuLiLoss(LiLoss):
    """Drop-in simulator replacement for `NpuLiLoss`
    (`torchtitan_npu.converters.kernels.npu_smla`). Real implementation is a "deferred
    computation" pattern: forward returns a zero scalar immediately with no hardware call;
    the real ACLNN kernel only fires in backward. `_SimLiLossFn` replicates this exactly.

    Note: subclassing `LiLoss` and defining `forward` here means Python's MRO always resolves
    a `SimNpuLiLoss` instance's `.forward()` to this method, never to the real base `LiLoss.
    forward` (even though `meta_env._patch_li_loss_to_skip_buggy_einsum` patches that base
    method elsewhere) -- no interaction or conflict with that existing patch."""

    def __init__(self, parent: "LiLoss") -> None:
        self.__dict__.update(parent.__dict__)

    def forward(
        self, q, kv, kv_compress, attn_sink, q_indexer, k_indexer, weights,
        sparse_indices, indexer_score, attention_masks, offset,
    ):
        if sparse_indices.dtype != torch.int32:
            sparse_indices = sparse_indices.to(torch.int32)
        key = kv_compress.unsqueeze(2)
        key_index = k_indexer.unsqueeze(2)
        sparse_indices = sparse_indices.unsqueeze(2)
        module_path = self.__class__.__name__
        loss = _SimLiLossFn.apply(q, key, q_indexer, key_index, weights, sparse_indices, module_path)
        self.save_loss(loss)
        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_shim.py -v`
Expected: all tests (Task 1's 5 + Task 2's 3 + Task 3's 2 = 10) PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/smla_shim.py tests/unit_tests/simulator/hardware_shims/test_smla_shim.py
git commit -m "feat(simulator): add SimNpuLiLoss hardware shim for SMLA lightning-indexer KL loss"
```

---

### Task 4: `SimSMLAConverter` + apply/unapply wiring

**Files:**
- Create: `torchtitan_npu/simulator/hardware_shims/smla_converter.py`
- Test: `tests/unit_tests/simulator/hardware_shims/test_smla_converter.py`

**Interfaces:**
- Consumes: `SimNpuSparseAttention`, `SimNpuLiCompute`, `SimNpuLiLoss` (Tasks 1-3);
  `SparseAttention`, `LiCompute`, `LiLoss` from `torchtitan_npu.models.deepseek_v4.model`;
  `NpuSMLAModelConfig` from `torchtitan_npu.converters.kernels.npu_smla`; `replace_module_with_name`
  from `torchtitan_npu.converters.convert_utils`.
- Produces: `apply_smla_shims() -> None`, `unapply_smla_shims() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/simulator/hardware_shims/test_smla_converter.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.converters.kernels.npu_smla import NpuSMLAModelConfig
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model, LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.hardware_shims.smla_converter import apply_smla_shims, unapply_smla_shims
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute, SimNpuLiLoss, SimNpuSparseAttention


class _FakeModelSpec:
    name = "deepseek_v4"


def test_apply_smla_shims_replaces_converter_target_class():
    original = NpuSMLAModelConfig.model_converter
    try:
        apply_smla_shims()
        assert NpuSMLAModelConfig.model_converter is not original
    finally:
        unapply_smla_shims()
        assert NpuSMLAModelConfig.model_converter is original


def test_applied_smla_converter_replaces_all_three_submodule_types():
    apply_smla_shims()
    try:
        args = DeepSeekV4Model.Config(n_heads=4, head_dim=8, compress_ratios=(4,), window_size=2, n_layers=1)
        model = nn.Sequential()
        model.add_module("sparse_attn", SparseAttention(SparseAttention.Config(layer_id=0, args=args)))
        model.add_module("li_compute", LiCompute(LiCompute.Config(ratio=4, index_topk=5)))
        model.add_module("li_loss", LiLoss(LiLoss.Config(n_heads=4, softmax_scale=0.1, compress_ratio=4, window_size=2, layer_id=0, n_layers=1)))
        converter = NpuSMLAModelConfig.model_converter(_FakeModelSpec())
        converter.convert(model)
        assert isinstance(model.sparse_attn, SimNpuSparseAttention)
        assert isinstance(model.li_compute, SimNpuLiCompute)
        assert isinstance(model.li_loss, SimNpuLiLoss)
    finally:
        unapply_smla_shims()


def test_unapply_is_idempotent_when_not_applied():
    unapply_smla_shims()  # must not raise even if apply_smla_shims was never called
    unapply_smla_shims()  # calling twice must also not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_converter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchtitan_npu.simulator.hardware_shims.smla_converter'`

- [ ] **Step 3: Implement `smla_converter.py`**

First, confirm the exact registered converter class name via:
```bash
grep -n "register_model_converter(\"npu_smla\")" -A 2 torchtitan_npu/converters/kernels/npu_smla.py
```
Expected output: `@register_model_converter("npu_smla")` immediately followed by
`class NpuSMLAModelConfig(ModelCustomConfig):` (confirmed at `npu_smla.py:1639-1640`).

Create `torchtitan_npu/simulator/hardware_shims/smla_converter.py`:

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reversible class-attribute patch that swaps the `npu_smla` model converter's target
implementation class for the simulator's shape-only shims (SimNpuSparseAttention/
SimNpuLiCompute/SimNpuLiLoss), instead of SimulationTrainer stripping this converter out
entirely. Mirrors mhc_converter.py's apply_mhc_shims()/unapply_mhc_shims() exactly --
NpuSMLAModelConfig is the real, already-registered converter-config class from
torchtitan_npu.converters.kernels.npu_smla (zero modification to that file: this module only
reassigns its `model_converter` class attribute at runtime, under simulation)."""

from __future__ import annotations

import torch.nn as nn

from torchtitan_npu.converters.convert_utils import replace_module_with_name
from torchtitan_npu.converters.kernels.npu_smla import NpuSMLAModelConfig
from torchtitan_npu.converters.model_custom_interface import ModelCustomConverter
from torchtitan_npu.models.deepseek_v4.model import LiCompute, LiLoss, SparseAttention
from torchtitan_npu.simulator.hardware_shims.smla_shim import SimNpuLiCompute, SimNpuLiLoss, SimNpuSparseAttention

_original_smla_converter: type | None = None


class SimSMLAConverter(ModelCustomConverter):
    """Replaces every SparseAttention/LiCompute/LiLoss submodule with the corresponding Sim*
    shim -- never selects the real fused (A5) or JIT-compiled (non-A5) implementation (see
    design doc §2: neither path can execute under simulation)."""

    def convert(self, model: nn.Module) -> None:
        for name, module in list(model.named_modules()):
            if isinstance(module, SparseAttention):
                replace_module_with_name(model, name, SimNpuSparseAttention(module))
            if isinstance(module, LiCompute):
                replace_module_with_name(model, name, SimNpuLiCompute(module))
            if isinstance(module, LiLoss):
                replace_module_with_name(model, name, SimNpuLiLoss(module))


def apply_smla_shims() -> None:
    """Patch NpuSMLAModelConfig.model_converter to point at SimSMLAConverter. Idempotent: the
    `is None` guard below means only the *first* call saves the pre-patch "original" value;
    every subsequent call is a no-op for that bookkeeping, so unapply_smla_shims() always
    restores the value active before the very first apply_smla_shims() call."""
    global _original_smla_converter
    if _original_smla_converter is None:
        _original_smla_converter = NpuSMLAModelConfig.model_converter
    NpuSMLAModelConfig.model_converter = SimSMLAConverter


def unapply_smla_shims() -> None:
    """Restore the original converter class. Safe to call even if apply_smla_shims() was
    never called (no-op), and safe to call more than once (idempotent)."""
    global _original_smla_converter
    if _original_smla_converter is not None:
        NpuSMLAModelConfig.model_converter = _original_smla_converter
        _original_smla_converter = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit_tests/simulator/hardware_shims/test_smla_converter.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/hardware_shims/smla_converter.py tests/unit_tests/simulator/hardware_shims/test_smla_converter.py
git commit -m "feat(simulator): add reversible SMLA converter patch (SimSMLAConverter)"
```

---

### Task 5: Wire into `SimulationTrainer`

**Files:**
- Modify: `torchtitan_npu/simulator/trainer.py`
- Modify: `tests/unit_tests/simulator/test_trainer.py`

**Interfaces:**
- Consumes: `apply_smla_shims` (Task 4)
- Produces: narrows `_HARDWARE_DEPENDENT_CONVERTER_NAMES` to `frozenset()` (empty).

- [ ] **Step 1: Write the failing test**

In `tests/unit_tests/simulator/test_trainer.py`, replace
`test_strip_hardware_dependent_model_converters_only_removes_smla` (from the MHC plan's Task 6 --
MHC is no longer stripped and now neither is SMLA) with:

```python
def test_strip_hardware_dependent_model_converters_removes_nothing():
    # Updated expectation (was: strips npu_smla only). SMLA is no longer stripped either --
    # SimulationTrainer now installs SimSMLAConverter via apply_smla_shims() instead (Task 4),
    # so npu_smla stays in the converters list and gets a real (shim) implementation rather
    # than being dropped to the base class. _HARDWARE_DEPENDENT_CONVERTER_NAMES is now empty.
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
    assert remaining_names == {"npu_rms_norm", "npu_mhc_pre", "npu_mhc_post", "npu_smla", "npu_gmm"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit_tests/simulator/test_trainer.py -v -k removes_nothing`
Expected: FAIL on the `assert remaining_names == ...` line (current code still strips `npu_smla`).

- [ ] **Step 3: Modify `trainer.py`**

View the current file first to find the exact locations:
```bash
grep -n "_HARDWARE_DEPENDENT_CONVERTER_NAMES\|apply_mhc_shims" torchtitan_npu/simulator/trainer.py
```

Make these two edits to `torchtitan_npu/simulator/trainer.py`:

1. Add the import right after the existing `apply_mhc_shims` import (alphabetically,
   `hardware_shims.mhc_converter` < `hardware_shims.smla_converter`):

```python
from torchtitan_npu.simulator.hardware_shims.smla_converter import apply_smla_shims
```

2. Change `_HARDWARE_DEPENDENT_CONVERTER_NAMES` to an empty frozenset, and update the comment
   block directly above it to explain both MHC and SMLA are now shimmed rather than stripped:

```python
# npu_mhc_pre/npu_mhc_post: no longer stripped -- SimulationTrainer.__init__ calls
# apply_mhc_shims() (torchtitan_npu.simulator.hardware_shims.mhc_converter) to install
# SimHcPre/SimHcHead/SimHcPost instead, preserving the real op names in the captured graph. See
# docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md.
# npu_smla: no longer stripped either -- SimulationTrainer.__init__ calls apply_smla_shims()
# (torchtitan_npu.simulator.hardware_shims.smla_converter) to install SimNpuSparseAttention/
# SimNpuLiCompute/SimNpuLiLoss instead. See
# docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md.
_HARDWARE_DEPENDENT_CONVERTER_NAMES = frozenset()
```

3. In `SimulationTrainer.__init__`, find the line `apply_mhc_shims()` (added by the MHC plan's
   Task 6, immediately before `_strip_hardware_dependent_model_converters(config)`) and add
   `apply_smla_shims()` right after it:

```python
        apply_mhc_shims()
        apply_smla_shims()
        _strip_hardware_dependent_model_converters(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit_tests/simulator/test_trainer.py -v`
Expected: all tests PASS.

Then run the full simulator suite to check for regressions:

Run: `python3 -m pytest tests/unit_tests/simulator/ tests/smoke_tests/simulator/ -q`
Expected: all PASS (previous 129 + this plan's new tests; some may report import errors for
`torch_npu`/`torchtitan_npu.models.deepseek_v4.model` if run outside the container per this repo's
existing sandbox limitation -- acceptable, matches the documented convention).

- [ ] **Step 5: Commit**

```bash
git add torchtitan_npu/simulator/trainer.py tests/unit_tests/simulator/test_trainer.py
git commit -m "feat(simulator): wire apply_smla_shims into SimulationTrainer, strip-out list now empty"
```

---

### Task 6: Small-scale container validation spike

**Files:** none (validation only, no code changes expected)

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

Expected: all tests pass (previous 129 + this plan's new tests, no import-error skips this time
since the container has the exact pinned `torchtitan` and real `torch_npu`).

- [ ] **Step 2: Run the 16-layer smoke config and confirm SMLA's real op names appear**

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
  --hf_assets_path=./tests/assets/tokenizer/deepseekv3_tokenizer 2>&1 | tail -60
echo EXIT_CODE=\$?
grep -oE 'label=\"aclnn\.[a-z_]*\"' simulator_output/deepseek_v4_pro_16_layers/compute_graph.dot | sort | uniq -c
"
```

Expected: `EXIT_CODE=0`, and the `grep` output shows at least `aclnn.npu_sparse_attn_sharedkv`,
`aclnn.npu_sparse_attn_sharedkv_metadata`, `aclnn.npu_sparse_attn_sharedkv_grad` (confirms
`SimNpuSparseAttention`'s real op names now appear where the base `SparseAttention`'s
`matmul`/`softmax`/`scatter_` sequence used to be). `aclnn.npu_lightning_indexer` and
`aclnn.npu_sparse_lightning_indexer_grad_kl_loss` should also appear if this 16-layer config
includes any `R==4` layers -- check `summary.txt`/`compute_graph.dot` for these two; if absent,
verify via `python3 -c "from torchtitan_npu.models.deepseek_v4.config_registry import
deepseek_v4_pro_debug_16_layers; print(deepseek_v4_pro_debug_16_layers().model.compress_ratios)"`
(or equivalent) whether this specific config actually has any `R==4` layer -- if not, this is
expected and not a bug (LiCompute/LiLoss only exist on `R==4` layers), and Step 3 of Task 7 (the
61-layer config, which per the MHC plan's records definitely has R==4 layers, since MHC's own
61-layer validation confirmed non-zero MoE/attention activity across all layer types) is where
this gets exercised for real.

- [ ] **Step 3: If Step 2 fails, debug before proceeding**

If the run crashes: read the stack trace, identify the exact failing call. Likely culprits (in
order of likelihood, based on the MHC plan's own debugging history): (a) a shape mismatch between
what `SimNpuSparseAttention`/`SimNpuLiCompute`/`SimNpuLiLoss` produce and what the surrounding
`InnerAttention.forward` expects downstream -- re-verify against the exact production call site in
`torchtitan_npu/models/deepseek_v4/model.py:693-729`; (b) a missing `torch.autograd.Function`
gradient-count mismatch (the number of values `backward()` returns must exactly match the number of
positional arguments `forward()` received) -- re-count carefully against this plan's exact code;
(c) the `SparseAttention.Config`/`LiCompute.Config`/`LiLoss.Config` real construction in the actual
model differs subtly from this plan's test fixtures. Fix the specific issue found, re-run this
task's Step 1 (full test suite) to catch regressions, then retry Step 2. Do not proceed to Task 7
until Step 2 passes cleanly.

- [ ] **Step 4: Commit any fixes made during debugging**

```bash
git add -A
git commit -m "fix(simulator): correct SMLA shim shape/signature found via 16-layer container validation"
```

(Skip this step if Step 2 passed on the first try with no code changes needed.)

---

### Task 7: Full container re-validation (16-layer + 61-layer)

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

Expected: `EXIT_CODE=0`. Confirm the "Unrecognized op types" list no longer contains
`aten.einsum.default`/`aten.topk.default` if they were previously present ONLY due to the base
`LiCompute`/`SparseAttention` implementations (some of these aten ops may still appear if used
elsewhere in the model unrelated to SMLA -- that's expected; the key check is that
`aclnn.npu_sparse_attn_sharedkv` etc. now appear in the list too, confirming the shim's real names
are captured, even if cost-model coverage for them is "unknown" -- same as MHC's `triton.*` entries).

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
grep -oE 'label=\"aclnn\.[a-z_]*\"' simulator_output/deepseek_v4_pro_61_layers/compute_graph.dot | sort | uniq -c
"
```

Expected: `EXIT_CODE=0` (takes ~8-9 minutes, matching prior acceptance runs' timing). Confirm
`RankTable`: `world_size=384`, `dim_degrees["ep"]=192`, `pp=tp=cp=1` (unchanged from all prior
acceptance runs -- SMLA's shim should not affect parallelism/RankTable at all). Confirm all 5
`aclnn.*` op names appear (`npu_sparse_attn_sharedkv`, `npu_sparse_attn_sharedkv_metadata`,
`npu_sparse_attn_sharedkv_grad`, `npu_lightning_indexer`, `npu_sparse_lightning_indexer_grad_kl_loss`)
-- the 61-layer config's `compress_ratios` include `4` (confirmed by the MHC plan's own container
validation showing non-trivial MoE/attention activity across all layer types), so all three shim
classes should be exercised. Confirm forward/backward/optimizer node counts are in the same order
of magnitude as the prior (MHC-only) acceptance run's baseline (forward=34250, backward=68398,
optimizer=3571) -- some difference is expected (SMLA's real op sequence has different granularity
than the base classes' sequence), but a wildly different order of magnitude would indicate a bug.

- [ ] **Step 3: Copy the updated output out of the container for review**

```bash
HOST_DIR="/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/simulator_review_output/deepseek_v4_pro_61_layers_post_smla_shim"
mkdir -p "$HOST_DIR"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/compute_graph.dot" "$HOST_DIR/"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/summary.txt" "$HOST_DIR/"
sudo docker cp "titan-npu-sim-e2e:/mnt/c/Users/admin/Documents/torchtitan-npu-simulator/.worktrees/feat-npu-simulator/simulator_output/deepseek_v4_pro_61_layers/trace.html" "$HOST_DIR/"
sudo chown -R $(id -u):$(id -g) "$HOST_DIR"
```

- [ ] **Step 4: Update the design doc with final validated results**

Add a "§9. 验证结果" section to
`docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md` recording: the exact
container commands run, exit codes, and the confirmed presence of all 5 `aclnn.*` op names in the
61-layer output, plus the forward/backward/optimizer node counts observed.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md
git commit -m "docs(simulator): record SMLA shim validation results (16-layer + 61-layer re-run)"
git push origin feat/npu-simulator
```

## Explicitly out of scope for this plan

- **`target_npu_device_type == "A5"`**: as with the MHC plan, the shims always record non-A5
  (ACLNN JIT) op names regardless of this config value -- the A5 path's real shapes were never
  verified (design doc §7) since it depends on the unavailable `custom_ops` private package
  regardless of environment.
