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
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch

# Sentinel: distinguishes "attribute did not exist before patching" (must be
# deleted on unpatch) from "attribute existed and was None" (must be restored
# to None on unpatch).
_MISSING = object()


class _MetaDeviceModule:
    """Minimal stand-in for `torch.cuda`/`torch_npu`, covering every method
    actually called on `device_module` by `torchtitan.trainer`,
    `torchtitan.components.metrics`, and `torchtitan.distributed.utils`
    (verified against the pinned torchtitan commit -- see design doc §5.1),
    plus the subset of the `torch.<device_type>` module API that
    `torch.distributed.device_mesh._get_device_handle`/FSDP2 resolve via
    `getattr(torch, device_type)` (`current_device`, `is_available`,
    `is_initialized`, `current_stream`, `stream`, `memory_summary`,
    `Event`, `Stream`)."""

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
        # total_memory must be non-zero: torchtitan's own
        # DeviceMemoryMonitor._to_pct() unconditionally divides by it
        # (`100 * memory / self.device_capacity`) when computing display
        # percentages. Since `memory_stats()` below always reports 0 bytes
        # under meta simulation (no real memory is ever allocated), the
        # resulting percentage is always 0% regardless of this
        # denominator's exact value -- it exists purely to avoid
        # ZeroDivisionError, not to represent a real capacity.
        return SimpleNamespace(total_memory=1, name=self.name)

    def memory_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {
            "active_bytes.all.peak": 0,
            "reserved_bytes.all.peak": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rng_state(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        # `torch.random.fork_rng` (used by pipeline schedule stage init)
        # calls `device_mod.get_rng_state(device)`.  Returning the host RNG
        # state is sufficient: meta simulation never relies on device RNG,
        # fork_rng only needs a storable/restorable byte buffer.
        return torch.get_rng_state()

    def set_rng_state(self, state: torch.Tensor, *_args: Any, **_kwargs: Any) -> None:
        # Pair with `get_rng_state`.  Ignore the restore under meta sim.
        return None

    def current_stream(self, *_args: Any, **_kwargs: Any) -> "_MetaDeviceModule.Stream":
        return _MetaDeviceModule.Stream()

    @contextmanager
    def stream(self, *_args: Any, **_kwargs: Any):
        yield

    def memory_summary(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    class Event:
        """Matches `torch.Event`'s generic API surface (`query`, `record`,
        `synchronize`, `wait`, `elapsed_time`) -- FSDP2's collectives code
        (`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`) calls
        these on the Event objects returned by `Stream.record_event()`."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def query(self) -> bool:
            return True

        def record(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def elapsed_time(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

    class Stream:
        """Matches `torch.Stream`'s generic API surface (`query`,
        `record_event`, `synchronize`, `wait_event`, `wait_stream`) --
        FSDP2's `foreach_all_gather`/`foreach_reduce` call these directly
        on `device_module.Stream()`/`device_handle.current_stream()`
        instances during every forward/backward pass, independent of the
        `capture_fake_collectives()` interception layer (Task 10), which
        only covers `torch.distributed`-level collective calls."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def query(self) -> bool:
            return True

        def record_event(self, event: "_MetaDeviceModule.Event | None" = None) -> "_MetaDeviceModule.Event":
            event = event or _MetaDeviceModule.Event()
            event.record(self)
            return event

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait_event(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait_stream(self, *_args: Any, **_kwargs: Any) -> None:
            return None


_PATCHED_MODULE_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchtitan.tools.utils", ("device_type", "device_module")),
    ("torchtitan.components.metrics", ("device_type", "device_module")),
    ("torchtitan.distributed.parallel_dims", ("device_type",)),
    ("torchtitan.distributed.utils", ("device_type", "device_module")),
)

# Any stable placeholder string works here: the only consumer is a string
# comparison against fixed version-name literals (see
# `torch_npu.utils._optim.patch_supported_devices`), never a real device query.
_DUMMY_NPU_DEVICE_NAME = "Ascend910B_Simulator"

_original_values: dict[tuple[str, str], Any] = {}
_original_fsdp_validate_no_meta_params: Any = _MISSING
_original_tensor_npu_method: Any = _MISSING
_original_torch_full: Any = _MISSING
_original_grouped_mm: Any = _MISSING
_original_moe_token_dispatch: Any = _MISSING
_original_li_loss_forward: Any = _MISSING
_original_pipeline_schedule_warmup_p2p: Any = _MISSING
_original_window_exchange: tuple[Any, Any] | None = None
_original_dtensor_meta_to_dtensor: Any = _MISSING
_original_rowwise_prepare_output: Any = _MISSING
_original_redistribute_local_tensor: Any = _MISSING
_original_recv_object_list: Any = _MISSING
_original_send_object_list: Any = _MISSING
_original_torch_equal: Any = _MISSING
_patched = False


def _patch_tensor_npu_method_to_meta() -> None:
    """`torch_npu` registers `torch.Tensor.npu(device=None, non_blocking=False,
    **kwargs)` (a `.cuda()`-style convenience alias) as a real device-move
    call -- e.g. `torchtitan_npu.converters.kernels.npu_smla`'s sparse
    attention forward uses `torch.tensor([]).npu()` as a placeholder
    default, which crashes with a real `aclInit()` hardware-init error
    with no NPU device present, completely independent of the
    `device_type`/`device_module` globals patched above (this is an
    explicit, hardcoded `.npu()` call, not a lookup through either of
    those). Redirects it to `.to("meta")` (ignoring the requested device
    index/non_blocking/kwargs -- under meta simulation everything lives on
    meta regardless). No-op if torch_npu does not register this method
    (e.g. this repo's CPU-only unit-test sandbox)."""
    global _original_tensor_npu_method
    if not hasattr(torch.Tensor, "npu"):
        return
    if _original_tensor_npu_method is not _MISSING:
        return
    _original_tensor_npu_method = torch.Tensor.npu
    torch.Tensor.npu = lambda self, *_args, **_kwargs: self.to("meta")


def _is_real_npu_device(device: Any) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "npu"
    return str(device) == "npu" or str(device).startswith("npu:")


def _patch_torch_full_npu_device_literal() -> None:
    """`torchtitan_npu.models.deepseek_v4.model.SparseAttention.forward`
    (the model's BASE, pre-conversion attention class -- used under
    simulation once `npu_smla` is stripped from
    `config.model_converters.converters`, see
    `torchtitan_npu.simulator.trainer._strip_hardware_dependent_model_converters`)
    hardcodes `device="npu"` when building its `index_mask` via
    `torch.full(...)`, independent of the `device_type`/`device_module`
    globals patched above. Wraps `torch.full` to redirect a literal
    `"npu"`/`"npu:N"` device (string or `torch.device`) to `"meta"`;
    passes every other call through unchanged."""
    global _original_torch_full
    if _original_torch_full is not _MISSING:
        return
    _original_torch_full = torch.full

    def _meta_safe_full(*args: Any, **kwargs: Any) -> torch.Tensor:
        if _is_real_npu_device(kwargs.get("device")):
            kwargs["device"] = "meta"
        return _original_torch_full(*args, **kwargs)

    torch.full = _meta_safe_full


def _patch_grouped_mm_offsets_dtype() -> None:
    """`torchtitan_npu.converters.kernels.gmm._run_experts_grouped_mm`
    computes MoE grouped-matmul offsets via
    `torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)`, then
    passes them to `torch._grouped_mm(..., offs=offsets)`. PyTorch's
    generic meta-kernel registration for `_grouped_mm`
    (`torch/_meta_registrations.py::meta_grouped_mm`) strictly requires
    `offs.dtype == torch.int32` and raises otherwise -- a check the real
    NPU kernel apparently does not enforce (this code path is presumably
    exercised routinely in real, non-simulated training), so it only
    surfaces once every op is forced through the generic meta-kernel
    under this simulator. Wraps `torch._grouped_mm` to downcast an
    `int64` `offs` kwarg to `int32` transparently; passes every other
    call through unchanged. Values themselves are never read (this cast
    is dtype-only, shape/value-preserving), so it is safe regardless of
    whether the underlying tensor is on meta or a real device."""
    global _original_grouped_mm
    if not hasattr(torch, "_grouped_mm"):
        return
    if _original_grouped_mm is not _MISSING:
        return
    _original_grouped_mm = torch._grouped_mm

    def _meta_safe_grouped_mm(*args: Any, **kwargs: Any) -> torch.Tensor:
        offs = kwargs.get("offs")
        if offs is not None and offs.dtype == torch.int64:
            kwargs["offs"] = offs.to(torch.int32)
        return _original_grouped_mm(*args, **kwargs)

    torch._grouped_mm = _meta_safe_grouped_mm


def _neutralize_torch_npu_optimizer_device_probe() -> None:
    """Real `torch_npu` installs (found under a genuine CANN container)
    monkeypatch `torch.optim.optimizer._get_foreach_kernels_supported_devices`
    to lazily cache a real device name the first time any optimizer's
    `.step()` runs `_default_to_fused_or_foreach()`. That cache-fill calls
    `torch_npu.npu.current_device()`, which unconditionally calls
    `torch_npu.npu._lazy_init()` -> `torch_npu._C._npu_init()` -- a real
    `aclInit()` hardware call that raises (`"Failed to obtain the SOC
    version"`) with no NPU device present. Pre-filling the module-level
    cache (`torch_npu.utils._optim._device_name`) short-circuits that
    lazy-init path entirely; harmless/no-op when `torch_npu` is not
    installed (e.g. this repo's CPU-only unit-test sandbox)."""
    try:
        import torch_npu.utils._optim as npu_optim
    except Exception:  # best-effort guard: any failure here means "not available in this environment"
        return

    if npu_optim._device_name is None:
        _original_values[("torch_npu.utils._optim", "_device_name")] = None
        npu_optim._device_name = _DUMMY_NPU_DEVICE_NAME


def _patch_swap_optimizer_get_device_info(stub: _MetaDeviceModule) -> None:
    """`torchtitan_npu.patches.optimizer.swap_optimizer` imports
    `get_device_info` BY VALUE (`from torchtitan.tools.utils import
    get_device_info`) and calls it fresh inside `get_torch_device()` to
    build `SwapOptimizersContainer.swap_to_device_stream = get_torch_device().Stream()`.
    `get_device_info()` independently re-detects the "live" device
    type/module via `_get_available_device_type()` every call -- bypassing
    the module-level `device_type`/`device_module` globals patched above
    entirely -- so under real torch_npu it resolves real `torch.cuda`
    (compiled into every torch build regardless of GPU presence), whose
    `.Stream()` unconditionally requires real CUDA hardware. Patches the
    by-value-imported name directly on `swap_optimizer`'s own module
    namespace; no-op if torch_npu (and therefore this torchtitan_npu
    submodule) is not importable, e.g. this repo's CPU-only unit-test
    sandbox."""
    try:
        import torchtitan_npu.patches.optimizer.swap_optimizer as swap_optimizer_mod
    except Exception:  # best-effort guard: any failure here means "not available in this environment"
        return

    key = ("torchtitan_npu.patches.optimizer.swap_optimizer", "get_device_info")
    if key in _original_values:
        return
    _original_values[key] = swap_optimizer_mod.get_device_info
    swap_optimizer_mod.get_device_info = lambda: ("meta", stub)


def _patch_moe_dispatch_to_avoid_meta_tensor_value_reads() -> None:
    """`NpuExpertParallel._token_dispatch` (via `_compute_all_to_all_splits`)
    unconditionally calls `.to(torch.device("cpu"), ...)` then `.tolist()`
    on `num_tokens_per_expert` (a real-data-dependent read of "how many
    tokens were routed to each expert") to compute all-to-all split sizes
    -- even under a fake process group, where `is_fake_process_group`
    already skips the real communication itself but not this real-data
    read. Meta tensors have no data (`NotImplementedError: Cannot copy out
    of meta tensor; no data!`).

    Per the user's original request ("MoE 路由分发策略等需要动态运行数据的，
    默认采用打patch的方式进行强制负载均衡"): `debug.moe_force_load_balance`
    (already forced True by `force_moe_load_balance`, Task 17) makes
    real routing perfectly uniform, so the split sizes are staticly
    derivable from `routed_input.shape[0]` (total routed tokens) and
    `ep_degree` alone -- no tensor-value read is needed. Replaces
    `_token_dispatch`'s fake-mode branch to compute
    `input_splits`/`output_splits` analytically (only their *sum* is ever
    consumed downstream, as `output_size=` for a `.repeat_interleave()`
    call that -- verified empirically -- has a working meta kernel given
    an explicit `output_size`), keeping every other op
    (`torch_npu.npu_moe_token_permute`, `.sum(dim=...)` reductions that
    stay as tensors) unchanged from the original implementation. Falls
    back to the real, unpatched implementation whenever the process group
    is not fake. No-op if torch_npu (and therefore this torchtitan_npu
    submodule) is not importable.

    Tracked separately from `_original_values` (a class attribute, not a
    module attribute -- `importlib.import_module()` cannot resolve
    `"...NpuExpertParallel"` as a module path, same reasoning as
    `_neutralize_fsdp_meta_param_validation`)."""
    global _original_moe_token_dispatch
    try:
        import torchtitan_npu.converters.kernels.moe_dispatch as moe_dispatch_mod
    except Exception:  # best-effort guard: any failure here means "not available in this environment"
        return

    expert_parallel_cls = moe_dispatch_mod.NpuExpertParallel
    if _original_moe_token_dispatch is not _MISSING:
        return
    original_token_dispatch = expert_parallel_cls._token_dispatch
    _original_moe_token_dispatch = (expert_parallel_cls, original_token_dispatch)

    is_fake_process_group = moe_dispatch_mod.is_fake_process_group
    torch_npu = moe_dispatch_mod.torch_npu

    def _meta_safe_token_dispatch(self, mod, inputs, device_mesh):  # noqa: ANN001
        routed_input, num_tokens_per_expert, routed_scores = inputs
        group = device_mesh.get_group()
        if not is_fake_process_group(group):
            return original_token_dispatch(self, mod, inputs, device_mesh)

        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        total_routed_tokens = routed_input.shape[0]
        base, remainder = divmod(total_routed_tokens, ep_degree)
        output_splits = [base + 1 if i < remainder else base for i in range(ep_degree)]
        self.input_splits = self.output_splits = output_splits

        num_tokens_per_expert_group = num_tokens_per_expert
        indices = (
            torch.arange(num_local_experts, dtype=torch.int64, device=routed_input.device)
            .repeat(ep_degree)
            .repeat_interleave(num_tokens_per_expert_group.view(-1), output_size=sum(output_splits))
        )
        routed_input, self.permuted_indices = torch_npu.npu_moe_token_permute(routed_input, indices)
        if routed_scores is not None:
            routed_scores, _ = torch_npu.npu_moe_token_permute(routed_scores, indices)
        num_tokens_per_local_expert = num_tokens_per_expert_group.view(ep_degree, -1).sum(0)
        return (routed_input, num_tokens_per_local_expert, routed_scores)

    expert_parallel_cls._token_dispatch = _meta_safe_token_dispatch


def _patch_li_loss_to_skip_buggy_einsum() -> None:
    """`LiLoss._current_selected_attn_dist` (the base, pre-conversion
    class -- used under simulation since `npu_smla` is stripped from
    `config.model_converters.converters`, see
    `torchtitan_npu.simulator.trainer._strip_hardware_dependent_model_converters`)
    has a real, pre-existing shape bug: it concatenates `kv`/`kv_compress`
    (each shaped `(bsz, seq, head_dim)` -- MLA-style attention shares KV
    across all heads, so there is no separate heads dimension) into
    `kv_states`, then computes
    `torch.einsum("bhsd,bkhd->bhsk", query, kv_states)`, whose equation
    expects `kv_states` to be 4-dimensional (`bkhd`) -- it is actually
    3-dimensional, raising `RuntimeError: the number of subscripts in the
    equation (4) does not match the number of dimensions (3)`. This path
    is never exercised in real production, which always uses the
    NPU-converted `LiLoss` (whose own custom op computes this
    auxiliary loss differently) -- but that NPU-converted form also
    crashes under meta simulation (a real hardware "Invalid device ID"
    check, unreachable from Python-level monkeypatching), which is why
    `npu_smla` is stripped entirely rather than surgically patched.

    `LiLoss.forward`'s only consumer,
    `InnerAttention.forward`, uses its return value purely as an
    auxiliary loss term via `DSAIndexerLossAutoScaler.apply(o, loss)`,
    whose own `forward()` returns `o` unchanged and only stashes `loss`
    for the backward pass (`ctx.save_for_backward(aux_loss)`) -- so the
    main attention output's shape is entirely unaffected by `loss`'s
    exact value. Replaces `LiLoss.forward` with a version that skips the
    buggy computation entirely, returning a zero-valued placeholder loss
    (still calling `self.save_loss(...)` for parity with the original
    logging side effect). No-op if torch_npu (and therefore this
    torchtitan_npu submodule) is not importable."""
    global _original_li_loss_forward
    try:
        import torchtitan_npu.models.deepseek_v4.model as model_mod
    except Exception:  # best-effort guard: any failure here means "not available in this environment"
        return

    li_loss_cls = model_mod.LiLoss
    if _original_li_loss_forward is not _MISSING:
        return
    original_forward = li_loss_cls.forward
    _original_li_loss_forward = (li_loss_cls, original_forward)

    def _meta_safe_forward(self, q, kv, kv_compress, attn_sink, q_indexer, k_indexer, weights,  # noqa: ANN001
                            compress_topk_idxs, index_score, attention_masks, offset):
        loss = torch.zeros((), device=q.device, dtype=torch.float32)
        self.save_loss(loss)
        return loss

    li_loss_cls.forward = _meta_safe_forward


def _neutralize_fsdp_meta_param_validation() -> None:
    """`FSDPParamGroup._lazy_init()` (triggered by the model's first
    forward call) unconditionally calls `_validate_no_meta_params()`,
    which raises `RuntimeError` if ANY sharded parameter's
    `.device.type == "meta"`. This check exists to catch real-training
    user errors (forgetting to materialize a meta-constructed model onto
    real hardware before training) -- but the simulator deliberately keeps
    every parameter on the meta device forever (that is the entire point:
    no real memory is ever allocated), so this defensive check is always a
    false positive here. Neutralizing it is pure PyTorch (`torch.distributed
    .fsdp`), not torchtitan/torchtitan_npu-specific, so this patch applies
    (and is tested) regardless of torch_npu availability.

    Tracked separately from `_original_values` (a class attribute, not a
    module attribute -- `importlib.import_module()` cannot resolve
    `"...FSDPParamGroup"` as a module path)."""
    global _original_fsdp_validate_no_meta_params
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    if _original_fsdp_validate_no_meta_params is not _MISSING:
        return
    _original_fsdp_validate_no_meta_params = FSDPParamGroup._validate_no_meta_params
    FSDPParamGroup._validate_no_meta_params = lambda self: None


def _patch_pipeline_schedule_warmup_for_meta() -> None:
    """PyTorch's pipeline schedule `_warmup_p2p` runs a forward/backward
    "vote" protocol to decide whether each stage operates in STATIC or
    DYNAMIC shape mode.  The vote result is a tensor, and the decision is
    made by calling `result.item() == 1`.  Under meta-device simulation the
    vote tensor lives on the meta device and `.item()` raises
    `RuntimeError: Tensor.item() cannot be called on meta tensors`.

    The vote is only a shape-inference heuristic; under simulation input
    shapes are statically known and the fake process group already bypasses
    real communication.  Forcing STATIC mode for every stage avoids the
    meta-tensor value read and lets the 1F1B/loop schedules proceed.  The
    original vote path is preserved for non-meta tensors so this patch is
    safe if ever exercised on a real device."""
    global _original_pipeline_schedule_warmup_p2p
    try:
        from torch.distributed.pipelining.schedules import (
            InferenceMode,
            _PipelineSchedule,
        )
    except Exception:  # pipeline API may not exist in all torch builds
        return

    if _original_pipeline_schedule_warmup_p2p is not _MISSING:
        return
    _original_pipeline_schedule_warmup_p2p = _PipelineSchedule._warmup_p2p

    def _meta_safe_warmup_p2p(self, stages, has_backward, p2p_done):  # noqa: ANN001
        # Under fake PG, force STATIC mode for all stages.  DYNAMIC mode
        # requires P2P metadata exchange between stages, which doesn't work
        # in a single-process simulation (each process has only one stage).
        # STATIC mode uses user-provided metadata; we populate it below in
        # the _prepare_forward_infra / _prepare_backward_infra patches.
        from torch.distributed.pipelining.stage import PipelineStage

        for stage in stages:
            if isinstance(stage, PipelineStage):
                stage._inference_mode = InferenceMode.STATIC

    _PipelineSchedule._warmup_p2p = _meta_safe_warmup_p2p


def _patch_dtensor_meta_to_dtensor_for_meta() -> None:
    """PyTorch's pipeline schedule metadata inference recreates DTensors from
    `_DTensorMeta` using the recorded local shape and placements.  When a
    previous stage produced a sequence-sharded DTensor, the recreated DTensor
    must agree with the mesh size on the sharded dimension.  Under meta-device
    simulation the local shapes seen by the first stage are sometimes smaller
    than the placement implies (e.g. CP/TP interaction or microbatch splitting),
    so redistribution inside `PrepareModuleInputOutput` fails with
    ``narrow unexpectedly changed concrete size``.

    For metadata inference only, we force the reconstructed DTensor to be
    replicated with the global shape as its local shape.  The module's own
    parallelize hooks then redistribute to the expected sharding, so the
    forward computation (and the captured graph) still sees the correct
    parallelism.  P2P buffer sizes are irrelevant under fake-process-group
    simulation.  The patch only activates when the target device is meta."""
    global _original_dtensor_meta_to_dtensor
    try:
        from torch.distributed.pipelining._utils import _DTensorMeta
        from torch.distributed.tensor import Replicate
    except Exception:
        return

    if _original_dtensor_meta_to_dtensor is not _MISSING:
        return
    _original_dtensor_meta_to_dtensor = _DTensorMeta.to_dtensor

    def _meta_safe_to_dtensor(self, device, mesh):  # noqa: ANN001
        if getattr(device, "type", str(device)) != "meta":
            return _original_dtensor_meta_to_dtensor(self, device, mesh)
        from torch.distributed.pipelining._utils import _make_tensor_from_meta
        from torch.distributed.tensor import DTensor

        local_tensor = _make_tensor_from_meta(
            type(self)(
                shape=self.global_shape,
                stride=self.global_stride,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
                global_shape=self.global_shape,
                global_stride=self.global_stride,
                placements=tuple(Replicate() for _ in self.placements),
                mesh_dim_names=self.mesh_dim_names,
                mesh_layout=self.mesh_layout,
            ),
            device,
        )
        return DTensor.from_local(
            local_tensor,
            device_mesh=mesh,
            placements=tuple(Replicate() for _ in self.placements),
            shape=self.global_shape,
            stride=self.global_stride,
            run_check=False,
        ).requires_grad_(self.requires_grad)

    _DTensorMeta.to_dtensor = _meta_safe_to_dtensor


def _patch_rowwise_parallel_output_for_meta() -> None:
    """PyTorch's `RowwiseParallel` on `nn.Embedding` redistributes the partial
    embedding output to the requested output layout (e.g. `Shard(1)` for
    sequence parallelism).  Under meta-device simulation the local shape of the
    redistributed DTensor can disagree with the mesh size on the sharded
    dimension -- the sequence length appears divided by the world size instead
    of the TP degree, causing downstream `PrepareModuleInputOutput` hooks to
    fail with ``narrow unexpectedly changed concrete size``.

    This patch intercepts `RowwiseParallel._prepare_output_fn` on the meta
    device and, when the output layout shards the sequence dimension, reshapes
    the local tensor so that `local_seq == global_seq // tp_mesh_size`.  Values
    are irrelevant under meta simulation; only the shape matters for capturing
    the correct compute graph."""
    global _original_rowwise_prepare_output
    try:
        from torch.distributed.tensor.parallel.style import RowwiseParallel
        from torch.distributed.tensor import DTensor, Shard
    except Exception:
        return

    if _original_rowwise_prepare_output is not _MISSING:
        return
    _original_rowwise_prepare_output = RowwiseParallel._prepare_output_fn

    @staticmethod
    def _meta_safe_prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):  # noqa: ANN001
        result = _original_rowwise_prepare_output(
            output_layouts, use_local_output, mod, outputs, device_mesh
        )
        if (
            not isinstance(result, torch.Tensor)
            or result.device.type != "meta"
            or not isinstance(outputs, DTensor)
        ):
            return result

        # Only fix Shard(1) (sequence-dim) layouts that are inconsistent.
        mesh_size = device_mesh.size()
        for placement in output_layouts:
            if not isinstance(placement, Shard) or placement.dim != 1:
                continue
            global_seq = outputs.shape[1]
            expected_local_seq = global_seq // mesh_size
            if result.shape[1] == expected_local_seq:
                continue
            # Values are irrelevant under meta simulation; create a tensor with
            # the correct local shape so downstream PrepareModuleInputOutput
            # hooks see consistent DTensor metadata.
            new_shape = list(result.shape)
            new_shape[1] = expected_local_seq
            result = torch.empty(
                new_shape,
                dtype=result.dtype,
                device=result.device,
                requires_grad=result.requires_grad,
            )
        return result

    RowwiseParallel._prepare_output_fn = _meta_safe_prepare_output_fn


def _patch_window_exchange_for_fake_pg() -> None:
    """DeepSeek-V4's context-parallel attention uses `_WindowExchange`, an
    autograd.Function that performs P2P `c10d.isend`/`irecv` between adjacent
    CP ranks to slide a local attention window.  Under a fake process group
    these P2P ops are not real sends, and more importantly DTensor has no
    sharding strategy for `c10d.send.default`, so the call raises
    `NotImplementedError`.  The simulator's `capture_fake_collectives()` only
    intercepts collective ops, not P2P, so we short-circuit the exchange
    analytically under fake PG: the tensor shape is still updated as if the
    window tokens were received/sent, but no actual communication is issued.
    Values are irrelevant under meta simulation.  Falls back to the real P2P
    path for non-fake process groups."""
    global _original_window_exchange
    try:
        from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import (
            _WindowExchange,
        )
        from torchtitan_npu.distributed.process_group import is_fake_process_group
    except Exception:
        return

    if _original_window_exchange is not None:
        return
    _original_window_exchange = (_WindowExchange.forward, _WindowExchange.backward)

    orig_forward = _WindowExchange.forward
    orig_backward = _WindowExchange.backward

    def _meta_safe_forward(ctx, tensor, window, group):  # noqa: ANN001
        if not is_fake_process_group(group):
            return orig_forward(ctx, tensor, window, group)

        rank = group.rank()
        world_size = group.size()
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.window = window
        ctx.forward_sent = rank + 1 < world_size
        ctx.forward_recvd = rank > 0

        if ctx.forward_recvd:
            recv_buf = torch.empty_like(tensor[:, -window:])
            tensor = torch.cat([recv_buf, tensor], dim=1)
        return tensor

    def _meta_safe_backward(ctx, grad_output):  # noqa: ANN001
        if not is_fake_process_group(ctx.group):
            return orig_backward(ctx, grad_output)

        window = ctx.window
        if ctx.forward_sent:
            grad_recv = torch.zeros_like(grad_output[:, :window])
            grad_output[:, -window:] = grad_output[:, -window:] + grad_recv
        if ctx.forward_recvd:
            grad_output = grad_output[:, window:]
        return grad_output, None, None

    _WindowExchange.forward = _meta_safe_forward
    _WindowExchange.backward = _meta_safe_backward


def _patch_redistribute_local_tensor_for_meta() -> None:
    """Under meta-device simulation, ``redistribute_local_tensor`` calls
    ``funcol.all_gather_tensor`` / ``reduce_scatter_tensor`` etc. to transform
    a DTensor's sharding.  The fake process group's all_gather does not
    actually concatenate shards (it returns a tensor of the *local* shape),
    so the subsequent ``_maybe_unpad_tensor`` sees a shape mismatch and raises
    ``RuntimeError: narrow unexpectedly changed concrete size``.

    Since values are irrelevant under meta simulation, we short-circuit the
    entire redistribution: compute the correct *local* shape from the target
    ``DTensorSpec`` and return an empty meta tensor with that shape.  This
    preserves correct shape propagation for the captured compute graph while
    avoiding all collective communication.  Falls back to the original for
    non-meta tensors."""
    global _original_redistribute_local_tensor
    try:
        from torch.distributed.tensor._redistribute import redistribute_local_tensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
        from torch.distributed.tensor.placement_types import Shard, Partial, Replicate
    except Exception:
        return

    if _original_redistribute_local_tensor is not _MISSING:
        return
    _original_redistribute_local_tensor = redistribute_local_tensor

    def _meta_safe_redistribute(local_tensor, current_spec, target_spec, *, async_op=False, use_graph_based_transform=None, is_explicit=False):  # noqa: ANN001
        # Only short-circuit on meta device tensors
        if not isinstance(local_tensor, torch.Tensor) or local_tensor.device.type != "meta":
            return _original_redistribute_local_tensor(
                local_tensor, current_spec, target_spec,
                async_op=async_op, use_graph_based_transform=use_graph_based_transform, is_explicit=is_explicit,
            )

        # Compute the correct local shape for the target spec.
        # The target spec's .shape is the *global* (logical) shape; we need
        # the local shape after applying target placements on the mesh.
        global_shape = tuple(target_spec.shape)
        mesh = target_spec.mesh
        placements = target_spec.placements
        local_shape = list(global_shape)
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                mesh_size = mesh.size(mesh_dim)
                if mesh_size > 1 and dim < len(local_shape):
                    local_shape[dim] = max(1, global_shape[dim] // mesh_size)
            # Partial -> local shape is same as global (each rank has full size)
            # Replicate -> local shape is same as global
        return torch.empty(
            tuple(local_shape),
            dtype=local_tensor.dtype,
            device="meta",
            requires_grad=local_tensor.requires_grad,
        )

    import torch.distributed.tensor._redistribute as _redistribute_module
    _redistribute_module.redistribute_local_tensor = _meta_safe_redistribute


def _patch_object_collectives_for_fake_pg() -> None:
    """PyTorch's pipeline schedule exchanges stage metadata (tensor shapes,
    dtypes, requires_grad) between stages via ``dist.recv_object_list`` /
    ``dist.send_object_list``.  These functions internally broadcast a size
    tensor and call ``.item()`` on it, which raises
    ``RuntimeError: Tensor.item() cannot be called on meta tensors`` under
    meta-device simulation.

    Under a fake process group there is only one process (rank 0), so every
    stage's metadata is already local -- the "send" and "recv" are no-ops.
    We short-circuit both functions: ``send_object_list`` does nothing, and
    ``recv_object_list`` leaves the caller-provided ``object_list`` unchanged
    (the caller pre-fills it with placeholder metadata that the pipeline
    schedule overwrites during its own inference).  Falls back to the real
    implementation for non-fake process groups."""
    global _original_recv_object_list, _original_send_object_list
    try:
        from torch.distributed.distributed_c10d import recv_object_list, send_object_list
        from torchtitan_npu.distributed.process_group import is_fake_process_group
    except Exception:
        return

    if _original_recv_object_list is not _MISSING:
        return
    _original_recv_object_list = recv_object_list
    _original_send_object_list = send_object_list

    def _meta_safe_recv_object_list(object_list, src=None, group=None, device=None, group_src=None, use_batch=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return _original_recv_object_list(object_list, src=src, group=group, device=device, group_src=group_src, use_batch=use_batch)
        # Under fake PG, there is only one process.  The pipeline schedule
        # pre-fills object_list with placeholder metadata; leave it as-is.
        return 0

    def _meta_safe_send_object_list(object_list, dst=None, group=None, device=None, group_dst=None, use_batch=False):  # noqa: ANN001
        if not is_fake_process_group(group):
            return _original_send_object_list(object_list, dst=dst, group=group, device=device, group_dst=group_dst, use_batch=use_batch)
        # No-op under fake PG
        return None

    import torch.distributed as dist_mod
    dist_mod.recv_object_list = _meta_safe_recv_object_list
    dist_mod.send_object_list = _meta_safe_send_object_list


def _patch_pipeline_stage_meta_exchange_for_fake_pg() -> None:
    """PyTorch's pipeline schedule exchanges forward/backward metadata
    (tensor shapes, dtypes, placements) between stages via
    ``PipelineStage._send_meta`` / ``_recv_meta``, which use
    ``dist.send_object_list`` / ``dist.recv_object_list`` under the hood.

    Under a fake process group all stages run in the same process, but each
    stage is assigned a different *group rank* (so ``_is_same_rank`` returns
    False), causing the schedule to take the P2P metadata path.  Since
    ``send_object_list`` / ``recv_object_list`` are no-ops under fake PG
    (see ``_patch_object_collectives_for_fake_pg``), the recv would return
    ``None`` instead of a ``_StageForwardMeta`` / ``_StageBackwardMeta``,
    raising ``PipeliningMetadataError``.

    This patch replaces ``_send_meta`` / ``_recv_meta`` with an in-process
    shared dict so that sent metadata is immediately available to recv.
    Falls back to the original for non-fake process groups."""
    try:
        from torch.distributed.pipelining.stage import PipelineStage
        from torchtitan_npu.distributed.process_group import is_fake_process_group
    except Exception:
        return

    if hasattr(PipelineStage, "_sim_shared_meta_buffer"):
        return  # already patched

    PipelineStage._sim_shared_meta_buffer = {}

    orig_send_meta = PipelineStage._send_meta
    orig_recv_meta = PipelineStage._recv_meta

    def _sim_send_meta(self, meta, dst_stage):  # noqa: ANN001
        if not is_fake_process_group(self.group):
            return orig_send_meta(self, meta, dst_stage)
        key = (self.stage_index, dst_stage)
        PipelineStage._sim_shared_meta_buffer[key] = meta

    def _sim_recv_meta(self, src_stage):  # noqa: ANN001
        if not is_fake_process_group(self.group):
            return orig_recv_meta(self, src_stage)
        key = (src_stage, self.stage_index)
        return PipelineStage._sim_shared_meta_buffer.get(key)

    PipelineStage._send_meta = _sim_send_meta
    PipelineStage._recv_meta = _sim_recv_meta

    # In STATIC mode, _prepare_forward_infra reads _user_meta.inputs/outputs
    # which are None because PipelineStage was created without explicit
    # input_args/output_args (DSV4's pipeline_module_split doesn't provide
    # them).  Patch _prepare_forward_infra to run a local forward pass on
    # meta tensors to infer input/output shapes, then populate _stage_meta.
    orig_prepare_forward = PipelineStage._prepare_forward_infra

    def _sim_prepare_forward_infra(self, num_microbatches, args, kwargs=None, has_backward=False):  # noqa: ANN001
        from torchtitan_npu.distributed.process_group import is_fake_process_group as _is_fake
        if not _is_fake(self.group):
            return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)

        # If user_meta.inputs is already set, use the original path
        if self._user_meta.inputs is not None:
            return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)

        # Run a local forward pass to infer input/output shapes
        from torch.distributed.pipelining._utils import (
            TensorMeta,
            _StageForwardMeta,
            extract_tensor_meta,
        )

        # Determine input tensors: first stage uses args, others use placeholder
        if self.is_first:
            if isinstance(args, _StageForwardMeta):
                input_tensors = args.forward_metas
            elif args is None:
                input_tensors = ()
            else:
                input_tensors = args if isinstance(args, tuple) else (args,)
        else:
            # Non-first stage: create placeholder input from the shared buffer
            fwd_meta = PipelineStage._sim_shared_meta_buffer.get((self.stage_index - 1, self.stage_index))
            if fwd_meta is not None:
                # Create empty tensors from the metadata
                input_tensors = tuple(
                    torch.empty(m.shape, dtype=getattr(torch, m.dtype) if isinstance(m.dtype, str) else m.dtype, device="meta")
                    if isinstance(m, TensorMeta)
                    else torch.empty((), device="meta")
                    for m in fwd_meta.forward_metas
                )
            else:
                input_tensors = ()

        # Run forward to get outputs (no_grad not needed on meta device)
        try:
            outputs = self.submod(*input_tensors, **(kwargs or {}))
        except Exception:
            # If forward fails, use empty outputs
            outputs = input_tensors

        if not isinstance(outputs, tuple):
            outputs = (outputs,) if outputs is not None else ()

        # Populate _user_meta for STATIC mode (the original _prepare_forward_infra
        # reads _user_meta.inputs/outputs in STATIC mode, not _stage_meta)
        self._user_meta.inputs = tuple(extract_tensor_meta(t) for t in input_tensors)
        self._user_meta.outputs = tuple(extract_tensor_meta(t) for t in outputs)

        # Send forward metadata to next stage via shared buffer
        if not self.is_last:
            fwd_meta = _StageForwardMeta(forward_metas=self._user_meta.outputs)
            PipelineStage._sim_shared_meta_buffer[(self.stage_index, self.stage_index + 1)] = fwd_meta

        # Now run the original _prepare_forward_infra which will use _stage_meta
        return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)

    PipelineStage._prepare_forward_infra = _sim_prepare_forward_infra

    # Similarly patch _prepare_backward_infra to infer backward metadata
    orig_prepare_backward = PipelineStage._prepare_backward_infra

    def _sim_prepare_backward_infra(self, num_microbatches, loss_fn=None, target=None, received_grad_meta=None):  # noqa: ANN001
        from torchtitan_npu.distributed.process_group import is_fake_process_group as _is_fake
        if not _is_fake(self.group):
            return orig_prepare_backward(self, num_microbatches, loss_fn=loss_fn, target=target, received_grad_meta=received_grad_meta)

        if self._user_meta.input_grads is not None or self._user_meta.output_grads is not None:
            return orig_prepare_backward(self, num_microbatches, loss_fn=loss_fn, target=target, received_grad_meta=received_grad_meta)

        from torch.distributed.pipelining._utils import (
            _StageBackwardMeta,
            _derive_grad_metas,
            extract_tensor_meta,
        )

        # Derive output_grads from outputs (gradient shape == output shape)
        if self._user_meta.outputs is not None:
            self._user_meta.output_grads = _derive_grad_metas(self._user_meta.outputs)
        # Derive input_grads from inputs
        if self._user_meta.inputs is not None:
            self._user_meta.input_grads = _derive_grad_metas(self._user_meta.inputs)

        # Send backward metadata to previous stage via shared buffer
        if not self.is_first:
            bwd_meta = _StageBackwardMeta(backward_metas=self._stage_meta.input_grads)
            PipelineStage._sim_shared_meta_buffer[(self.stage_index, self.stage_index - 1)] = bwd_meta

        return orig_prepare_backward(self, num_microbatches, loss_fn=loss_fn, target=target, received_grad_meta=received_grad_meta)

    PipelineStage._prepare_backward_infra = _sim_prepare_backward_infra


def _patch_torch_equal_for_meta() -> None:
    """``torch.equal`` has no Meta kernel registered.  Under meta-device
    simulation, DTensor's ``_partition_value`` (used by Partial placement
    reduction) calls ``mask_buffer.materialize_mask`` which calls
    ``torch.equal(self.data, mask)`` to check if the mask buffer needs
    updating.  Since values are irrelevant under meta simulation, we
    short-circuit ``torch.equal`` to return ``True`` when either operand is
    a meta tensor.  Falls back to the original for non-meta tensors."""
    global _original_torch_equal
    if _original_torch_equal is not _MISSING:
        return
    _original_torch_equal = torch.equal

    def _meta_safe_equal(a, b):  # noqa: ANN001
        if (isinstance(a, torch.Tensor) and a.device.type == "meta") or (
            isinstance(b, torch.Tensor) and b.device.type == "meta"
        ):
            return True
        return _original_torch_equal(a, b)

    torch.equal = _meta_safe_equal


def patch_device_type_to_meta() -> None:
    """Idempotently rebind `device_type="meta"` / `device_module=<stub>`
    across every module that imported them by value at load time, register
    the same stub as `torch.meta` (see `_MetaDeviceModule`'s docstring for
    why), neutralize `torch_npu`'s real-hardware optimizer device probe,
    swap-optimizer device stream construction, explicit
    `torch.Tensor.npu()` calls, a hardcoded `torch.full(...,
    device="npu")` literal, a grouped-matmul offsets dtype mismatch, MoE
    dispatch's real-data-dependent all-to-all split-size computation, and
    the base LiLoss class's real shape bug (see
    `_neutralize_torch_npu_optimizer_device_probe`,
    `_patch_swap_optimizer_get_device_info`,
    `_patch_tensor_npu_method_to_meta`,
    `_patch_torch_full_npu_device_literal`,
    `_patch_grouped_mm_offsets_dtype`,
    `_patch_moe_dispatch_to_avoid_meta_tensor_value_reads`, and
    `_patch_li_loss_to_skip_buggy_einsum`), and disable FSDP2's meta-param
    validation (see `_neutralize_fsdp_meta_param_validation`)."""
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

    _original_values[("torch", "meta")] = getattr(torch, "meta", _MISSING)
    torch.meta = stub

    # torch_npu registers itself as `torch.npu` (the real device-accessor
    # module, e.g. `torch.npu.current_stream()`/`.current_device()`) and
    # ALSO patches core PyTorch internals to call it directly, hardcoded,
    # bypassing every device_type/device_module indirection above --
    # e.g. torch_npu's own FSDP2 patch (`torch_npu/distributed/fsdp/
    # _add_fsdp_patch.py::_patched_finalize_backward`) calls
    # `torch.npu.current_stream().wait_event(event)` unconditionally
    # during the backward pass, triggering a real aclInit() hardware
    # init with no NPU device present. Redirecting `torch.npu` itself to
    # the same meta stub (mirroring `torch.meta` above) covers this and
    # any other such hardcoded `torch.npu.*` call generically. Only
    # relevant when torch_npu is installed (`torch.npu` does not exist
    # otherwise), but the sentinel-based save/restore logic in
    # `unpatch_device_type_to_meta` handles the "did not exist" case too.
    _original_values[("torch", "npu")] = getattr(torch, "npu", _MISSING)
    torch.npu = stub

    _neutralize_torch_npu_optimizer_device_probe()
    _patch_swap_optimizer_get_device_info(stub)
    _patch_tensor_npu_method_to_meta()
    _patch_torch_full_npu_device_literal()
    _patch_grouped_mm_offsets_dtype()
    _patch_moe_dispatch_to_avoid_meta_tensor_value_reads()
    _patch_li_loss_to_skip_buggy_einsum()
    _neutralize_fsdp_meta_param_validation()
    _patch_pipeline_schedule_warmup_for_meta()
    _patch_dtensor_meta_to_dtensor_for_meta()
    _patch_rowwise_parallel_output_for_meta()
    _patch_window_exchange_for_fake_pg()
    _patch_redistribute_local_tensor_for_meta()
    _patch_object_collectives_for_fake_pg()
    _patch_pipeline_stage_meta_exchange_for_fake_pg()
    _patch_torch_equal_for_meta()
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched, _original_fsdp_validate_no_meta_params, _original_tensor_npu_method, _original_torch_full
    global _original_moe_token_dispatch, _original_grouped_mm, _original_li_loss_forward, _original_pipeline_schedule_warmup_p2p
    global _original_window_exchange, _original_dtensor_meta_to_dtensor, _original_rowwise_prepare_output
    for (module_path, attr_name), original in _original_values.items():
        module = importlib.import_module(module_path)
        if original is _MISSING:
            delattr(module, attr_name)
        else:
            setattr(module, attr_name, original)
    _original_values.clear()

    if _original_fsdp_validate_no_meta_params is not _MISSING:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        FSDPParamGroup._validate_no_meta_params = _original_fsdp_validate_no_meta_params
        _original_fsdp_validate_no_meta_params = _MISSING

    if _original_tensor_npu_method is not _MISSING:
        torch.Tensor.npu = _original_tensor_npu_method
        _original_tensor_npu_method = _MISSING

    if _original_torch_full is not _MISSING:
        torch.full = _original_torch_full
        _original_torch_full = _MISSING

    if _original_grouped_mm is not _MISSING:
        torch._grouped_mm = _original_grouped_mm
        _original_grouped_mm = _MISSING

    if _original_moe_token_dispatch is not _MISSING:
        expert_parallel_cls, original_token_dispatch = _original_moe_token_dispatch
        expert_parallel_cls._token_dispatch = original_token_dispatch
        _original_moe_token_dispatch = _MISSING

    if _original_li_loss_forward is not _MISSING:
        li_loss_cls, original_forward = _original_li_loss_forward
        li_loss_cls.forward = original_forward
        _original_li_loss_forward = _MISSING

    if _original_pipeline_schedule_warmup_p2p is not _MISSING:
        from torch.distributed.pipelining.schedules import _PipelineSchedule

        _PipelineSchedule._warmup_p2p = _original_pipeline_schedule_warmup_p2p
        _original_pipeline_schedule_warmup_p2p = _MISSING

    if _original_window_exchange is not None:
        from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import (
            _WindowExchange,
        )

        _WindowExchange.forward, _WindowExchange.backward = _original_window_exchange
        _original_window_exchange = None

    if _original_dtensor_meta_to_dtensor is not _MISSING:
        from torch.distributed.pipelining._utils import _DTensorMeta

        _DTensorMeta.to_dtensor = _original_dtensor_meta_to_dtensor
        _original_dtensor_meta_to_dtensor = _MISSING

    if _original_rowwise_prepare_output is not _MISSING:
        from torch.distributed.tensor.parallel.style import RowwiseParallel

        RowwiseParallel._prepare_output_fn = _original_rowwise_prepare_output
        _original_rowwise_prepare_output = _MISSING

    if _original_redistribute_local_tensor is not _MISSING:
        import torch.distributed.tensor._redistribute as _redistribute_module

        _redistribute_module.redistribute_local_tensor = _original_redistribute_local_tensor
        _original_redistribute_local_tensor = _MISSING

    if _original_recv_object_list is not _MISSING:
        import torch.distributed as dist_mod

        dist_mod.recv_object_list = _original_recv_object_list
        dist_mod.send_object_list = _original_send_object_list
        _original_recv_object_list = _MISSING
        _original_send_object_list = _MISSING

    if _original_torch_equal is not _MISSING:
        torch.equal = _original_torch_equal
        _original_torch_equal = _MISSING

    _patched = False
