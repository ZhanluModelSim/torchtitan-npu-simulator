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
from datetime import timedelta
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
_original_torch_split: Any = _MISSING
_original_redistribute_local_tensor: Any = _MISSING
_original_recv_object_list: Any = _MISSING
_original_send_object_list: Any = _MISSING
_original_torch_equal: Any = _MISSING
_original_fwd_one_chunk: Any = _MISSING
_original_bwd_one_chunk: Any = _MISSING
_original_fused_adamw: Any = _MISSING
_original_llama4_fsdp_mesh_info: Any = _MISSING
_original_bwd_weight_one_chunk: Any = _MISSING
_original_get_stage_indices: Any = _MISSING
_patched = False

# Pipeline parallel execution context, updated by patched
# PipelineStage.forward_one_chunk / backward_one_chunk so that
# comm_events.py's P2P interceptors can attribute isend/irecv calls
# to the correct microbatch, phase, and stage.
#
# `comp_type` is the fine-grained compute-graph class, drawn from
# torch.distributed.pipelining.schedules._ComputationType:
#   "F" forward, "B" full backward (I+W in one autograd.backward pass),
#   "I" backward_input only (stage_backward_input, full_backward=False),
#   "W" backward_weight only (stage_backward_weight / backward_weight_one_chunk),
#   "F_RECOMPUTE" activation-checkpoint recompute forward, "OPTIMIZER".
# It is set by the patched chunk methods below and read by
# dispatch_capture._record_event to bucket L0 ops into per-(stage, comp_type)
# StepGraphs instead of the coarse forward/backward/optimizer triple.
_pp_context: dict[str, int | str] = {
    "mb_idx": 0,
    "phase": "forward",
    # Default stage is -1 ("unattributed"): ops captured outside any compute
    # chunk (FSDP _lazy_init / inter-chunk comm / framework setup, which run
    # before the first forward_one_chunk stamps the real stage) bucket into a
    # clearly-labelled `s-1_*` template instead of masquerading as stage 0's
    # real forward. The chunk patches overwrite this with the real stage_index.
    "stage": -1,
    "comp_type": "F",
    "fsdp_state": "NA",
}

# Per-stage FSDP sharding state machine, updated by the patched
# FSDPParamGroup.unshard/reshard (and by _PipelineScheduleRuntime
# UNSHARD/RESHARD actions). Values: "SHARDED" / "UNSHARDED" / "NA".
# Read by the chunk patches to stamp the current FSDP state onto each
# captured compute graph class. FSDP2 with PP keeps params unsharded
# across the whole step (reshard_after_backward=False), so this usually
# stays "UNSHARDED" — but tracking it preserves the cross-microbatch
# reshard variation the schedule explicitly issues as UNSHARD/RESHARD
# actions (visible in the L2 timeline as comm DataPasses).
_fsdp_state: dict[int, str] = {}

# Global flag: when True, all collective/P2P communication is intercepted
# as no-op (regardless of ProcessGroup type). Set by SimulationTrainer
# when comm.mode is "fake_backend" or "multi_proc_meta".
_is_meta_simulation: bool = False

# Communication layer context: set by patches at call sites to indicate
# whether a comm call originates from model compute ("L1") or framework
# scheduling ("L2"). Read by _record_comm to classify CommEvent.
_comm_layer: str = ""

# True while a pipeline stage runs its DYNAMIC-mode metadata-inference forward
# (`_forward_metadata_inference` -> `_compute_outputs` -> `submod(...)`), which
# PyTorch issues once per stage BEFORE the real compute chunks to infer output
# shapes for cross-rank P2P. That forward is a *framework shape-inference
# artifact*, not a training microbatch: its ops run with the default
# `_pp_context["stage"]=0` (no chunk has stamped the real stage yet) and would
# otherwise pollute the per-(stage, comp_type) L0 templates with a spurious
# `s0_F` graph. dispatch_capture._record_event skips recording while this is
# True, so only the real forward_one_chunk calls define each `s{stage}_F`
# template (with the correct stage attribution).
_in_metadata_inference: bool = False


class _MetaLocalShape(torch.autograd.Function):
    """Change a local meta tensor's shape without detaching autograd."""

    @staticmethod
    def forward(ctx, input_tensor, output_shape):  # noqa: ANN001
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.input_dtype = input_tensor.dtype
        return torch.empty(tuple(output_shape), dtype=input_tensor.dtype, device="meta")

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001
        return torch.empty(ctx.input_shape, dtype=ctx.input_dtype, device="meta"), None


def _local_tensor_for_shape(tensor: torch.Tensor) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    return local_tensor if isinstance(local_tensor, torch.Tensor) else tensor


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
        pass
    else:
        if npu_optim._device_name is None:
            _original_values[("torch_npu.utils._optim", "_device_name")] = None
            npu_optim._device_name = _DUMMY_NPU_DEVICE_NAME

    # Also patch _get_fused_kernels_supported_devices to include "meta"
    # so that fused=True validation passes for meta tensors. Without this,
    # torch.optim.AdamW.__init__ raises:
    #   RuntimeError: `fused=True` requires all the params to be floating
    #   point Tensors of supported devices: [...] but torch.float32 and meta
    try:
        import torch.optim.optimizer as opt_mod
        if not hasattr(opt_mod, "_sim_orig_get_fused_kernels_supported_devices"):
            opt_mod._sim_orig_get_fused_kernels_supported_devices = (
                opt_mod._get_fused_kernels_supported_devices
            )

            def _meta_safe_get_fused_kernels_supported_devices():
                devices = opt_mod._sim_orig_get_fused_kernels_supported_devices()
                if _is_meta_simulation:
                    devices = list(devices) + ["meta"]
                return devices

            opt_mod._get_fused_kernels_supported_devices = (
                _meta_safe_get_fused_kernels_supported_devices
            )
    except Exception:
        pass


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

        # Record EP all_to_all comm events before short-circuiting.
        # EP all_to_all is part of MoE compute (L1), same as CP P2P/allgather.
        global _comm_layer
        _comm_layer = "L1"
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm_with_l0
        recorder = get_active_recorder()
        if recorder is not None:
            _record_comm_with_l0(recorder, "all_to_all", group, routed_input)
            if routed_scores is not None:
                _record_comm_with_l0(recorder, "all_to_all", group, routed_scores)

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

    # Also patch _token_combine to record the combine all_to_all before
    # the is_fake early return skips it.
    original_token_combine = expert_parallel_cls._token_combine

    def _meta_safe_token_combine(self, mod, routed_output, device_mesh):  # noqa: ANN001
        group = device_mesh.get_group()
        if not is_fake_process_group(group):
            return original_token_combine(self, mod, routed_output, device_mesh)

        # Record combine all_to_all before short-circuiting
        global _comm_layer
        _comm_layer = "L1"
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm_with_l0
        recorder = get_active_recorder()
        if recorder is not None:
            _record_comm_with_l0(recorder, "all_to_all", group, routed_output)

        # Original fake-mode logic: unpermute then return (skip all_to_all)
        routed_output = moe_dispatch_mod.NPUMoeTokenUnpermute.apply(
            routed_output, self.permuted_indices, routed_output.shape
        )
        return routed_output

    expert_parallel_cls._token_combine = _meta_safe_token_combine


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
        # Under fake PG (single-process), force STATIC mode.
        # Under multi_proc_meta (gloo, multi-process), use DYNAMIC mode
        # so that non-first stages can receive shapes from previous
        # stages via _send_meta/_recv_meta (gloo send/recv_object_list).
        from torch.distributed.pipelining.stage import PipelineStage
        from torchtitan_npu.distributed.process_group import is_fake_process_group

        for stage in stages:
            if isinstance(stage, PipelineStage):
                if is_fake_process_group(stage.group):
                    # Single-process fake PG: STATIC mode
                    stage._inference_mode = InferenceMode.STATIC
                else:
                    # Multi-process gloo: DYNAMIC mode
                    stage._inference_mode = InferenceMode.DYNAMIC

        # In multi_proc_meta, the vote protocol uses dist.send/recv
        # with int tensors (not meta). Our comm_events interceptors
        # would no-op these, breaking the vote. So for multi_proc_meta,
        # we skip the vote entirely and let DYNAMIC mode handle shape
        # inference via _send_meta/_recv_meta (which use
        # send_object_list/recv_object_list, not intercepted).
        if stages and not is_fake_process_group(stages[0].group):
            # Multi-process: skip vote, DYNAMIC mode will handle it
            return

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
            local_result = _local_tensor_for_shape(result)
            if local_result.shape[1] == expected_local_seq:
                continue
            # Values are irrelevant under meta simulation; create a tensor with
            # the correct local shape so downstream PrepareModuleInputOutput
            # hooks see consistent DTensor metadata.
            new_shape = list(local_result.shape)
            new_shape[1] = expected_local_seq
            corrected_local = _MetaLocalShape.apply(local_result, tuple(new_shape))
            if isinstance(result, DTensor):
                result = DTensor.from_local(
                    corrected_local,
                    device_mesh=result.device_mesh,
                    placements=result.placements,
                    shape=result.shape,
                    stride=result.stride(),
                    run_check=False,
                )
            else:
                result = corrected_local
        return result

    RowwiseParallel._prepare_output_fn = _meta_safe_prepare_output_fn


def _split_meta_dtensor(tensor: torch.Tensor, split_sizes: list[int] | tuple[int, ...], dim: int) -> tuple:
    normalized_dim = dim if dim >= 0 else tensor.ndim + dim
    outputs = []
    start = 0
    for size in split_sizes:
        size = int(size)
        outputs.append(tensor.narrow(normalized_dim, start, size))
        start += size
    return tuple(outputs)


def _patch_torch_split_for_meta_dtensor() -> None:
    """Avoid DTensor SplitWithSizesBackward mixing local and distributed grads."""
    global _original_torch_split
    if _original_torch_split is not _MISSING:
        return
    _original_torch_split = torch.split

    def _meta_safe_split(tensor, split_size_or_sections, dim=0):  # noqa: ANN001
        try:
            from torch.distributed.tensor import DTensor

            is_meta_dtensor = isinstance(tensor, DTensor) and tensor.device.type == "meta"
        except Exception:
            is_meta_dtensor = False
        if not is_meta_dtensor or not isinstance(split_size_or_sections, (list, tuple)):
            return _original_torch_split(tensor, split_size_or_sections, dim=dim)

        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        if sum(int(size) for size in split_size_or_sections) != tensor.shape[normalized_dim]:
            return _original_torch_split(tensor, split_size_or_sections, dim=dim)
        return _split_meta_dtensor(tensor, split_size_or_sections, dim)

    torch.split = _meta_safe_split


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
        global _comm_layer
        _comm_layer = "L1"  # CP P2P is part of attention compute
        if not is_fake_process_group(group) and not _is_meta_simulation:
            return orig_forward(ctx, tensor, window, group)

        rank = group.rank()
        world_size = group.size()
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.window = window
        ctx.forward_sent = rank + 1 < world_size
        ctx.forward_recvd = rank > 0

        # Record CP P2P communication events before short-circuiting.
        # This ensures _WindowExchange's isend/irecv appear in the captured IR.
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm_with_l0
        recorder = get_active_recorder()
        if recorder is not None:
            send_buf = tensor[:, -window:]
            if ctx.forward_sent:
                event = _record_comm_with_l0(recorder, "p2p_send", group, send_buf)
                event.p2p_peer_rank = rank + 1
                event.p2p_direction = "cp_forward_send"
                event.p2p_mb_idx = -1
                event.p2p_stage = rank
            if ctx.forward_recvd:
                recv_buf = torch.empty_like(send_buf)
                event = _record_comm_with_l0(recorder, "p2p_recv", group, recv_buf)
                event.p2p_peer_rank = rank - 1
                event.p2p_direction = "cp_forward_recv"
                event.p2p_mb_idx = -1
                event.p2p_stage = rank

        if ctx.forward_recvd:
            recv_buf = torch.empty_like(tensor[:, -window:])
            tensor = torch.cat([recv_buf, tensor], dim=1)
        return tensor

    def _meta_safe_backward(ctx, grad_output):  # noqa: ANN001
        global _comm_layer
        _comm_layer = "L1"  # CP P2P backward is part of attention compute
        if not is_fake_process_group(ctx.group) and not _is_meta_simulation:
            return orig_backward(ctx, grad_output)

        window = ctx.window
        rank = ctx.rank

        # Record CP P2P backward communication events.
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder, _record_comm_with_l0
        recorder = get_active_recorder()
        if recorder is not None:
            if ctx.forward_recvd:
                grad_send = grad_output[:, :window]
                event = _record_comm_with_l0(recorder, "p2p_send", ctx.group, grad_send)
                event.p2p_peer_rank = rank - 1
                event.p2p_direction = "cp_backward_send"
                event.p2p_mb_idx = -1
                event.p2p_stage = rank
            if ctx.forward_sent:
                grad_recv = torch.empty_like(grad_output[:, :window])
                event = _record_comm_with_l0(recorder, "p2p_recv", ctx.group, grad_recv)
                event.p2p_peer_rank = rank + 1
                event.p2p_direction = "cp_backward_recv"
                event.p2p_mb_idx = -1
                event.p2p_stage = rank

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
        if not is_fake_process_group(self.group) and not _is_meta_simulation:
            return orig_send_meta(self, meta, dst_stage)
        # In multi_proc_meta mode (gloo PG), use real send_object_list
        # to pass metadata between processes
        if _is_meta_simulation and not is_fake_process_group(self.group):
            # Real multi-process: use gloo send_object_list
            import torch.distributed as dist
            peer_global = self._resolve_peer_global_rank(dst_stage)
            dist.send_object_list([meta], dst=peer_global, group=self.group)
            return
        # Single-process fake PG: use shared buffer
        key = (self.stage_index, dst_stage)
        PipelineStage._sim_shared_meta_buffer[key] = meta

    def _sim_recv_meta(self, src_stage):  # noqa: ANN001
        if not is_fake_process_group(self.group) and not _is_meta_simulation:
            return orig_recv_meta(self, src_stage)
        # In multi_proc_meta mode (gloo PG), use real recv_object_list
        if _is_meta_simulation and not is_fake_process_group(self.group):
            import torch.distributed as dist
            peer_global = self._resolve_peer_global_rank(src_stage)
            obj_list = [None]
            dist.recv_object_list(obj_list, src=peer_global, group=self.group)
            return obj_list[0]
        # Single-process fake PG: use shared buffer
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
        # In multi_proc_meta mode (gloo PG, not fake), DYNAMIC mode handles
        # shape inference via _send_meta/_recv_meta (gloo send/recv_object_list).
        # Let the original _prepare_forward_infra run — it will call
        # _forward_metadata_inference which uses _recv_meta to get input
        # shapes from the previous stage.
        global _in_metadata_inference
        if not _is_fake(self.group) and not _is_meta_simulation:
            return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)

        # For multi_proc_meta (gloo, not fake PG), let original handle it
        # (DYNAMIC mode + _send_meta/_recv_meta via gloo). Bracket with the
        # metadata-inference flag so the FSDP _lazy_init / unshard setup and
        # the _compute_outputs forward it triggers are NOT recorded into the
        # per-(stage, comp_type) L0 templates (they are framework
        # shape-inference artifacts, not training microbatches).
        if not _is_fake(self.group) and _is_meta_simulation:
            prev = _in_metadata_inference
            _in_metadata_inference = True
            try:
                return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)
            finally:
                _in_metadata_inference = prev

        # If user_meta.inputs is already set, use the original path
        if self._user_meta.inputs is not None:
            return orig_prepare_forward(self, num_microbatches, args, kwargs=kwargs, has_backward=has_backward)

        # Single-process fake PG: run local forward to infer shapes
        from torch.distributed.pipelining._utils import (
            TensorMeta,
            _StageForwardMeta,
            extract_tensor_meta,
        )

        # Determine input tensors: first stage uses args
        if self.is_first:
            if isinstance(args, _StageForwardMeta):
                input_tensors = args.forward_metas
            elif args is None:
                input_tensors = ()
            else:
                input_tensors = args if isinstance(args, tuple) else (args,)
        else:
            # Non-first stage in single-process fake PG: receive from shared buffer
            fwd_meta = self._recv_meta(self.stage_index - 1)
            if fwd_meta is not None and hasattr(fwd_meta, "forward_metas"):
                input_tensors = tuple(
                    torch.empty(m.shape, dtype=getattr(torch, m.dtype) if isinstance(m.dtype, str) else m.dtype, device="meta")
                    if isinstance(m, TensorMeta)
                    else torch.empty((), device="meta")
                    for m in fwd_meta.forward_metas
                )
            else:
                input_tensors = ()

        # Run forward to get outputs (no_grad not needed on meta device).
        # This is a framework shape-inference forward (the fake-PG analogue of
        # DYNAMIC mode's `_compute_outputs`), not a training microbatch — gate
        # it with `_in_metadata_inference` so dispatch_capture skips recording
        # its ops into the per-(stage, comp_type) L0 templates. (`global`
        # declared once near the top of this function.)
        prev_inf = _in_metadata_inference
        _in_metadata_inference = True
        try:
            outputs = self.submod(*input_tensors, **(kwargs or {}))
        except Exception:
            # If forward fails, use empty outputs
            outputs = input_tensors
        finally:
            _in_metadata_inference = prev_inf

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
        if not _is_fake(self.group) and not _is_meta_simulation:
            return orig_prepare_backward(self, num_microbatches, loss_fn=loss_fn, target=target, received_grad_meta=received_grad_meta)

        # For multi_proc_meta (gloo, not fake), DYNAMIC mode handles
        # backward metadata via _send_meta/_recv_meta. Bracket with the
        # metadata-inference flag so the backward metadata inference (and any
        # FSDP setup it triggers) is not recorded as training compute.
        global _in_metadata_inference
        if not _is_fake(self.group):
            prev = _in_metadata_inference
            _in_metadata_inference = True
            try:
                return orig_prepare_backward(self, num_microbatches, loss_fn=loss_fn, target=target, received_grad_meta=received_grad_meta)
            finally:
                _in_metadata_inference = prev

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


def _patch_pipeline_stage_for_pp_context() -> None:
    """Patch PipelineStage.forward_one_chunk / backward_one_chunk /
    backward_weight_one_chunk to update the global ``_pp_context`` dict
    with the current microbatch index, phase, stage index, and *fine-grained
    compute-graph class* (``comp_type``), and to gate L0 capture on a
    per-(stage, comp_type) class key instead of ``mb_idx == 0``.

    ``comp_type`` maps directly to
    ``torch.distributed.pipelining.schedules._ComputationType``:
      * forward_one_chunk                              -> "F"
      * backward_one_chunk(full_backward=True)        -> "B" (I+W in one pass)
      * backward_one_chunk(full_backward=False)       -> "I" (input-grad only)
      * backward_weight_one_chunk                      -> "W" (weight-grad only)

    The L0 capture gate is ``cap.begin_chunk((stage, comp_type))``:
    the FIRST occurrence of each (stage, comp_type) class is captured in
    full, every subsequent occurrence (same class, later microbatch) is a
    pass-through that only records an L2 timeline event and bumps the
    class's instance count.  This makes every distinct compute graph
    appear exactly once in the L0 IR while keeping capture cost
    proportional to the number of distinct classes (not num_microbatches).

    All PP schedule types (1F1B, GPipe, DualPipe, ZBV, Interleaved, etc.)
    go through these methods, so one set of patches covers every schedule
    variant.  Runtime zero-bubble schedules additionally dispatch
    UNSHARD/RESHARD actions, whose FSDP state is tracked by the unshard/
    reshard patches in ``_patch_comm_layer_context``."""
    global _original_fwd_one_chunk, _original_bwd_one_chunk, _original_bwd_weight_one_chunk
    try:
        from torch.distributed.pipelining.stage import PipelineStage
    except Exception:
        return

    if _original_fwd_one_chunk is not _MISSING:
        return
    _original_fwd_one_chunk = PipelineStage.forward_one_chunk
    _original_bwd_one_chunk = PipelineStage.backward_one_chunk
    _original_bwd_weight_one_chunk = getattr(PipelineStage, "backward_weight_one_chunk", None)

    def _stamp_chunk_context(self, mb_idx: int, comp_type: str, phase: str) -> None:  # noqa: ANN001
        _pp_context["mb_idx"] = mb_idx
        _pp_context["phase"] = phase
        _pp_context["stage"] = self.stage_index
        _pp_context["comp_type"] = comp_type
        _pp_context["fsdp_state"] = _fsdp_state.get(self.stage_index, "NA")

    def _patched_fwd_one_chunk(self, mb_idx, *args, **kwargs):  # noqa: ANN001
        _stamp_chunk_context(self, mb_idx, "F", "forward")
        from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
        cap = get_active_capture()
        if cap is not None:
            cap.begin_chunk((self.stage_index, "F"))
        try:
            result = _original_fwd_one_chunk(self, mb_idx, *args, **kwargs)
        finally:
            if cap is not None:
                cap.end_chunk()
        # Record L2 timeline event (always, regardless of L0 capture)
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.record_timeline_event(
                action="forward_one_chunk",
                pp_stage=self.stage_index,
                pp_mb_idx=mb_idx,
                phase="forward",
                comp_type="F",
            )
        return result

    def _patched_bwd_one_chunk(self, mb_idx, *args, **kwargs):  # noqa: ANN001
        # backward_one_chunk carries a `full_backward` kwarg/arg: True -> "B"
        # (I+W in one autograd.backward pass), False -> "I" (input-grad only,
        # a subsequent backward_weight_one_chunk call will compute "W").
        full_backward = kwargs.get("full_backward", True)
        if not full_backward and len(args) > 0 and isinstance(args[-1], bool):
            full_backward = args[-1]
        comp_type = "B" if full_backward else "I"
        _stamp_chunk_context(self, mb_idx, comp_type, "backward")
        from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
        cap = get_active_capture()
        if cap is not None:
            cap.begin_chunk((self.stage_index, comp_type))
        try:
            result = _original_bwd_one_chunk(self, mb_idx, *args, **kwargs)
        finally:
            if cap is not None:
                cap.end_chunk()
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.record_timeline_event(
                action="backward_one_chunk",
                pp_stage=self.stage_index,
                pp_mb_idx=mb_idx,
                phase="backward",
                comp_type=comp_type,
            )
        return result

    def _patched_bwd_weight_one_chunk(self, bwd_chunk_id, *args, **kwargs):  # noqa: ANN001
        # backward_weight_one_chunk runs the deferred weight-grad pass that
        # pairs with a prior backward_one_chunk(full_backward=False) ("I").
        # Use the chunk_id as the microbatch index for context attribution.
        mb_idx = bwd_chunk_id if isinstance(bwd_chunk_id, int) else _pp_context.get("mb_idx", 0)
        _stamp_chunk_context(self, mb_idx, "W", "backward")
        from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
        cap = get_active_capture()
        if cap is not None:
            cap.begin_chunk((self.stage_index, "W"))
        try:
            result = _original_bwd_weight_one_chunk(self, bwd_chunk_id, *args, **kwargs)
        finally:
            if cap is not None:
                cap.end_chunk()
        from torchtitan_npu.simulator.capture.comm_events import get_active_recorder
        recorder = get_active_recorder()
        if recorder is not None:
            recorder.record_timeline_event(
                action="backward_weight_one_chunk",
                pp_stage=self.stage_index,
                pp_mb_idx=mb_idx,
                phase="backward",
                comp_type="W",
            )
        return result

    PipelineStage.forward_one_chunk = _patched_fwd_one_chunk
    PipelineStage.backward_one_chunk = _patched_bwd_one_chunk
    if _original_bwd_weight_one_chunk is not None:
        PipelineStage.backward_weight_one_chunk = _patched_bwd_weight_one_chunk


def _patch_metadata_inference_skip() -> None:
    """Wrap ``_PipelineStageBase._compute_outputs`` so the DYNAMIC-mode
    metadata-inference forward (run once per stage before the real compute
    chunks to infer cross-rank P2P shapes) sets ``_in_metadata_inference``
    during its execution. ``dispatch_capture._record_event`` skips recording
    while this flag is set, so the inference forward — a framework
    shape-inference artifact that runs with the default ``_pp_context["stage"]=0``
    (no chunk has stamped the real stage yet) — does not pollute the
    per-(stage, comp_type) L0 templates with a spurious ``s0_F`` graph. The
    real ``forward_one_chunk`` calls define each ``s{stage}_F`` template with
    correct stage attribution."""
    try:
        from torch.distributed.pipelining.stage import PipelineStage
    except Exception:
        return
    if hasattr(PipelineStage, "_sim_orig_compute_outputs"):
        return
    PipelineStage._sim_orig_compute_outputs = PipelineStage._compute_outputs

    def _patched_compute_outputs(self, *args, module, **kwargs):  # noqa: ANN001
        global _in_metadata_inference
        prev = _in_metadata_inference
        _in_metadata_inference = True
        try:
            return PipelineStage._sim_orig_compute_outputs(self, *args, module=module, **kwargs)
        finally:
            _in_metadata_inference = prev

    PipelineStage._compute_outputs = _patched_compute_outputs


def _patch_pipeline_action_context() -> None:
    """Patch ``schedules._get_profiler_function_name`` (called by
    ``_PipelineScheduleRuntime._step_microbatches`` as
    ``with record_function(_get_profiler_function_name(action))`` for EVERY
    action in the lowered plan, including non-compute ones) to stamp the
    correct ``_pp_context["stage"]`` / ``comp_type`` per action.

    Problem: only ``forward_one_chunk`` / ``backward_one_chunk`` /
    ``backward_weight_one_chunk`` (my patches) set ``_pp_context["stage"]``.
    The non-compute plan actions — ``UNSHARD`` / ``RESHARD`` / ``SEND_F`` /
    ``RECV_F`` / ``SEND_B`` / ``RECV_B`` / ``REDUCE_GRAD`` — run inside
    ``_perform_action`` WITHOUT going through those chunk methods, so any L0
    op they emit (e.g. the FSDP all-gather from ``unshard()``) is stamped
    with the STALE ``stage`` (the last chunk's, or the default -1 before the
    first chunk) and lands in the wrong template (the ``s-1_F`` "unattributed"
    bucket). This patch stamps the real ``action.stage_index`` (and a
    comm-specific ``comp_type``) right before each action's body runs, so
    those comm ops bucket into ``s{stage}_UNSHARD`` / ``s{stage}_RESHARD``
    etc. Compute actions (F/B/I/W/OVERLAP_F_B) are still overridden by the
    chunk patches that run inside the body, so their attribution is
    unaffected. Single-stage schedules (1F1B/GPipe) don't call this helper
    (they use ``record_function(f"Forward {i}")``) and are unaffected — their
    UNSHARD runs inside ``forward_one_chunk`` where the stage is already set.

    This is the capture-side half of fixing the UNSHARD/RESHARD <-> L0
    linkage gap; the build-side half (``build_schedule_plan`` linking the
    action to its L0 comm op via ``comm_op_id``) lives in schedule_builder.
    """
    try:
        from torch.distributed.pipelining import schedules as sch
    except Exception:
        return
    if hasattr(sch, "_sim_orig_profiler_name"):
        return
    sch._sim_orig_profiler_name = sch._get_profiler_function_name

    def _patched_profiler_name(action):  # noqa: ANN001
        try:
            _pp_context["stage"] = int(getattr(action, "stage_index", -1))
            ct = getattr(getattr(action, "computation_type", None), "value", "")
            # Comm actions get a dedicated comp_type so their L0 ops bucket
            # into s{stage}_{UNSHARD|RESHARD|REDUCE_GRAD}; SEND/RECV inherit
            # the forward/backward comp_type of the phase they belong to.
            if ct in ("UNSHARD", "RESHARD", "REDUCE_GRAD"):
                _pp_context["comp_type"] = ct
            elif ct in ("SEND_F", "RECV_F"):
                _pp_context["comp_type"] = "F"
            elif ct in ("SEND_B", "RECV_B"):
                _pp_context["comp_type"] = "B"
            # compute actions (F/B/I/W/OVERLAP_F_B): leave comp_type to the
            # chunk patch which runs inside the action body (it sets the
            # correct F/B/I/W and the same stage).
        except Exception:
            pass
        return sch._sim_orig_profiler_name(action)

    sch._get_profiler_function_name = _patched_profiler_name


def _patch_get_stage_indices_for_fake_pg() -> None:
    """Patch ``_get_stage_indices`` inside ``pipeline_module_split`` to
    return ALL stage indices (not just the local PP rank's stages) when
    running under a fake process group.

    Under fake PG, only one process runs (pp_rank=0), so
    ``_get_stage_indices`` normally returns ``[0]`` — only stage 0 is
    created and executed.  Other PP stages (1..N-1) never run, so their
    ops and P2P communication are never captured.

    This patch makes ``_get_stage_indices`` return ``range(num_stages)``
    (all stages) under fake PG, so all ``PipelineStage`` objects are
    created in one process.  The PP schedule then runs all stages
    sequentially, and we capture every stage's ops and P2P events.

    Falls back to the original for non-fake process groups."""
    global _original_get_stage_indices
    try:
        from torchtitan.distributed.pipeline_parallel import pipeline_module_split
        from torchtitan_npu.distributed.process_group import is_fake_process_group
    except Exception:
        return

    if _original_get_stage_indices is not _MISSING:
        return

    # _get_stage_indices is a nested function inside pipeline_module_split,
    # so we cannot patch it directly.  Instead, we patch pipeline_module_split
    # itself to override the stage indices when under fake PG.
    import torchtitan.distributed.pipeline_parallel as pp_mod
    _original_pms = pp_mod.pipeline_module_split
    _original_get_stage_indices = _original_pms  # store for unpatch

    def _patched_pipeline_module_split(model, pp_mesh, pp_schedule, device, module_names_per_stage):  # noqa: ANN001
        """Wrapper that creates ALL PP stages under fake PG."""
        # Check if we are under fake PG
        try:
            pg = pp_mesh.get_group("pp")
            is_fake = is_fake_process_group(pg)
        except Exception:
            is_fake = False

        if not is_fake:
            return _original_pms(model, pp_mesh, pp_schedule, device, module_names_per_stage)

        # Under fake PG: call the original, which will use _get_stage_indices
        # to return only [0].  We then replicate the stage-creation loop
        # for ALL indices by calling the original function's inner
        # _build_stage_from_modules.  Since that's a nested function,
        # we replicate the stage creation here.
        import copy
        import torch.nn as nn
        from torch.distributed.pipelining.stage import PipelineStage

        num_stages = len(module_names_per_stage)
        whole_model = model
        pp_group = pp_mesh.get_group("pp")

        def _build_stage(stage_idx):
            m = copy.deepcopy(whole_model)
            modules_to_keep = set(module_names_per_stage[stage_idx])
            for module_name, module_value in m.named_children():
                if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                    layers_to_keep = {
                        name.split(".", 1)[1]
                        for name in modules_to_keep
                        if name.startswith(f"{module_name}.")
                    }
                    if layers_to_keep:
                        if isinstance(module_value, nn.ModuleDict):
                            for layer_name in list(module_value.keys()):
                                if layer_name not in layers_to_keep:
                                    del module_value[layer_name]
                        elif isinstance(module_value, nn.ModuleList):
                            indices_to_keep = {
                                int(idx) for idx in layers_to_keep if idx.isdigit()
                            }
                            new_layers = nn.ModuleList(
                                [layer for i, layer in enumerate(module_value) if i in indices_to_keep]
                            )
                            setattr(m, module_name, new_layers)
                    else:
                        if isinstance(module_value, nn.ModuleDict):
                            setattr(m, module_name, nn.ModuleDict())
                        elif isinstance(module_value, nn.ModuleList):
                            setattr(m, module_name, nn.ModuleList())
                elif module_name not in modules_to_keep:
                    setattr(m, module_name, None)
            stage = PipelineStage(m, stage_idx, num_stages, device, group=pp_group)
            return stage, m

        stages = []
        models = []
        for stage_idx in range(num_stages):
            stage, model_chunk = _build_stage(stage_idx)
            stages.append(stage)
            models.append(model_chunk)

        return stages, models

    pp_mod.pipeline_module_split = _patched_pipeline_module_split


def _patch_build_pipeline_schedule_for_fake_pg() -> None:
    """Patch ``build_pipeline_schedule`` to use ``len(stages)`` as
    ``num_total_stages`` when under fake PG.

    Normally, ``num_total_stages = pp_degree * len(stages)`` because
    each PP rank has ``len(stages)`` stages.  But when
    ``_patch_get_stage_indices_for_fake_pg`` creates ALL stages in one
    process, ``len(stages)`` already equals the total number of stages,
    so multiplying by ``pp_degree`` gives an inflated count (e.g.
    16 * 16 = 256 instead of 16).

    This patch overrides ``num_total_stages`` to ``len(stages)`` under
    fake PG by wrapping the schedule class constructor."""
    try:
        import torchtitan.distributed.pipeline_parallel as pp_mod
        from torchtitan_npu.distributed.process_group import is_fake_process_group
    except Exception:
        return

    if hasattr(pp_mod, "_sim_orig_build_pipeline_schedule"):
        return
    pp_mod._sim_orig_build_pipeline_schedule = pp_mod.build_pipeline_schedule

    def _patched_build_pipeline_schedule(*args, **kwargs):  # noqa: ANN001
        orig = pp_mod._sim_orig_build_pipeline_schedule
        # Check if under fake PG by inspecting the stages' group
        stages = kwargs.get("stages") or (args[0] if args else None)
        is_fake = False
        if stages:
            try:
                pg = stages[0].group
                is_fake = is_fake_process_group(pg)
            except Exception:
                pass

        if not is_fake:
            return orig(*args, **kwargs)

        # Under fake PG: all stages are in one process.
        # Temporarily override parallelism.pipeline_parallel_degree to 1
        # so that num_total_stages = 1 * len(stages) = len(stages)
        parallelism = kwargs.get("parallelism")
        if parallelism is not None and hasattr(parallelism, "pipeline_parallel_degree"):
            orig_pp = parallelism.pipeline_parallel_degree
            parallelism.pipeline_parallel_degree = 1
            try:
                result = orig(*args, **kwargs)
            finally:
                parallelism.pipeline_parallel_degree = orig_pp
            return result

        return orig(*args, **kwargs)

    pp_mod.build_pipeline_schedule = _patched_build_pipeline_schedule


def _patch_device_mesh_world_size_check() -> None:
    """Patch DeviceMesh._setup_world_group_and_device to skip the
    ``mesh > world_size`` check when using fake backend.

    In multi_proc_meta mode, the mesh has 2048 ranks but gloo only
    has 16 processes.  When we use fake backend for the mesh, the
    check ``self._layout.numel() > world_size`` fails because
    ``world_size`` is the gloo world_size (16), not the full
    simulated world_size (2048)."""
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except Exception:
        return

    if hasattr(DeviceMesh, "_sim_orig_setup_world_group"):
        return
    DeviceMesh._sim_orig_setup_world_group = DeviceMesh._setup_world_group_and_device

    def _patched_setup_world_group_and_device(self):  # noqa: ANN001
        from torch.distributed.distributed_c10d import is_initialized, init_process_group, get_world_size, _get_default_group
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        # Skip the mesh > world_size check when device_type is "fake"
        # (the mesh was created with fake backend to simulate a larger
        # world than the actual gloo world_size)
        if self._device_type == "fake":
            # Create a FakeProcessGroup with the full mesh size as the
            # "default group" for this mesh.  This allows new_group() to
            # create subgroups up to mesh_size without hitting the
            # group_world_size > global_world_size check.
            mesh_size = self._layout.numel()
            gloo_rank = get_world_size()  # just to ensure dist is initialized
            import torch.distributed as dist
            global_rank = dist.get_rank() if dist.is_initialized() else 0
            from torch._C._distributed_c10d import FakeProcessGroup
            opts = FakeProcessGroup.Options()
            fake_pg = FakeProcessGroup._create_internal(global_rank, mesh_size, opts)
            return fake_pg

        # For non-fake backend, use the original logic
        world_size = get_world_size()
        if self._layout.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size {world_size}, "
                f"but found {self._layout.numel()} ranks!"
            )
        return DeviceMesh._sim_orig_setup_world_group(self)

    DeviceMesh._setup_world_group_and_device = _patched_setup_world_group_and_device


def _patch_parallel_dims_for_multi_proc(full_ws: int, gloo_ws: int) -> None:
    """Patch ParallelDims to use gloo world_size for mesh creation
    while keeping full_ws for validation.

    In multi_proc_meta mode, we have gloo_ws (e.g. 16) real processes,
    but want to simulate full_ws (e.g. 2048) ranks.  ParallelDims
    validates that the product of all parallel degrees == world_size,
    and build_mesh creates a DeviceMesh of size world_size.  But
    init_device_mesh checks mesh size <= gloo world_size.

    This patch:
    1. Sets ParallelDims.world_size = full_ws (for validation)
    2. Patches init_device_mesh to accept mesh > gloo_ws by using
       fake backend for dimensions that exceed gloo_ws
    """
    try:
        from torch.distributed.device_mesh import init_device_mesh as orig_init_mesh
        import torch.distributed.device_mesh as dm_mod
    except Exception:
        return

    if hasattr(dm_mod, "_sim_orig_init_device_mesh"):
        return
    dm_mod._sim_orig_init_device_mesh = orig_init_mesh

    def _patched_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None, **kwargs):  # noqa: ANN001
        """Wrapper that allows mesh > world_size by using fake backend
        for the oversized mesh."""
        import torch.distributed as dist
        gloo_ws = dist.get_world_size() if dist.is_initialized() else 1
        mesh_size = 1
        for d in mesh_shape:
            mesh_size *= d

        if mesh_size <= gloo_ws:
            return orig_init_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names, **kwargs)

        # Mesh is bigger than gloo world_size: use fake backend.
        # Also patch DeviceMesh to skip the mesh > world_size check.
        _patch_device_mesh_world_size_check()
        return orig_init_mesh("fake", mesh_shape, mesh_dim_names=mesh_dim_names, **kwargs)

    dm_mod.init_device_mesh = _patched_init_device_mesh

    # Also patch the by-value import in torchtitan.distributed.parallel_dims
    try:
        import torchtitan.distributed.parallel_dims as pd_mod
        pd_mod.init_device_mesh = _patched_init_device_mesh
    except Exception:
        pass


def _patch_new_group_for_fake_backend() -> None:
    """Patch ``_new_group_with_tag`` to skip the
    ``group_world_size > global_world_size`` check when
    ``backend="fake"``.

    In multi_proc_meta mode, the gloo world_size is small (e.g. 4),
    but we need to create fake process groups for larger dimensions
    (e.g. FSDP=16, EP=128).  FakeProcessGroup doesn't need real
    processes — it just stores rank/size — so the size check is
    unnecessary for fake backend subgroups."""
    try:
        from torch.distributed import distributed_c10d as dc
    except Exception:
        return

    if hasattr(dc, "_sim_orig_new_group_with_tag"):
        return
    dc._sim_orig_new_group_with_tag = dc._new_group_with_tag

    orig = dc._new_group_with_tag

    def _patched_new_group_with_tag(  # noqa: ANN001
        ranks=None, timeout=None, backend=None, backend_options=None,
        pg_tag=None, use_local_synchronization=False, group_desc=None,
        device_id=None, sort_ranks=True,
    ):
        # In meta simulation mode, when group_world_size > gloo world_size
        # or any rank >= gloo world_size, create a FakeProcessGroup directly
        # (bypassing the size/range checks).  This happens for subgroups of
        # a fake world_mesh whose rank IDs exceed the gloo process count.
        if _is_meta_simulation and dc.is_initialized():
            gloo_ws = dc.get_world_size()
            group_ws = len(ranks) if ranks is not None else gloo_ws
            needs_fake = group_ws > gloo_ws
            if not needs_fake and ranks:
                needs_fake = any(r >= gloo_ws for r in ranks)
            if needs_fake:
                # Bypass _new_group_with_tag's size check by calling
                # _new_process_group_helper directly with backend="fake".
                if sort_ranks and ranks is not None:
                    ranks = sorted(ranks)
                global_rank = dc.get_rank()
                group_rank = ranks.index(global_rank) if ranks and global_rank in ranks else -1

                if group_rank == -1:
                    # This process is not in the subgroup.
                    # Still need to call _new_process_group_helper for
                    # collective consistency, but it returns NON_GROUP_MEMBER.
                    pass

                from torch.distributed.distributed_c10d import (
                    _new_process_group_helper, _process_group_name, _world,
                    Backend, PrefixStore, _get_default_timeout,
                )
                from datetime import timedelta
                group_name = _process_group_name(
                    ranks or [], use_hashed_name=use_local_synchronization,
                )
                group_desc = "undefined" if group_desc is None else group_desc
                default_pg = dc._get_default_group()
                _, default_store = _world.pg_map[default_pg]
                if timeout is None:
                    timeout = _get_default_timeout(Backend.FAKE)

                pg, _ = _new_process_group_helper(
                    group_size=group_ws,
                    group_rank=group_rank if group_rank >= 0 else 0,
                    global_ranks_in_group=ranks or [],
                    backend=Backend.FAKE,
                    store=default_store,
                    group_name=group_name,
                    backend_options=backend_options,
                    timeout=timeout,
                    pg_tag=pg_tag,
                    device_id=device_id,
                    group_desc=group_desc,
                )

                if group_rank == -1:
                    return dc.GroupMember.NON_GROUP_MEMBER
                return pg

        return orig(
            ranks, timeout, backend, backend_options, pg_tag,
            use_local_synchronization, group_desc, device_id, sort_ranks,
        )

    dc._new_group_with_tag = _patched_new_group_with_tag
    # Also patch the public new_group — it has a different signature than
    # _new_group_with_tag (e.g. pg_options), so use **kwargs to forward.
    def _patched_new_group(*args, **kwargs):  # noqa: ANN001
        # Map new_group's kwargs to _new_group_with_tag's kwargs
        return _patched_new_group_with_tag(
            ranks=kwargs.get("ranks"),
            timeout=kwargs.get("timeout"),
            backend=kwargs.get("backend"),
            backend_options=kwargs.get("pg_options"),  # new_group uses pg_options
            pg_tag=None,
            use_local_synchronization=kwargs.get("use_local_synchronization", False),
            group_desc=kwargs.get("group_desc"),
            device_id=kwargs.get("device_id"),
            sort_ranks=kwargs.get("sort_ranks", True),
        )
    dc.new_group = _patched_new_group

    # Patch by-value imports in device_mesh and torch.distributed
    try:
        import torch.distributed.device_mesh as dm_mod
        dm_mod.new_group = dc.new_group
    except Exception:
        pass
    try:
        import torch.distributed as dist_mod
        dist_mod.new_group = dc.new_group
    except Exception:
        pass


def _patch_fsdp_get_device_from_mesh() -> None:
    """Patch FSDP's ``_get_device_from_mesh`` to return a meta device when
    the mesh was created with ``device_type="fake"`` (multi_proc_meta mode).

    FSDP calls ``_get_device_from_mesh(mesh)`` during ``fully_shard()``
    to determine the device for parameter sharding.  When the mesh uses
    fake backend, ``_get_device_handle("fake")`` returns None, causing
    an AttributeError.  This patch returns ``torch.device("meta")`` for
    fake meshes, which is correct under meta simulation."""
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_init
    except Exception:
        return

    if hasattr(_fsdp_init, "_sim_orig_get_device_from_mesh"):
        return
    _fsdp_init._sim_orig_get_device_from_mesh = _fsdp_init._get_device_from_mesh

    def _patched_get_device_from_mesh(mesh):  # noqa: ANN001
        if mesh.device_type == "fake":
            return torch.device("meta")
        return _fsdp_init._sim_orig_get_device_from_mesh(mesh)

    _fsdp_init._get_device_from_mesh = _patched_get_device_from_mesh

    # Also patch the by-value import in _fully_shard
    try:
        from torch.distributed.fsdp._fully_shard import _fully_shard as fs_mod
        fs_mod._get_device_from_mesh = _patched_get_device_from_mesh
    except Exception:
        pass


def _patch_dtensor_random_for_fake_mesh() -> None:
    """Patch DTensor's ``_resolve_device`` in ``tensor._random`` to return
    a meta device when the mesh uses ``device_type="fake"``.

    ``_resolve_device`` calls ``torch.device(f"{device_type}:{device_idx}")``
    which fails for "fake" because it's not a registered torch device type.
    This patch returns ``torch.device("meta")`` for fake meshes."""
    try:
        from torch.distributed.tensor import _random as rng_mod
    except Exception:
        return

    if hasattr(rng_mod, "_sim_orig_resolve_device"):
        return
    rng_mod._sim_orig_resolve_device = rng_mod._resolve_device

    def _patched_resolve_device(device_mesh=None):  # noqa: ANN001
        if device_mesh is not None and device_mesh.device_type == "fake":
            return torch.device("meta")
        return rng_mod._sim_orig_resolve_device(device_mesh=device_mesh)

    rng_mod._resolve_device = _patched_resolve_device


def _patch_comm_layer_context() -> None:
    """Patch call sites to set ``_comm_layer`` context variable, so
    ``_record_comm`` can classify each CommEvent as L1 (model compute)
    or L2 (framework scheduling) based on the call path, not name patterns.

    L1 (model compute):
      - _WindowExchange (already patched in _patch_window_exchange_for_fake_pg)
      - _allgather_seq (CompressorAttentionCP._post_hook)

    L2 (framework scheduling):
      - FSDPParamGroup.unshard / reshard
      - PipelineSchedule._step_microbatches
    """
    global _comm_layer

    # Patch _allgather_seq → L1
    try:
        from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import (
            _allgather_seq as _orig_allgather_seq,
        )
        import torchtitan_npu.distributed.context_parallel.compressor_attention_cp as _cp_mod

        if not hasattr(_cp_mod, "_sim_orig_allgather_seq"):
            _cp_mod._sim_orig_allgather_seq = _orig_allgather_seq

            def _patched_allgather_seq(tensor, mesh, seq_dim=1):  # noqa: ANN001
                global _comm_layer
                _comm_layer = "L1"
                return _orig_allgather_seq(tensor, mesh, seq_dim)

            _cp_mod._allgather_seq = _patched_allgather_seq
    except Exception:
        pass

    # Patch FSDPParamGroup.unshard/reshard → L2 + sync _fsdp_state
    try:
        from torchtitan_npu.simulator.capture.fsdp_residency import install_fsdp_residency_hooks

        install_fsdp_residency_hooks()
    except Exception:
        pass

    # Patch PipelineSchedule._step_microbatches → L2
    try:
        from torch.distributed.pipelining.schedules import _PipelineSchedule

        if not hasattr(_PipelineSchedule, "_sim_orig_step_microbatches"):
            _PipelineSchedule._sim_orig_step_microbatches = _PipelineSchedule._step_microbatches

            def _patched_step_microbatches(self, *args, **kwargs):  # noqa: ANN001
                global _comm_layer
                _comm_layer = "L2"
                return _PipelineSchedule._sim_orig_step_microbatches(self, *args, **kwargs)

            _PipelineSchedule._step_microbatches = _patched_step_microbatches
    except Exception:
        pass

    # Also set L1 in _WindowExchange (already patched, but set _comm_layer)
    # The _meta_safe_forward/backward already set _comm_layer="L1" via
    # the _WindowExchange patch. We add it here for clarity.
    global _original_window_exchange
    if _original_window_exchange is not None:
        # Already patched, just ensure _comm_layer is set
        pass


def _patch_mxfp8_for_meta() -> None:
    """Patch MXFP8 (torchao) for meta-device simulation.

    Three problems are addressed:
    1. ``has_mx_capability`` checks NPU hardware → return True in meta mode
    2. ``NpuMXFP8MM`` calls ``npu_dynamic_mx_quant``/``npu_quant_matmul``
       which need real data → replace with ``SimMXFP8MM`` shim
    3. ``NpuMXFP8GroupedMM`` same → replace with ``SimMXFP8GroupedMM`` shim

    The shims record the real NPU op names via ``record_synthetic_op``
    while using standard ``torch.matmul`` for shape inference on meta
    tensors."""
    # 1. Bypass hardware capability check
    try:
        from torchtitan_npu.patches.torchao_npu import mx_capability_check
        if not hasattr(mx_capability_check, "_sim_orig_has_mx_capability"):
            mx_capability_check._sim_orig_has_mx_capability = mx_capability_check.has_mx_capability

            def _meta_safe_has_mx_capability(major, minor):  # noqa: ANN001
                if _is_meta_simulation:
                    return True
                return mx_capability_check._sim_orig_has_mx_capability(major, minor)

            mx_capability_check.has_mx_capability = _meta_safe_has_mx_capability
            # Patch the by-value import in torchtitan.tools.utils
            try:
                from torchtitan.tools import utils as tt_utils
                tt_utils.has_cuda_capability = _meta_safe_has_mx_capability
            except Exception:
                pass
            # Also patch the by-value import in MXFP8Converter's module
            try:
                import torchtitan.components.quantization.mx as mx_mod
                mx_mod.has_cuda_capability = _meta_safe_has_mx_capability
            except Exception:
                pass
    except Exception:
        pass

    # 2. Replace MXFP8 linear matmul with meta-safe shim
    try:
        import torchao.prototype.mx_formats.mx_linear as mx_linear_mod
        from torchtitan_npu.simulator.hardware_shims.mxfp8_shim import SimMXFP8MM

        if not hasattr(mx_linear_mod, "_sim_orig_to_mxfp8_then_scaled_mm"):
            mx_linear_mod._sim_orig_to_mxfp8_then_scaled_mm = mx_linear_mod._to_mxfp8_then_scaled_mm

            def _meta_safe_to_mxfp8_then_scaled_mm(input_hp, weight_hp, **kwargs):  # noqa: ANN001
                if not _is_meta_simulation:
                    return mx_linear_mod._sim_orig_to_mxfp8_then_scaled_mm(input_hp, weight_hp, **kwargs)
                return SimMXFP8MM.apply(input_hp, weight_hp)

            mx_linear_mod._to_mxfp8_then_scaled_mm = _meta_safe_to_mxfp8_then_scaled_mm
    except Exception:
        pass

    # 3. Replace MXFP8 grouped matmul with meta-safe shim
    try:
        import torchao.prototype.moe_training.mxfp8_grouped_mm as grouped_mm_mod
        from torchtitan_npu.simulator.hardware_shims.mxfp8_shim import SimMXFP8GroupedMM

        if not hasattr(grouped_mm_mod, "_sim_orig_to_mxfp8_then_scaled_grouped_mm"):
            grouped_mm_mod._sim_orig_to_mxfp8_then_scaled_grouped_mm = (
                grouped_mm_mod._to_mxfp8_then_scaled_grouped_mm
            )

            def _meta_safe_to_mxfp8_then_scaled_grouped_mm(A, B_t, offs, **kwargs):  # noqa: ANN001
                if not _is_meta_simulation:
                    return grouped_mm_mod._sim_orig_to_mxfp8_then_scaled_grouped_mm(A, B_t, offs, **kwargs)
                return SimMXFP8GroupedMM.apply(A, B_t, offs)

            grouped_mm_mod._to_mxfp8_then_scaled_grouped_mm = _meta_safe_to_mxfp8_then_scaled_grouped_mm
    except Exception:
        pass


def _patch_init_distributed_for_multi_proc_meta() -> None:
    """Patch ``init_distributed`` to handle ``comm.mode=multi_proc_meta``.

    In multi_proc_meta mode, we use ``gloo`` backend (real multi-process
    rendezvous via torchrun) instead of ``fake`` backend.  All collective
    and P2P communication is still intercepted by ``comm_events.py``
    (via ``_is_meta_simulation`` flag), so gloo only handles the initial
    ``init_process_group`` rendezvous -- no meta tensors ever reach gloo.

    This patch adds a branch for ``multi_proc_meta`` mode that calls
    ``init_process_group("gloo")`` and returns the full simulated
    world_size (from ``simulated_parallel_degrees``) so that
    ``ParallelDims._validate()`` passes with the real parallel degrees."""
    try:
        from torchtitan.distributed import utils as dist_utils
    except Exception:
        return

    if hasattr(dist_utils, "_sim_orig_init_distributed"):
        return
    dist_utils._sim_orig_init_distributed = dist_utils.init_distributed

    def _patched_init_distributed(comm_config, *args, **kwargs):  # noqa: ANN001
        orig = dist_utils._sim_orig_init_distributed
        if comm_config.mode != "multi_proc_meta":
            return orig(comm_config, *args, **kwargs)

        # Multi-process meta simulation: use gloo backend
        import os
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(seconds=comm_config.init_timeout_seconds),
        )

        # Patch new_group to allow fake subgroups larger than gloo world_size
        _patch_new_group_for_fake_backend()

        # Return the full simulated world_size so ParallelDims validation
        # passes with the real parallel degrees (e.g. PP=4, CP=4, DP=4 → 64).
        # The gloo world_size is only PP degree (e.g. 4); the rest is
        # simulated by FakeProcessGroup subgroups.
        sim_ws = os.environ.get("TORCHTITAN_SIM_WORLD_SIZE")
        if sim_ws:
            return int(sim_ws)
        # Fallback: return gloo world_size (PP degree only)
        return dist.get_world_size()

    dist_utils.init_distributed = _patched_init_distributed


def _patch_fused_adamw_for_meta() -> None:
    """Patch ``torch._fused_adamw_`` for meta-device simulation.

    In real NPU training, ``torch._fused_adamw_`` dispatches to
    ``npu.npu_apply_adam_w`` (a fused NPU kernel). Under meta simulation,
    ``fused=True`` raises RuntimeError (meta device not in supported list).

    This patch replaces ``torch._fused_adamw_`` with a meta-safe shim that
    records ``npu.npu_apply_adam_w.default`` without performing numerical
    optimizer updates.
    """
    global _original_fused_adamw
    if _original_fused_adamw is not _MISSING:
        return
    try:
        from torchtitan_npu.simulator.hardware_shims.optimizer_shim import _meta_safe_fused_adamw
        import torch

        _original_fused_adamw = torch._fused_adamw_

        def _patched_fused_adamw(params, grads, exp_avgs, exp_avg_sqs,
                                 max_exp_avg_sqs, state_steps, **kwargs):
            if not _is_meta_simulation:
                return _original_fused_adamw(
                    params, grads, exp_avgs, exp_avg_sqs,
                    max_exp_avg_sqs, state_steps, **kwargs
                )
            return _meta_safe_fused_adamw(
                params, grads, exp_avgs, exp_avg_sqs,
                max_exp_avg_sqs, state_steps, **kwargs
            )

        torch._fused_adamw_ = _patched_fused_adamw
    except Exception:
        _original_fused_adamw = _MISSING
        pass


def _patch_llama4_hsdp_ep_mesh_info() -> None:
    """Preserve the replicate axis for llama4's HSDP+EP placement callback.

    The pinned torchtitan ``llama4.apply_fsdp`` builds ``FSDPMeshInfo``
    explicitly for per-parameter EP placement. On a 2D HSDP mesh that makes
    dim 0 (``dp_replicate``) the shard axis, instead of replicating on dim 0
    and sharding on dim 1 (``fsdp``). PyTorch's normal 2D fully_shard path
    uses ``HSDPMeshInfo`` with the correct axes.
    """
    global _original_llama4_fsdp_mesh_info
    if _original_llama4_fsdp_mesh_info is not _MISSING:
        return

    import torchtitan.models.llama4.parallelize as llama4_parallelize
    from torch.distributed.fsdp._fully_shard._fsdp_common import HSDPMeshInfo

    original = llama4_parallelize.FSDPMeshInfo
    _original_llama4_fsdp_mesh_info = original

    def _hsdp_aware_mesh_info(mesh, *args, **kwargs):  # noqa: ANN001, ANN202
        if mesh.ndim == 2:
            return HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        return original(mesh, *args, **kwargs)

    llama4_parallelize.FSDPMeshInfo = _hsdp_aware_mesh_info


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

    # When multi_proc_meta mode creates a fake world_mesh (device_type="fake"),
    # DTensor's sharding propagation calls _get_device_handle("fake") which
    # does getattr(torch, "fake", None).  Without this, it returns None and
    # crashes on device_count()/current_device().  Point torch.fake to the
    # same meta stub so all device queries are no-ops.
    _original_values[("torch", "fake")] = getattr(torch, "fake", _MISSING)
    torch.fake = stub

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
    _patch_torch_split_for_meta_dtensor()
    _patch_window_exchange_for_fake_pg()
    _patch_redistribute_local_tensor_for_meta()
    _patch_object_collectives_for_fake_pg()
    _patch_pipeline_stage_meta_exchange_for_fake_pg()
    _patch_pipeline_stage_for_pp_context()
    _patch_metadata_inference_skip()
    _patch_pipeline_action_context()
    _patch_torch_equal_for_meta()
    _patch_init_distributed_for_multi_proc_meta()
    _patch_fsdp_get_device_from_mesh()
    _patch_dtensor_random_for_fake_mesh()
    _patch_comm_layer_context()
    _patch_mxfp8_for_meta()
    _patch_llama4_hsdp_ep_mesh_info()
    _patch_fused_adamw_for_meta()
    global _is_meta_simulation
    _is_meta_simulation = True
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched, _is_meta_simulation
    global _original_fsdp_validate_no_meta_params, _original_tensor_npu_method, _original_torch_full
    global _original_moe_token_dispatch, _original_grouped_mm, _original_li_loss_forward, _original_pipeline_schedule_warmup_p2p
    global _original_window_exchange, _original_dtensor_meta_to_dtensor, _original_rowwise_prepare_output
    global _original_torch_split
    global _original_redistribute_local_tensor, _original_recv_object_list, _original_send_object_list
    global _original_torch_equal
    global _original_fused_adamw
    global _original_llama4_fsdp_mesh_info
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

    if _original_torch_split is not _MISSING:
        torch.split = _original_torch_split
        _original_torch_split = _MISSING

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

    if _original_fused_adamw is not _MISSING:
        torch._fused_adamw_ = _original_fused_adamw
        _original_fused_adamw = _MISSING

    if _original_llama4_fsdp_mesh_info is not _MISSING:
        import torchtitan.models.llama4.parallelize as llama4_parallelize

        llama4_parallelize.FSDPMeshInfo = _original_llama4_fsdp_mesh_info
        _original_llama4_fsdp_mesh_info = _MISSING

    _is_meta_simulation = False
    _patched = False
