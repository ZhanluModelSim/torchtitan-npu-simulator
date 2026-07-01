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

    _neutralize_torch_npu_optimizer_device_probe()
    _patch_swap_optimizer_get_device_info(stub)
    _patch_tensor_npu_method_to_meta()
    _patch_torch_full_npu_device_literal()
    _patch_grouped_mm_offsets_dtype()
    _patch_moe_dispatch_to_avoid_meta_tensor_value_reads()
    _patch_li_loss_to_skip_buggy_einsum()
    _neutralize_fsdp_meta_param_validation()
    _patched = True


def unpatch_device_type_to_meta() -> None:
    """Restore the original device_type/device_module bindings (test-only helper)."""
    global _patched, _original_fsdp_validate_no_meta_params, _original_tensor_npu_method, _original_torch_full
    global _original_moe_token_dispatch, _original_grouped_mm, _original_li_loss_forward
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

    _patched = False
