# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

import math
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import Any, TypedDict, cast

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed._tensor import DTensor
from torch.optim.optimizer import Optimizer, _get_scalar_dtype, _use_grad_for_differentiable
from torchtitan.components.checkpoint_utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.dist_muon import DistMuon
from torchtitan.tools.logging import logger

from torchtitan_npu.extensions.cpu_offload import runtime as clip_state
from torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw import CpuOffloadAdamW
from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging
from torchtitan_npu.extensions.novaswap import swap_api
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer

_ADAMW_SWAP_BUCKET_TIMES = 16
_ADAMW_SWAP_RESIDENT_TARGET_BUCKETS = 3


class _CheckpointMetadata(TypedDict):
    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def make_checkpointable_view(
    tensor_cpu: torch.Tensor,
    *,
    byte_offset: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    global_shape: tuple[int, ...],
    global_offsets: tuple[tuple[int, ...], ...],
    local_offsets: tuple[tuple[int, ...], ...],
    local_sizes: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    """Expose a logical tensor inside one raw NovaSwap CPU buffer to DCP."""
    if tensor_cpu.device.type != "cpu" or tensor_cpu.dtype != torch.uint8:
        raise TypeError("checkpoint view requires a CPU uint8 swap buffer")

    logical_nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if byte_offset < 0 or byte_offset + logical_nbytes > tensor_cpu.numel():
        raise ValueError(
            f"checkpoint view byte range [{byte_offset}, {byte_offset + logical_nbytes}) "
            f"exceeds the {tensor_cpu.numel()}-byte swap buffer"
        )

    raw_view = tensor_cpu.narrow(0, byte_offset, logical_nbytes)
    view = raw_view.view(dtype).as_strided(shape, stride)
    if not view.is_contiguous():
        raise ValueError("checkpoint view requires a compact contiguous NovaSwap layout")

    setattr(view, "global_shape", global_shape)  # noqa: B010
    setattr(view, "global_offsets", global_offsets)  # noqa: B010
    setattr(view, "local_offsets", local_offsets)  # noqa: B010
    setattr(view, "local_sizes", local_sizes)  # noqa: B010
    return view


def get_checkpoint_view(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    byte_offset: int = 0,
) -> torch.Tensor:
    """Build a zero-copy DCP view of one swapped optimizer-state tensor."""
    tensor_cpu = swap_api.get_d2h_cpu_buffer(tensor_name)
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return make_checkpointable_view(
        tensor_cpu,
        byte_offset=byte_offset,
        dtype=local.dtype,
        shape=tuple(local.shape),
        stride=tuple(local.stride()),
        **_checkpoint_metadata(tensor, local),
    )


def _checkpoint_metadata(tensor: torch.Tensor, local: torch.Tensor) -> _CheckpointMetadata:
    if isinstance(tensor, DTensor):
        chunks = tensor.__create_chunk_list__()
        if len(chunks) != 1:
            raise RuntimeError(
                f"CheckpointableTensor currently requires exactly one local DTensor chunk, got {len(chunks)}"
            )
        chunk = chunks[0]
        global_offsets = (tuple(chunk.offsets),)
        local_sizes = (tuple(chunk.sizes),)
        global_shape = tuple(tensor.shape)
    else:
        global_offsets = (tuple(0 for _ in local.shape),)
        local_sizes = (tuple(local.shape),)
        global_shape = tuple(local.shape)

    return {
        "global_shape": global_shape,
        "global_offsets": global_offsets,
        "local_offsets": (tuple(0 for _ in local.shape),),
        "local_sizes": local_sizes,
    }


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _replace_local_tensor(tensor: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, DTensor):
        return local
    return DTensor.from_local(
        local,
        tensor.device_mesh,
        tensor.placements,
        shape=tensor.size(),
        stride=tensor.stride(),
        run_check=False,
    )


def make_swap_state_name(*parts: object) -> str:
    return ".".join(map(str, parts))


@dataclass
class _AdamWSwapBucket:
    group: dict[str, Any]
    parameters: tuple[torch.Tensor, ...]
    state_name: str
    state_numel: int
    state_nbytes: int
    flat: torch.Tensor | None = None


_SwappedState = tuple[torch.Tensor, str, str, torch.Tensor]


class _NovaSwapAdamW:
    """Run stock AdamW state through NovaSwap with a bucket-level pipeline."""

    def __init__(self, optimizer: torch.optim.AdamW) -> None:
        self.optimizer = optimizer
        self._original_step = optimizer.step
        self._buckets: list[_AdamWSwapBucket] = []
        self._target_bucket_bytes = 1
        self._largest_bucket_bytes = 1
        self.checkpoint_locations: dict[tuple[torch.Tensor, str], tuple[str, int]] = {}

    @staticmethod
    def _submit(bucket: _AdamWSwapBucket, action: str) -> None:
        swap_api.execute(bucket.state_name, action)

    @staticmethod
    def _state_keys(group: dict[str, Any]) -> tuple[str, ...]:
        if group["amsgrad"]:
            return "exp_avg", "exp_avg_sq", "max_exp_avg_sq"
        return "exp_avg", "exp_avg_sq"

    @staticmethod
    def _initialize_step(group: dict[str, Any], device: torch.device) -> torch.Tensor:
        if group["differentiable"]:
            raise ValueError("AdamW NovaSwap only supports differentiable=False")
        if group["foreach"]:
            raise ValueError("AdamW NovaSwap only supports foreach=False")
        # Match AdamW._init_group: fused/capturable paths keep step on-device;
        # otherwise the scalar remains on CPU. Only the moment allocation is
        # replaced by the bucket flat storage above.
        if group["capturable"] or group["fused"]:
            return torch.zeros(
                (),
                dtype=_get_scalar_dtype(is_fused=group["fused"]),
                device=device,
            )
        return torch.tensor(0.0, dtype=_get_scalar_dtype(), device="cpu")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        new_buckets = self._plan_uninitialized_buckets()
        device_budget = _ADAMW_SWAP_RESIDENT_TARGET_BUCKETS * max(
            self._target_bucket_bytes,
            self._largest_bucket_bytes,
        )
        pending_releases: deque[_AdamWSwapBucket] = deque()
        resident_bytes = 0

        def reserve(bucket: _AdamWSwapBucket) -> None:
            nonlocal resident_bytes
            while pending_releases and resident_bytes + bucket.state_nbytes > device_budget:
                oldest = pending_releases.popleft()
                swap_api.wait_for_device_release(oldest.state_name)
                resident_bytes -= oldest.state_nbytes
            if resident_bytes + bucket.state_nbytes > device_budget:
                raise RuntimeError("AdamW swap resident-state accounting exceeded the device budget")
            resident_bytes += bucket.state_nbytes

        def submit_h2d(bucket: _AdamWSwapBucket) -> None:
            reserve(bucket)
            swap_api.wait_for_device_release(bucket.state_name)
            self._submit(bucket, "H2D")

        for bucket in new_buckets:
            reserve(bucket)
            self._initialize_bucket_state(bucket)
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
            pending_releases.append(bucket)
        self._buckets.extend(new_buckets)

        new_bucket_ids = {id(bucket) for bucket in new_buckets}
        active_buckets = [
            bucket
            for bucket in self._buckets
            if id(bucket) not in new_bucket_ids and any(parameter.grad is not None for parameter in bucket.parameters)
        ]
        if not active_buckets:
            return loss

        submit_h2d(active_buckets[0])
        for bucket_index, bucket in enumerate(active_buckets):
            self._submit(bucket, "WAIT_DEVICE")
            if bucket_index + 1 < len(active_buckets):
                submit_h2d(active_buckets[bucket_index + 1])
            self._update_bucket(bucket)
            self._submit(bucket, "D2H")
            pending_releases.append(bucket)
        return loss

    def ensure_all_state(self) -> None:
        """Create only missing state, one bucket at a time, for checkpoint I/O."""
        new_buckets = self._plan_uninitialized_buckets(include_without_grad=True)
        if not new_buckets:
            return

        # A checkpoint is a synchronization point. Retire prior D2H storage
        # before allocating a missing-state bucket so the device peak stays bounded.
        for bucket in self._buckets:
            swap_api.wait_for_device_release(bucket.state_name)

        for bucket in new_buckets:
            # Checkpoint materialization must reproduce AdamW's lazy state
            # allocation, not perform a synthetic update.  Calling AdamW.step
            # with zero gradients increments ``state['step']`` and changes the
            # first real update's bias correction relative to no-swap AdamW.
            self._initialize_bucket_state(bucket)

            self._submit(bucket, "D2H")
            swap_api.wait_for_device_release(bucket.state_name)
            self._buckets.append(bucket)

    def refresh_bucket_groups(self) -> None:
        group_by_parameter = {
            id(parameter): group for group in self.optimizer.param_groups for parameter in group["params"]
        }
        for bucket in self._buckets:
            bucket.group = group_by_parameter[id(bucket.parameters[0])]

    def _plan_uninitialized_buckets(
        self,
        *,
        include_without_grad: bool = False,
    ) -> list[_AdamWSwapBucket]:
        """Plan only states that PyTorch AdamW would lazily create this step."""
        uninitialized = []
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                should_initialize = parameter.grad is not None or (include_without_grad and parameter.requires_grad)
                if should_initialize and not self.optimizer.state[parameter]:
                    uninitialized.append((group, parameter))
        if not uninitialized:
            return []

        total_state_bytes = sum(
            self._state_numel(group, parameter) * _local_tensor(parameter).element_size()
            for group, parameter in uninitialized
        )
        bucket_byte_limit = max(total_state_bytes // _ADAMW_SWAP_BUCKET_TIMES, 1)
        self._target_bucket_bytes = max(self._target_bucket_bytes, bucket_byte_limit)
        buckets: list[_AdamWSwapBucket] = []
        current_group: dict[str, Any] | None = None
        current_parameters: list[torch.Tensor] = []
        current_state_bytes = 0
        current_signature: tuple[torch.device, torch.dtype] | None = None

        def append_bucket() -> None:
            if not current_parameters:
                return
            if current_group is None:
                raise RuntimeError("AdamW swap bucket has no parameter group")
            state_numel = sum(self._state_numel(current_group, parameter) for parameter in current_parameters)
            state_nbytes = state_numel * _local_tensor(current_parameters[0]).element_size()
            self._largest_bucket_bytes = max(self._largest_bucket_bytes, state_nbytes)
            buckets.append(
                _AdamWSwapBucket(
                    group=current_group,
                    parameters=tuple(current_parameters),
                    state_name=make_swap_state_name(
                        "adamw", id(self.optimizer), "bucket", len(self._buckets) + len(buckets)
                    ),
                    state_numel=state_numel,
                    state_nbytes=state_nbytes,
                )
            )

        for group, parameter in uninitialized:
            local_parameter = _local_tensor(parameter)
            parameter_signature = (local_parameter.device, local_parameter.dtype)
            parameter_state_bytes = self._state_numel(group, parameter) * local_parameter.element_size()
            if current_parameters:
                same_group = group is current_group
                fits_bucket = current_state_bytes + parameter_state_bytes <= bucket_byte_limit
                same_signature = parameter_signature == current_signature
                if not (same_group and fits_bucket and same_signature):
                    append_bucket()
                    current_parameters = []
                    current_state_bytes = 0
            current_group = group
            current_signature = parameter_signature
            current_parameters.append(parameter)
            current_state_bytes += parameter_state_bytes
        append_bucket()
        return buckets

    def _state_numel(self, group: dict[str, Any], parameter: torch.Tensor) -> int:
        return len(self._state_keys(group)) * _local_tensor(parameter).numel()

    def _initialize_bucket_state(self, bucket: _AdamWSwapBucket) -> None:
        """Install flat NPU state views before the first stock AdamW update."""
        if bucket.flat is not None:
            raise RuntimeError(f"AdamW swap bucket {bucket.state_name!r} is already initialized")
        first_local = _local_tensor(bucket.parameters[0])
        flat = torch.zeros(bucket.state_numel, dtype=first_local.dtype, device=first_local.device)
        offset = 0
        state_keys = self._state_keys(bucket.group)
        for parameter in bucket.parameters:
            state = self.optimizer.state[parameter]
            if state:
                raise RuntimeError("AdamW swap attempted to initialize an existing optimizer state")
            local_parameter = _local_tensor(parameter)
            if (local_parameter.device, local_parameter.dtype) != (first_local.device, first_local.dtype):
                raise RuntimeError("AdamW swap bucket parameters must share a device and dtype")
            state["step"] = self._initialize_step(bucket.group, first_local.device)
            for state_key in state_keys:
                view = flat.narrow(0, offset, local_parameter.numel()).view_as(local_parameter)
                state[state_key] = _replace_local_tensor(parameter, view)
                self.checkpoint_locations[(parameter, state_key)] = (
                    bucket.state_name,
                    int(view.storage_offset()) * view.element_size(),
                )
                offset += local_parameter.numel()
        if offset != bucket.state_numel:
            raise RuntimeError("AdamW swap bucket state size does not match its planned layout")
        bucket.flat = flat
        swap_api.register_tensor(flat, bucket.state_name)

    def _update_bucket(self, bucket: _AdamWSwapBucket) -> None:
        original_param_groups = self.optimizer.param_groups
        pre_hooks = self.optimizer._optimizer_step_pre_hooks
        post_hooks = self.optimizer._optimizer_step_post_hooks
        saved_pre_hooks = pre_hooks.copy()
        saved_post_hooks = post_hooks.copy()
        self.optimizer.param_groups = [{**bucket.group, "params": bucket.parameters}]
        # The public, instance-level wrapper retains PyTorch's step hooks once
        # per logical optimizer step. Calling the saved stock step per bucket
        # must not replay those external hooks for every bucket.
        pre_hooks.clear()
        post_hooks.clear()
        try:
            self._original_step()
        finally:
            self.optimizer.param_groups = original_param_groups
            pre_hooks.update(saved_pre_hooks)
            post_hooks.update(saved_post_hooks)


def _make_swap(t: torch.Tensor) -> torch.Tensor:
    local = t.to_local() if isinstance(t, DTensor) else t
    # A DTensor may have p.numel() > 0 globally but local.numel() == 0 on ranks
    # that own an empty shard. Swap-memory allocation rejects zero-sized tensors,
    # so preserve such local shards with a regular empty tensor instead.
    out = (
        torch.empty_like(local)
        if local.numel() == 0
        else torch_npu.empty_with_swapped_memory(local.size(), dtype=local.dtype, device=local.device)
    )
    return _replace_local_tensor(t, out)


def _initialize_swap_state(optimizer, *, only_with_grad=True):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if only_with_grad and p.grad is None:
                continue
            state = optimizer.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=p.device,
                )
                state["exp_avg"] = _make_swap(p).zero_()
                state["exp_avg_sq"] = _make_swap(p).zero_()
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = _make_swap(p).zero_()


def _swap_state_init_hook(optimizer, args, kwargs):
    _initialize_swap_state(optimizer)


class VirtualOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts):
        super().__init__(config=config, model_parts=model_parts)
        for opt in self._swap_optimizers():
            opt.register_step_pre_hook(_swap_state_init_hook)

    def _swap_optimizers(self):
        return (opt for opt in self.optimizers if isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)))

    def state_dict(self):
        # DCP also needs state for parameters that were unused in training.
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
        return super().state_dict()

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict may replace the moment tensors. Preserve
        # swap-backed destinations even when loading ordinary checkpoint tensors.
        destinations = []
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
            for group in opt.param_groups:
                for param in group["params"]:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in opt.state[param]:
                            destinations.append((opt, param, key, opt.state[param][key]))
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for opt, param, key, destination in destinations:
                destination.copy_(opt.state[param][key])
                opt.state[param][key] = destination


class VirtualHostSparseOptimizersContainer(VirtualOptimizersContainer, HostSparseOptimizersContainer):
    """Swap dense Adam states while retaining Host sparse-table lifecycle hooks."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config, VirtualOptimizersContainer.Config):
        pass


@override(
    target=OptimizersContainer.Config,
    description="Allocate Adam/AdamW states in swap memory (host-offload)",
)
def virtual(
    cfg: OptimizersContainer.Config,
) -> VirtualOptimizersContainer.Config | VirtualHostSparseOptimizersContainer.Config:
    if isinstance(cfg, HostSparseOptimizersContainer.Config):
        return derive(cfg, VirtualHostSparseOptimizersContainer.Config)
    return derive(cfg, VirtualOptimizersContainer.Config)


def _build_cpu_offload_adamw(params, **kwargs: Any) -> CpuOffloadAdamW:
    """Adapt master optimizer implementation flags to the CPU-offload kernel."""
    # The master container always adds ``fused``/``foreach`` according to
    # ``optimizer.implementation``.  CPU-offload uses a staged functional
    # update and therefore owns those implementation choices internally.
    kwargs.pop("fused", None)
    kwargs.pop("foreach", None)
    params = (params,) if isinstance(params, dict) else params
    normalized_params = [
        {key: value for key, value in group.items() if key not in {"fused", "foreach"}}
        if isinstance(group, dict)
        else group
        for group in params
    ]
    return CpuOffloadAdamW(normalized_params, **kwargs)


class CpuOffloadOptimizersContainer(OptimizersContainer):
    """Keep canonical optimizer tensors on CPU and compute updates on NPU."""

    # Whether optimizer state joins the CPU-offload plane. ``False`` keeps the
    # moments NPU-resident, updating in place with no per-step staging.
    _offload_states = True

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts) -> None:
        # The CPU-offload override is the single owner of the FSDP and
        # gradient-clip patches: they are imported and installed only when
        # the user selects ``cpu_offload``. Unselected runs never import
        # these modules and keep pristine upstream behavior.
        from torchtitan_npu.extensions.distributed import grad_accum, grad_clip

        grad_accum.install()
        grad_accum.register_cpu_offload_hooks()
        grad_clip.install()
        self._compute_device = torch.device(  # pyrefly: ignore [read-only]
            "npu",
            torch_npu.npu.current_device(),
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._staging = CpuStaging(
            self._compute_device,
            owner=f"optimizer.rank{rank}",
        )
        # The channel is the explicit clip->optimizer handoff: injected into
        # the optimizers below and exposed to the patched clip through the
        # module-level active handle (see GradientClipChannel).
        self._clip_channel = clip_state.GradientClipChannel(
            preserve_coefficient_dtype=isinstance(self, HostSparseOptimizersContainer),
        )
        clip_state.set_active_channel(self._clip_channel)
        self._closed = False
        super().__init__(config=config, model_parts=model_parts)
        for optimizer in self.optimizers:
            materialize = getattr(optimizer, "materialize_state", None)
            if materialize is not None:
                materialize()
        if self.optimizers:
            self._clip_channel.register_consumer()
        logger.info(
            "CPU offload: optimizer state %s",
            "CPU-canonical (staged per step)" if self._offload_states else "NPU-resident",
        )

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            self.close()

    def close(self) -> None:
        """Release every CPU-offload runtime resource owned here.

        The container is the single owner: its staging lane, the clip
        channel, and the module-level FSDP and clip pipelines. Idempotent;
        the trainer teardown calls this on both normal and exception paths,
        ``__del__`` is only a fallback.
        """
        if self._closed:
            return
        self._closed = True
        from torchtitan_npu.extensions.distributed import grad_accum, grad_clip

        self._clip_channel.close()
        clip_state.set_active_channel(None)
        self._staging.close()
        grad_accum.clear()
        grad_clip.clear()
        grad_accum.unregister_cpu_offload_hooks()

    def state_dict(self) -> dict[str, Any]:
        self._staging.wait()
        result = super().state_dict()
        self._staging.wait()
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._clip_channel.clear_pending()
        self._staging.wait()
        super().load_state_dict(state_dict)
        self._staging.wait()

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        try:
            return super().step(closure)
        finally:
            self._clip_channel.clear_pending()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._clip_channel.clear_pending()
        self._staging.wait()
        super().zero_grad(set_to_none=set_to_none)

    def _resolve_optimizer_factory(self, name: str) -> Callable[..., Optimizer]:
        if name == "SparseAdam":
            return torch.optim.SparseAdam
        if name in {"DistributedMuon", "DistMuon"}:
            from torchtitan_npu.extensions.cpu_offload.cpu_offload_muon import (
                build_cpu_offload_distributed_muon,
            )

            return partial(
                build_cpu_offload_distributed_muon,
                staging=self._staging,
                clip_channel=self._clip_channel,
                offload_states=self._offload_states,
            )
        if name == "AdamW":
            return partial(
                _build_cpu_offload_adamw,
                staging=self._staging,
                clip_channel=self._clip_channel,
                offload_states=self._offload_states,
            )
        raise ValueError(f"CPU-offload optimizer does not support {name}")


class CpuOffloadHostSparseOptimizersContainer(  # pyrefly: ignore [inconsistent-inheritance]
    CpuOffloadOptimizersContainer, HostSparseOptimizersContainer
):
    """Stage dense updates on NPU while retaining CPU sparse-table updates."""

    # CpuOffload goes first to own staging/channel setup; Config puts
    # HostSparse first so materialize() keeps SparseAdam groups ahead of Muon.
    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config, CpuOffloadOptimizersContainer.Config):
        pass

    def _scale_dense_gradients(self, parameters, correction: float) -> None:
        # Offloaded dense gradients may already be cached on NPU. Correct the
        # deferred coefficient, not the CPU copy that the optimizer will bypass.
        if not self._clip_channel.rescale_pending(correction):
            super()._scale_dense_gradients(parameters, correction)


class CpuOffloadNpuStateOptimizersContainer(CpuOffloadOptimizersContainer):
    """Offload the data plane (parameters+gradients) while optimizer state
    stays NPU-resident: moments update in place on the compute device and
    never round-trip through staging buffers.

    Selected by ``--training.enable-cpu-offload`` on its own; listing the
    ``swap_optimizer`` (or ``cpu_offload``) override re-derives the node to
    the CPU-canonical-state container instead.
    """

    _offload_states = False

    @dataclass(kw_only=True, slots=True)
    class Config(CpuOffloadOptimizersContainer.Config):
        # Carrier so a later swap_optimizer/cpu_offload override still sees
        # the flag after ``derive`` replaced this node (source-only fields
        # drop). Kept off the shared base: a slotted field there conflicts
        # with the HostSparse Config's multiple-inheritance layout.
        _cpu_offload: bool = False


class CpuOffloadHostSparseNpuStateOptimizersContainer(
    CpuOffloadNpuStateOptimizersContainer, CpuOffloadHostSparseOptimizersContainer
):
    """HostSparse lifecycle with NPU-resident optimizer state.

    The HostSparse counterpart of ``CpuOffloadNpuStateOptimizersContainer``
    for schemas that own host sparse tables (Engram): the dense moments stay
    on the compute device while the sparse-table hooks (clip, refresh,
    checkpoint materialization) keep running. Listing ``swap_optimizer``
    re-derives to ``CpuOffloadHostSparseOptimizersContainer`` as before.

    ``CpuOffloadHostSparseOptimizersContainer`` is a base so method
    resolution picks up its offload-aware ``_scale_dense_gradients`` (the
    plain HostSparse fallback would rescale a CPU gradient copy this
    optimizer never reads).

    Note that this container's Config deliberately inherits only from
    ``HostSparseOptimizersContainer.Config`` and is therefore NOT a
    ``CpuOffloadNpuStateOptimizersContainer.Config`` subtype, unlike the
    container itself. Config-level checks for the state plane must use the
    container's ``_offload_states`` attribute instead of isinstance on the
    Config.
    """

    _offload_states = False

    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config):  # pyrefly: ignore [bad-override]
        # Single base: HostSparse's Config already carries the
        # ``_cpu_offload`` carrier through OptimizerConfig, and adding the
        # NPU-state Config as a second base would duplicate that slot name
        # across independent branches and break the dataclass layout. The
        # state plane comes from this class's ``_offload_states``.
        pass


@override(
    target=OptimizersContainer.Config,
    description="Keep optimizer canonical state on CPU and execute updates on NPU",
)
def cpu_offload(
    cfg: OptimizersContainer.Config,
) -> CpuOffloadOptimizersContainer.Config:
    if not getattr(cfg, "_cpu_offload", False):
        raise ValueError(
            "The cpu_offload optimizer override requires --training.enable-cpu-offload; "
            "without it FSDP keeps parameters on NPU, the CPU-offload optimizers cannot "
            "run, and gradients would clip through the bounded fallback path"
        )
    if isinstance(cfg, HostSparseOptimizersContainer.Config):
        return derive(cfg, CpuOffloadHostSparseOptimizersContainer.Config)
    return derive(cfg, CpuOffloadOptimizersContainer.Config)


class OptimizerStateSwapContainer(OptimizersContainer):
    supports_async_with_pinned_mem = False

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[Any]) -> None:
        super().__init__(config=config, model_parts=model_parts)
        parameter_to_part = {
            id(parameter): part_index
            for part_index, model_part in enumerate(model_parts)
            for parameter in model_part.parameters()
        }
        for optimizer in self.optimizers:
            if isinstance(optimizer, DistMuon):
                part_indices = {
                    parameter_to_part[id(parameter)]
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                if len(part_indices) != 1:
                    raise RuntimeError(
                        "DistMuon optimizer must own parameters from exactly "
                        f"one model part, got part indices {sorted(part_indices)}"
                    )
                self._swap_muon(optimizer, next(iter(part_indices)))
            elif isinstance(optimizer, torch.optim.AdamW):
                self._swap_adamw(optimizer)

    @staticmethod
    def _swap_muon(optimizer: DistMuon, model_part: int) -> None:
        checkpoint_locations: dict[tuple[torch.Tensor, str], tuple[str, int]] = {}
        setattr(optimizer, "_torchtitan_npu_checkpoint_locations", checkpoint_locations)  # noqa: B010
        original_momentum = optimizer._momentum
        original_prepare_local = optimizer._prepare_local
        runtime = optimizer._redistribution_runtime
        original_enqueue_storage_to_compute = runtime._enqueue_storage_to_compute
        deferred_d2h_names: list[str] | None = None

        def momentum(optimizer_self, compute_layout, grad):
            state = optimizer_self.state[compute_layout.param]
            created = "momentum_buffer" not in state
            result = original_momentum(compute_layout, grad)
            if not created:
                return result
            local = _local_tensor(result)
            if local.numel() == 0:
                return result
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            swap_api.register_tensor(local, name)
            checkpoint_locations[(compute_layout.param, "momentum_buffer")] = (
                name,
                0,
            )
            swap_api.execute(name, "D2H")
            return result

        def submit_h2d(compute_layout) -> None:
            state = optimizer.state.get(compute_layout.param)
            if state is None or "momentum_buffer" not in state:
                return
            momentum = state["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                return
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            if swap_api.get_handle_phase(name) != "H2D":
                swap_api.execute(name, "H2D")

        def prefetch_plan(plan) -> None:
            """Submit a plan's H2D work before FlexShard enters its transfer stream."""
            redistributed = getattr(plan, "redistributed_items", None)
            items = plan.items if redistributed is None else (*redistributed, *plan.unredistributed_items)
            for compute_layout in items:
                submit_h2d(compute_layout)

        def enqueue_storage_to_compute(plan, slot, context, *, prepare):
            """Align swap prefetch/offload with one FlexShard redistribution bucket."""
            nonlocal deferred_d2h_names
            if deferred_d2h_names is not None:
                raise RuntimeError("Nested FlexShard storage-to-compute enqueue is unsupported")

            prefetch_plan(plan)
            deferred_d2h_names = []
            try:
                work = original_enqueue_storage_to_compute(
                    plan,
                    slot,
                    context,
                    prepare=prepare,
                )
            except Exception:
                deferred_d2h_names = None
                raise

            completed_d2h_names = deferred_d2h_names
            deferred_d2h_names = None

            with torch_npu.npu.stream(context.transfer_stream):
                for name in completed_d2h_names:
                    swap_api.execute(name, "D2H")
            return work

        def prepare_local(optimizer_self, compute_layout, out) -> None:
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            momentum = optimizer_self.state[compute_layout.param]["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                original_prepare_local(compute_layout, out)
                return
            submit_h2d(compute_layout)
            swap_api.execute(name, "WAIT_DEVICE")
            original_prepare_local(compute_layout, out)

            if deferred_d2h_names is None:
                swap_api.execute(name, "D2H")
            else:
                deferred_d2h_names.append(name)

        optimizer._momentum = MethodType(momentum, optimizer)
        optimizer._prepare_local = MethodType(prepare_local, optimizer)
        runtime._enqueue_storage_to_compute = enqueue_storage_to_compute

    @staticmethod
    def _swap_adamw(optimizer: torch.optim.AdamW) -> None:
        if any(state for state in optimizer.state.values()):
            raise ValueError("AdamW NovaSwap must be installed before AdamW initializes optimizer state")
        swap = _NovaSwapAdamW(optimizer)
        setattr(optimizer, "_torchtitan_npu_checkpoint_locations", swap.checkpoint_locations)  # noqa: B010
        setattr(optimizer, "_torchtitan_npu_swap_adapter", swap)  # noqa: B010

        @Optimizer.profile_hook_step
        @_use_grad_for_differentiable
        def step(_optimizer, closure=None):
            return swap.step(closure)

        # Install the intentional instance-level AdamW bucket-swap wrapper.
        optimizer.step = MethodType(step, optimizer)  # pyrefly: ignore [bad-override]

    @staticmethod
    def _replace_swapped_states_with_checkpoint_views(
        optimizer: torch.optim.Optimizer,
        flat_state: dict[str, Any],
    ) -> None:
        if not _uses_novaswap(optimizer):
            # Host-resident optimizers keep their ordinary checkpoint tensors.
            return
        locations = getattr(optimizer, "_torchtitan_npu_checkpoint_locations")  # noqa: B009
        for parameter, fqn, state_name, tensor in _swapped_state_tensors(optimizer):
            key = f"state.{fqn}.{state_name}"
            local = _local_tensor(tensor)
            if local.numel() == 0:
                flat_state[key] = make_checkpointable_view(
                    torch.empty(0, dtype=torch.uint8),
                    byte_offset=0,
                    dtype=local.dtype,
                    shape=tuple(local.shape),
                    stride=tuple(local.stride()),
                    **_checkpoint_metadata(tensor, local),
                )
                continue
            location = locations.get((parameter, state_name))
            if location is None:
                raise RuntimeError(f"missing NovaSwap checkpoint location for optimizer state {key}")
            tensor_name, byte_offset = location
            flat_state[key] = get_checkpoint_view(
                tensor_name,
                tensor,
                byte_offset=byte_offset,
            )

    @staticmethod
    def _state_for_optimizer(
        state_dict: dict[str, Any],
        checkpoint_views: dict[str, Any],
        originals: list[_SwappedState],
    ) -> dict[str, Any]:
        state_for_optimizer = dict(state_dict)
        for _parameter, fqn, state_name, tensor in originals:
            key = f"state.{fqn}.{state_name}"
            incoming = state_dict.get(key)
            target_view = checkpoint_views[key]
            if incoming is not None and _local_tensor(tensor).numel() != 0:
                if not isinstance(incoming, torch.Tensor):
                    raise TypeError(f"optimizer state {key} must be a Tensor")
                # DCP has already populated the view when both references share an address.
                if incoming.data_ptr() != target_view.data_ptr():
                    target_view.copy_(incoming)
            # Keep the optimizer bound to its original NovaSwap NPU/DTensor object.
            state_for_optimizer[key] = tensor
        return state_for_optimizer

    @staticmethod
    def _restore_optimizer_references(
        optimizer: torch.optim.Optimizer,
        originals: list[_SwappedState],
        param_names: list[Any],
    ) -> None:
        for parameter, _fqn, state_name, tensor in originals:
            optimizer.state[parameter][state_name] = tensor
        for group, names in zip(optimizer.param_groups, param_names, strict=True):
            if names is not None:
                group["param_names"] = names

    @contextmanager
    def _unswapped_checkpoint_optimizers(self) -> Iterator[None]:
        optimizers = self.optimizers
        # Let the original container (including HostSparse) handle only its states.
        self.optimizers = [optimizer for optimizer in optimizers if not _uses_novaswap(optimizer)]
        try:
            yield
        finally:
            self.optimizers = optimizers

    def state_dict(self) -> dict[str, Any]:
        with self._unswapped_checkpoint_optimizers():
            result = super().state_dict() if self.optimizers else {}
        for optimizer in self.optimizers:
            if not _uses_novaswap(optimizer):
                continue
            _ensure_all_optim_state(optimizer)
            flat_state = get_flat_optim_state_dict(optimizer)
            self._replace_swapped_states_with_checkpoint_views(optimizer, flat_state)
            result.update(flat_state)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        with self._unswapped_checkpoint_optimizers():
            if self.optimizers:
                super().load_state_dict(state_dict)
        for optimizer in self.optimizers:
            if _uses_novaswap(optimizer):
                self._load_optimizer_state(optimizer, state_dict)

    def _load_optimizer_state(self, optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]) -> None:
        _ensure_all_optim_state(optimizer)
        originals = list(_swapped_state_tensors(optimizer))
        # DCP writes these target CPU views in place; direct load copies into them below.
        checkpoint_views = get_flat_optim_state_dict(optimizer)
        self._replace_swapped_states_with_checkpoint_views(optimizer, checkpoint_views)
        state_for_optimizer = self._state_for_optimizer(state_dict, checkpoint_views, originals)
        param_names = [group.get("param_names") for group in optimizer.param_groups]
        try:
            load_flat_optim_state_dict(optimizer, state_for_optimizer)
        finally:
            self._restore_optimizer_references(optimizer, originals, param_names)

        swap = getattr(optimizer, "_torchtitan_npu_swap_adapter", None)
        if isinstance(swap, _NovaSwapAdamW):
            swap.refresh_bucket_groups()


def _uses_novaswap(optimizer: torch.optim.Optimizer) -> bool:
    return getattr(optimizer, "_torchtitan_npu_checkpoint_locations", None) is not None


def _ensure_all_optim_state(optimizer: torch.optim.Optimizer) -> None:
    if isinstance(optimizer, DistMuon) and hasattr(optimizer, "_replica_dedup_all_layouts"):
        # Initialize only this replica's owned momentum without a full step.
        for compute_layout in optimizer._parameter_compute_layouts:
            state = optimizer.state[compute_layout.param]
            if "momentum_buffer" not in state:
                grad = cast("DTensor", torch.zeros_like(compute_layout.param))
                optimizer._momentum(compute_layout, grad)
        return
    swap = getattr(optimizer, "_torchtitan_npu_swap_adapter", None)
    if isinstance(swap, _NovaSwapAdamW):
        swap.ensure_all_state()
    else:
        init_optim_state(optimizer)


def _named_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> Iterator[tuple[torch.Tensor, str]]:
    for group in optimizer.param_groups:
        yield from zip(group["params"], group["param_names"], strict=True)


def _swapped_state_names(optimizer: torch.optim.Optimizer) -> tuple[str, ...]:
    if isinstance(optimizer, DistMuon):
        return ("momentum_buffer",)
    if isinstance(optimizer, torch.optim.AdamW):
        return ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    return ()


def _swapped_state_tensors(
    optimizer: torch.optim.Optimizer,
) -> Iterator[_SwappedState]:
    state_names = _swapped_state_names(optimizer)
    for parameter, fqn in _named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        for state_name in state_names:
            tensor = state.get(state_name)
            if tensor is not None:
                yield parameter, fqn, state_name, tensor


# Both branches inherit the same OptimizersContainer.step overloads; Python's
# MRO selects HostSparse's implementation before their common base.
class HostSparseOptimizerStateSwapContainer(  # pyrefly: ignore [inconsistent-inheritance]
    OptimizerStateSwapContainer, HostSparseOptimizersContainer
):
    """Apply the upstream dense-state swap while retaining Host sparse updates."""

    @dataclass(kw_only=True, slots=True)
    class Config(HostSparseOptimizersContainer.Config, OptimizerStateSwapContainer.Config):
        pass


@override(
    target=OptimizersContainer.Config,
    description=(
        "Optimizer offload strategy: CPU-offload container when "
        "--training.enable-cpu-offload is set, else optimizer state swap"
    ),
)
def swap_optimizer(
    cfg: OptimizersContainer.Config,
) -> OptimizersContainer.Config:
    if getattr(cfg, "_cpu_offload", False):
        return cpu_offload(cfg)
    if isinstance(cfg, HostSparseOptimizersContainer.Config):
        return derive(cfg, HostSparseOptimizerStateSwapContainer.Config)
    return derive(cfg, OptimizerStateSwapContainer.Config)
