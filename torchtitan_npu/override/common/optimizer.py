# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides for NPU swap-backed optimizer states and their checkpoints."""

import logging
import math
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MethodType
from typing import Any, TypedDict, cast

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed._tensor import DTensor
from torch.optim.optimizer import Optimizer, _get_scalar_dtype, _use_grad_for_differentiable
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.optim.optimizer import BaseOptimizer
from torchtitan.components.optim.optimizer import DistMuon as ConfigurableDistMuon
from torchtitan.components.optim.utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.dist_muon import DistMuon

from torchtitan_npu.extensions.cpu_offload import runtime as clip_state
from torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw import CpuOffloadAdamW
from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging
from torchtitan_npu.extensions.novaswap import swap_api
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer

logger = logging.getLogger(__name__)

_ADAMW_SWAP_BUCKET_TIMES = 16
_ADAMW_SWAP_RESIDENT_TARGET_BUCKETS = 3
_MUON_SWAP_GROUP_MAX_BYTES = 512 * 1024 * 1024
_MUON_SWAP_RESIDENT_GROUPS = 2


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

    def __init__(self, config: Config, *, model_parts, enable_cuda_graph: bool = False):
        super().__init__(config=config, model_parts=model_parts, enable_cuda_graph=enable_cuda_graph)
        for opt in self._swap_optimizers():
            opt.register_step_pre_hook(_swap_state_init_hook)

    def _swap_optimizers(self):
        return (opt for opt in self.optimizers if isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)))

    def state_dict(self):
        # DCP also needs state for parameters that were unused in training.
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
        result = super().state_dict()
        for opt in self._swap_optimizers():
            for param, name in _named_optimizer_parameters(opt):
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = opt.state[param].get(key)
                    if value is None:
                        continue
                    local = _local_tensor(value)
                    # Direct D2H from swapped virtual memory can segfault in
                    # torch_npu. Clone one moment to ordinary device memory
                    # first; retain only the CPU snapshot for DCP. This costs
                    # an additional full host copy of the optimizer moments.
                    snapshot = local.detach().clone().cpu()
                    for field, metadata in _checkpoint_metadata(value, local).items():
                        setattr(snapshot, field, metadata)
                    result[f"state.{name}.{key}"] = snapshot
        return result

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict may replace the moment tensors. Preserve
        # swap-backed destinations even when loading ordinary checkpoint tensors.
        destinations = []
        state_for_optimizer = dict(state_dict)
        for opt in self._swap_optimizers():
            _initialize_swap_state(opt, only_with_grad=False)
            for group in opt.param_groups:
                for param, name in zip(group["params"], group["param_names"], strict=True):
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in opt.state[param]:
                            destination = opt.state[param][key]
                            flat_key = f"state.{name}.{key}"
                            source = state_dict.get(flat_key)
                            if source is not None:
                                destinations.append((opt, param, key, destination, source))
                                # Keep optimizer loading from allocating all
                                # moments on device at once or replacing swap
                                # storage; copy the loaded snapshots below.
                                state_for_optimizer[flat_key] = destination
        try:
            super().load_state_dict(state_for_optimizer)
            with torch.no_grad():
                for _, _, _, destination, source in destinations:
                    local = _local_tensor(destination)
                    incoming = _local_tensor(source)
                    # Direct CPU-to-swap copy is also unsupported. Materialize
                    # one ordinary device tensor before writing the swap view.
                    local.copy_(incoming.to(device=local.device))
        finally:
            for opt, param, key, destination, _ in destinations:
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
    """Preserve the selected NPU AdamW kernel with CPU canonical storage."""
    # Canonical CPU parameter groups use non-fused storage bookkeeping;
    # the selected fused kernel runs on the staged NPU work tensors.
    npu_fused = kwargs.pop("fused", None)
    kwargs.pop("foreach", None)
    params = (params,) if isinstance(params, dict) else params
    normalized_params = [
        {key: value for key, value in group.items() if key not in {"fused", "foreach"}}
        if isinstance(group, dict)
        else group
        for group in params
    ]
    return CpuOffloadAdamW(normalized_params, fused=npu_fused, **kwargs)


@dataclass(kw_only=True, slots=True)
class _CpuOffloadOptimizerConfig(BaseOptimizer.Config):
    """Bind staging resources while retaining upstream parameter selection."""

    optimizer: BaseOptimizer.Config
    staging: CpuStaging
    clip_channel: clip_state.GradientClipChannel
    offload_states: bool

    def build(self, **kwargs) -> Optimizer:
        params = kwargs.pop("params")
        if kwargs:
            raise TypeError(f"Unexpected CPU-offload optimizer build arguments: {sorted(kwargs)}")
        config = self.optimizer
        runtime = dict(staging=self.staging, clip_channel=self.clip_channel, offload_states=self.offload_states)
        if isinstance(config, AdamW.Config):
            if config.moment_dtype != "parameter":
                raise ValueError("CPU-offload AdamW requires moment_dtype='parameter'")
            return _build_cpu_offload_adamw(
                params,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                fused=config.fused,
                **runtime,
            )
        if isinstance(config, ConfigurableDistMuon.Config):
            from torchtitan_npu.extensions.cpu_offload.cpu_offload_muon import build_cpu_offload_distributed_muon

            return build_cpu_offload_distributed_muon(
                params,
                compute_sharding_by_fqn=config.compute_sharding_by_fqn,
                bucket_configs=config.bucket_configs,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
                ns_coefficients=config.ns_coefficients,
                eps=config.eps,
                ns_steps=config.ns_steps,
                adjust_lr_fn=config.adjust_lr_fn,
                enable_hsdp_replica_dedup=getattr(config, "enable_hsdp_replica_dedup", False),
                **runtime,
            )
        raise ValueError(f"CPU-offload optimizer does not support {type(config).__qualname__}")


class CpuOffloadOptimizersContainer(OptimizersContainer):
    """Keep canonical optimizer tensors on CPU and compute updates on NPU."""

    # Whether optimizer state joins the CPU-offload plane. ``False`` keeps the
    # moments NPU-resident, updating in place with no per-step staging.
    _offload_states = True

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts, enable_cuda_graph: bool = False) -> None:
        if enable_cuda_graph:
            raise ValueError("CPU-offload optimizers do not support CUDA graph capture")
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
        from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import SparseAdam

        # The upstream container owns first-match selection and validates that
        # every trainable parameter belongs to exactly one optimizer. Host
        # SparseAdam stays on CPU and retains its specialized FQN handling.
        optimizer_configs = [
            optimizer
            if isinstance(optimizer, SparseAdam.Config)
            else _CpuOffloadOptimizerConfig(
                pattern=optimizer.pattern,
                optimizer=optimizer,
                staging=self._staging,
                clip_channel=self._clip_channel,
                offload_states=self._offload_states,
            )
            for optimizer in config.optimizers
        ]
        try:
            super().__init__(config=replace(config, optimizers=optimizer_configs), model_parts=model_parts)
        except Exception:
            self.close()
            raise
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


class CpuOffloadHostSparseOptimizersContainer(  # pyrefly: ignore [inconsistent-inheritance]
    CpuOffloadOptimizersContainer, HostSparseOptimizersContainer
):
    """Stage dense updates on NPU while retaining CPU sparse-table updates."""

    # CpuOffload goes first to own staging/channel setup; Config puts
    # Preserve the HostSparse container for recipe-owned SparseAdam groups.
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


@dataclass(frozen=True, slots=True)
class _MuonSwapPlanKey:
    kind: str
    unredistributed_fqns: tuple[str, ...]
    redistributed_fqns: tuple[str, ...]


@dataclass(slots=True)
class _MuonSwapGroup:
    name: str
    kind: str
    plan_key: _MuonSwapPlanKey
    layouts: tuple[Any, ...]
    state_nbytes: int = 0
    registered_layout_ids: frozenset[int] = frozenset()
    first_consumer_layout_id: int | None = None
    last_consumer_layout_id: int | None = None
    registered: bool = False


class _MuonSwapAdapter:
    """Own Muon momentum in bounded groups without changing FlexShard plans."""

    def __init__(
        self, optimizer: DistMuon, model_part: int, locations: dict[tuple[torch.Tensor, str], tuple[str, int]]
    ):
        self.optimizer = optimizer
        self.locations = locations
        self.ordered_plans = tuple(getattr(optimizer, "_bucket_plans", ()))
        self.groups: dict[_MuonSwapPlanKey, tuple[_MuonSwapGroup, ...]] = {}
        self.owner: dict[int, _MuonSwapGroup] = {}
        self.payloads: list[_MuonSwapGroup] = []
        self.payload_index: dict[int, int] = {}
        self._resident_groups: deque[_MuonSwapGroup] = deque()
        self._resident_group_ids: set[int] = set()
        self._resident_bytes = 0
        for index, plan in enumerate(self.ordered_plans):
            key = self.key(plan)
            if key in self.groups:
                raise RuntimeError(f"duplicate Muon swap plan key: {key!r}")
            redistributed = getattr(plan, "redistributed_items", None)
            kinds = (
                (("local", tuple(plan.items)),)
                if redistributed is None
                else (
                    ("redistributed", tuple(redistributed)),
                    ("unredistributed", tuple(plan.unredistributed_items)),
                )
            )
            groups: list[_MuonSwapGroup] = []
            for kind, layouts in kinds:
                if not layouts:
                    continue
                for group_index, group_layouts in enumerate(self._split_group_layouts(layouts)):
                    group = _MuonSwapGroup(
                        f"muon.{model_part}.plan.{index}.{kind}.{group_index}",
                        kind,
                        key,
                        group_layouts,
                    )
                    groups.append(group)
                    self.payloads.append(group)
                    for layout in group_layouts:
                        if id(layout) in self.owner:
                            raise RuntimeError(f"Muon layout {layout.fqn!r} belongs to multiple swap payloads")
                        self.owner[id(layout)] = group
            self.groups[key] = tuple(groups)
        self.payload_index = {id(group): index for index, group in enumerate(self.payloads)}

    @staticmethod
    def key(plan: Any) -> _MuonSwapPlanKey:
        redistributed = getattr(plan, "redistributed_items", None)
        if redistributed is None:
            return _MuonSwapPlanKey("local", tuple(layout.fqn for layout in plan.items), ())
        return _MuonSwapPlanKey(
            "redistribution",
            tuple(layout.fqn for layout in plan.unredistributed_items),
            tuple(layout.fqn for layout in redistributed),
        )

    def _local(self, layout: Any) -> torch.Tensor | None:
        state = self.optimizer.state.get(layout.param)
        if state is None or "momentum_buffer" not in state:
            return None
        local = _local_tensor(state["momentum_buffer"])
        return local if local.numel() else torch.empty(0, device=local.device, dtype=local.dtype)

    def _layout_size(self, layout: Any) -> tuple[int, int]:
        local = self._local(layout)
        if local is None:
            # DistMuon creates momentum with zeros_like(grad); use the parameter before gradients exist.
            grad = layout.param.grad
            local = _local_tensor(grad if grad is not None else layout.param)
        return local.numel() * local.element_size(), local.element_size()

    def _split_group_layouts(self, layouts: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
        current: list[Any] = []
        current_bytes = 0
        for layout in layouts:
            layout_bytes, alignment = self._layout_size(layout)
            padding = -current_bytes % alignment if layout_bytes else 0
            if current and current_bytes + padding + layout_bytes > _MUON_SWAP_GROUP_MAX_BYTES:
                yield tuple(current)
                current = []
                current_bytes = 0
                padding = 0
            if layout_bytes > _MUON_SWAP_GROUP_MAX_BYTES:
                yield (layout,)
            else:
                current.append(layout)
                current_bytes += padding + layout_bytes
        if current:
            yield tuple(current)

    def finalize_registration(self, *, require_complete: bool) -> None:
        for plan_groups in self.groups.values():
            for group in plan_groups:
                if group.registered:
                    continue
                members = tuple((layout, self._local(layout)) for layout in group.layouts)
                missing = [layout.fqn for layout, local in members if local is None]
                if missing:
                    if require_complete:
                        raise RuntimeError(f"Muon group {group.name!r} is missing momentum state for {missing!r}")
                    continue
                nonempty = tuple((layout, local) for layout, local in members if local is not None and local.numel())
                if not nonempty:
                    group.registered = True
                    continue
                registration = swap_api.register_group(tuple(local for _, local in nonempty), group.name)
                member_ids = frozenset(id(layout) for layout, _ in nonempty)
                group.registered_layout_ids = member_ids
                group.first_consumer_layout_id = id(nonempty[0][0])
                group.last_consumer_layout_id = id(nonempty[-1][0])
                group.state_nbytes = registration.total_bytes
                for (layout, _), member in zip(nonempty, registration.members, strict=True):
                    self.locations[(layout.param, "momentum_buffer")] = (group.name, member.byte_offset)
                swap_api.offload(group.name)
                group.registered = True

    def materialize_current(
        self,
        plan: Any,
        materialize: Callable[[Any], None],
    ) -> None:
        """Create one plan's missing state without retaining all groups on NPU."""
        for group in self.groups.get(self.key(plan), ()):
            for layout in group.layouts:
                if self._local(layout) is None:
                    materialize(layout)
            self.finalize_registration(require_complete=False)
            if group.registered_layout_ids:
                self._wait_for_checkpoint_release(group.name)

    def materialize_group(self, layout: Any, materialize: Callable[[Any], None]) -> None:
        """Create the complete owning group before its first local use."""
        group = self.owner.get(id(layout))
        if group is None:
            materialize(layout)
            return
        for candidate in group.layouts:
            if self._local(candidate) is None:
                materialize(candidate)
        self.finalize_registration(require_complete=False)
        if group.registered_layout_ids:
            self._wait_for_checkpoint_release(group.name)

    def _wait_for_checkpoint_release(self, tensor_name: str) -> None:
        """Retire one checkpoint payload before materializing the next one."""
        if swap_api.get_handle_phase(tensor_name) != "D2H":
            swap_api.offload(tensor_name)
        swap_api.wait_for_device_release(tensor_name)

    def materialize_all_for_checkpoint(self, materialize: Callable[[Any], None]) -> None:
        """Create DCP targets group-by-group without whole-optimizer residency."""
        for plan in self.ordered_plans:
            plan_groups = self.groups.get(self.key(plan), ())
            for group in plan_groups:
                for layout in group.layouts:
                    if self._local(layout) is None:
                        materialize(layout)
                self.finalize_registration(require_complete=False)
                if group.registered_layout_ids:
                    self._wait_for_checkpoint_release(group.name)

        self.finalize_registration(require_complete=True)

    def group_for(self, layout: Any) -> _MuonSwapGroup | None:
        group = self.owner.get(id(layout))
        return group if group is not None and id(layout) in group.registered_layout_ids else None

    @property
    def _resident_budget(self) -> int:
        # Conservatively reserve both state and equally sized H2D staging for each resident group.
        return 2 * _MUON_SWAP_RESIDENT_GROUPS * _MUON_SWAP_GROUP_MAX_BYTES

    def _reserve(self, group: _MuonSwapGroup) -> None:
        if id(group) in self._resident_group_ids:
            return
        required_bytes = 2 * group.state_nbytes
        budget = max(self._resident_budget, required_bytes)
        while self._resident_groups and self._resident_bytes + required_bytes > budget:
            oldest = self._resident_groups[0]
            assert swap_api.get_handle_phase(oldest.name) == "D2H", "Cannot evict a live Muon group"
            swap_api.wait_for_device_release(oldest.name)
            self._resident_groups.popleft()
            self._resident_group_ids.remove(id(oldest))
            self._resident_bytes -= 2 * oldest.state_nbytes
        self._resident_groups.append(group)
        self._resident_group_ids.add(id(group))
        self._resident_bytes += required_bytes

    def restore(self, group: _MuonSwapGroup) -> None:
        self.finalize_registration(require_complete=False)
        if not group.registered_layout_ids:
            return
        if swap_api.get_handle_phase(group.name) == "D2H":
            self._reserve(group)
            swap_api.restore(group.name)

    def first_consumed_group(self, plan: Any) -> _MuonSwapGroup | None:
        self.finalize_registration(require_complete=False)
        for group in self.groups.get(self.key(plan), ()):
            if group.registered_layout_ids:
                return group
        return None

    def prefetch_first_consumed_group(self, plan: Any) -> None:
        group = self.first_consumed_group(plan)
        if group is not None:
            self.restore(group)

    def prefetch_first_plan(self) -> None:
        if self.ordered_plans:
            self.prefetch_first_consumed_group(self.ordered_plans[0])

    def next_consumed_group(self, group: _MuonSwapGroup) -> _MuonSwapGroup | None:
        self.finalize_registration(require_complete=False)
        start = self.payload_index[id(group)] + 1
        for candidate in self.payloads[start:]:
            if candidate.registered_layout_ids:
                return candidate
        return None

    def prefetch_successor(self, group: _MuonSwapGroup) -> None:
        successor = self.next_consumed_group(group)
        if successor is not None:
            # Load oversized singletons on demand instead of overlapping their residency.
            if max(group.state_nbytes, successor.state_nbytes) > _MUON_SWAP_GROUP_MAX_BYTES:
                return
            self.restore(successor)

    def wait_for(self, layout: Any) -> _MuonSwapGroup | None:
        group = self.group_for(layout)
        if group is not None:
            self.restore(group)
            swap_api.wait_device_if_needed(group.name)
        return group

    def finish_prepare(self, group: _MuonSwapGroup, layout: Any) -> None:
        if id(layout) == group.last_consumer_layout_id:
            swap_api.offload(group.name)


class OptimizerStateSwapContainer(OptimizersContainer):
    supports_async_with_pinned_mem = False

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[Any], enable_cuda_graph: bool = False) -> None:
        if any(
            isinstance(optimizer, AdamW.Config) and optimizer.moment_dtype != "parameter"
            for optimizer in config.optimizers
        ):
            # BF16 moment hooks allocate state before NovaSwap installs its bucket views.
            raise ValueError("AdamW NovaSwap requires moment_dtype='parameter'")
        super().__init__(config=config, model_parts=model_parts, enable_cuda_graph=enable_cuda_graph)
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
        original_preflight_step = getattr(optimizer, "_preflight_step", None)
        runtime = optimizer._redistribution_runtime
        original_enqueue_storage_to_compute = runtime._enqueue_storage_to_compute
        adapter = _MuonSwapAdapter(optimizer, model_part, checkpoint_locations)
        setattr(optimizer, "_torchtitan_npu_muon_swap_adapter", adapter)  # noqa: B010
        deferred_d2h_names: list[str] | None = None
        bootstrap_active = original_preflight_step is not None and not optimizer._first_step_validated
        bootstrap_complete = False

        def has_all_momentum() -> bool:
            return all(
                "momentum_buffer" in optimizer.state[layout.param] for layout in optimizer._parameter_compute_layouts
            )

        def finish_bootstrap_if_complete() -> None:
            nonlocal bootstrap_complete
            if not bootstrap_active or bootstrap_complete or not has_all_momentum():
                return
            adapter.finalize_registration(require_complete=True)
            optimizer._first_step_validated = True
            bootstrap_complete = True

        def preflight_step(optimizer_self) -> None:
            nonlocal bootstrap_active
            preflight = original_preflight_step
            assert preflight is not None
            if not bootstrap_active:
                preflight()
            elif has_all_momentum():
                adapter.finalize_registration(require_complete=True)
                bootstrap_active = False
                preflight()
            else:

                def validate_only_momentum(_optimizer_self, _compute_layout, _grad):
                    return None

                materializing_momentum = optimizer_self._momentum
                optimizer_self._momentum = MethodType(validate_only_momentum, optimizer_self)
                try:
                    preflight()
                finally:
                    optimizer_self._momentum = materializing_momentum
                optimizer_self._first_step_validated = False
            adapter.prefetch_first_plan()

        def momentum(optimizer_self, compute_layout, grad):
            state = optimizer_self.state[compute_layout.param]
            created = "momentum_buffer" not in state
            result = original_momentum(compute_layout, grad)
            if not created:
                return result
            if id(compute_layout) in adapter.owner:
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
                swap_api.restore(name)

        def enqueue_storage_to_compute(plan, slot, context, *, prepare):
            """Align swap prefetch/offload with one FlexShard redistribution bucket."""
            nonlocal deferred_d2h_names
            if deferred_d2h_names is not None:
                raise RuntimeError("Nested FlexShard storage-to-compute enqueue is unsupported")

            if bootstrap_active:

                def materialize(layout) -> None:
                    grad = layout.param.grad
                    if grad is None:
                        raise RuntimeError(f"DistMuon requires gradient before materializing {layout.fqn!r}")
                    optimizer._momentum(layout, grad)

                adapter.materialize_current(plan, materialize)
                finish_bootstrap_if_complete()
            adapter.prefetch_first_consumed_group(plan)
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

            if completed_d2h_names:
                with torch_npu.npu.stream(context.transfer_stream):
                    for name in completed_d2h_names:
                        swap_api.execute(name, "D2H")
            return work

        def prepare_local(optimizer_self, compute_layout, out) -> None:
            name = make_swap_state_name("optimizer_state", model_part, compute_layout.fqn, "momentum_buffer")
            state = optimizer_self.state[compute_layout.param]
            if "momentum_buffer" not in state:

                def materialize(layout) -> None:
                    grad = layout.param.grad
                    if grad is None:
                        raise RuntimeError(f"DistMuon requires gradient before materializing {layout.fqn!r}")
                    optimizer_self._momentum(layout, grad)

                adapter.materialize_group(compute_layout, materialize)
                finish_bootstrap_if_complete()
                state = optimizer_self.state[compute_layout.param]
            momentum = state["momentum_buffer"]
            local = momentum.to_local() if isinstance(momentum, DTensor) else momentum
            if local.numel() == 0:
                original_prepare_local(compute_layout, out)
                return
            group = adapter.wait_for(compute_layout)
            if group is None:
                submit_h2d(compute_layout)
                swap_api.wait_device_if_needed(name)
            elif id(compute_layout) == group.first_consumer_layout_id:
                adapter.prefetch_successor(group)
            original_prepare_local(compute_layout, out)

            if group is not None:
                adapter.finish_prepare(group, compute_layout)
            elif deferred_d2h_names is None:
                swap_api.execute(name, "D2H")
            else:
                deferred_d2h_names.append(name)

        optimizer._momentum = MethodType(momentum, optimizer)
        if original_preflight_step is not None:
            optimizer._preflight_step = MethodType(preflight_step, optimizer)
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
    swap = getattr(optimizer, "_torchtitan_npu_muon_swap_adapter", None)
    if (
        isinstance(optimizer, DistMuon)
        and not hasattr(optimizer, "_replica_dedup_all_layouts")
        and isinstance(swap, _MuonSwapAdapter)
    ):
        materialize_momentum = optimizer._momentum

        def materialize(layout: Any) -> None:
            # Unlike init_optim_state(), keep only one plan's synthetic
            # gradient live. The adapter retires its NPU state before it
            # moves on to the next plan.
            materialize_momentum(layout, cast("DTensor", torch.zeros_like(layout.param)))

        swap.materialize_all_for_checkpoint(materialize)
        return

    if isinstance(optimizer, DistMuon) and hasattr(optimizer, "_replica_dedup_all_layouts"):
        # Initialize only this replica's owned momentum without a full step.
        for compute_layout in optimizer._parameter_compute_layouts:
            state = optimizer.state[compute_layout.param]
            if "momentum_buffer" not in state:
                grad = cast("DTensor", torch.zeros_like(compute_layout.param))
                optimizer._momentum(compute_layout, grad)
        if isinstance(swap, _MuonSwapAdapter):
            swap.finalize_registration(require_complete=True)
        return
    adamw_swap = getattr(optimizer, "_torchtitan_npu_swap_adapter", None)
    if isinstance(adamw_swap, _NovaSwapAdamW):
        adamw_swap.ensure_all_state()
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
