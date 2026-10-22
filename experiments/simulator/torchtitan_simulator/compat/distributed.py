# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fake collectives and native mesh construction for a logical Meta world."""

import inspect
from contextlib import contextmanager
from datetime import timedelta
from importlib import import_module
from typing import Any, cast

import torch
import torch.distributed as dist

from torchtitan_simulator.compat.device import MetaDeviceModule


def install_meta_fsdp(scope):
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    device_module = MetaDeviceModule()
    scope.replace(torch, "meta", device_module)
    if hasattr(torch.accelerator, "current_stream"):
        original_stream = torch.accelerator.current_stream

        def current_stream(device=None):
            if device is None or (not isinstance(device, int) and torch.device(device).type == "meta"):
                return device_module.current_stream()
            return original_stream(device)

        scope.replace(torch.accelerator, "current_stream", current_stream)
    original_validate = FSDPParamGroup._validate_no_meta_params

    def validate(group):
        if group.device.type == "meta":
            return
        return original_validate(group)

    scope.replace(FSDPParamGroup, "_validate_no_meta_params", validate)
    init = import_module("torch.distributed.fsdp._fully_shard._fsdp_init")
    original_device = init._get_device_from_mesh

    def device_from_mesh(mesh):
        if mesh.device_type == "meta":
            return torch.device("meta")
        return original_device(mesh)

    # The constructor captures this helper by value.
    for name in ("torch.distributed.fsdp._fully_shard._fsdp_init", "torch.distributed.fsdp._fully_shard._fully_shard"):
        owner = import_module(name)
        if getattr(owner, "_get_device_from_mesh", None) is original_device:
            scope.replace(owner, "_get_device_from_mesh", device_from_mesh)
    original_record_stream = torch.Tensor.record_stream

    def record_stream(tensor, stream):
        if tensor.device.type == "meta":
            return
        return original_record_stream(tensor, stream)

    scope.replace(torch.Tensor, "record_stream", record_stream)


@contextmanager
def logical_world(prepared, runtime, *, rank=0):
    from torchtitan.distributed import ParallelDims, utils
    from torchtitan.distributed import spmd_types as spmd_runtime

    if dist.is_initialized():
        raise RuntimeError("Simulator logical world must own its process group")
    clear_native_cache = getattr(torch._C, "_clear_DTensor_sharding_propagator_cache", None)
    if clear_native_cache is not None:
        clear_native_cache()
    previous_backend = utils.get_spmd_backend()
    mesh_tls = getattr(spmd_runtime, "_MESH_TLS", None)
    previous_mesh_context = dict(vars(mesh_tls)) if mesh_tls is not None else None
    dist.init_process_group(
        "fake",
        store=dist.HashStore(),
        rank=rank,
        world_size=prepared.simulation.world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        utils.set_spmd_backend(prepared.trainer.parallelism.spmd_backend)
        install_meta_fsdp(runtime.patches)
        # Cached DTensor output specs retain DeviceMesh objects. Equal logical
        # worlds rebuilt in one process must not reuse the previous world's
        # mesh identity (FSDP validates identity, not just rank equality).
        from torch.distributed.tensor import DTensor

        propagator = DTensor._op_dispatcher.sharding_propagator
        cached = propagator.propagate_op_sharding
        if not hasattr(cached, "cache_clear"):
            raise RuntimeError("DTensor sharding cache isolation capability requires review")
        runtime.patches.replace(
            propagator,
            "propagate_op_sharding",
            type(cached)(propagator.propagate_op_sharding_non_cached),
        )
        dims = ParallelDims.from_config(prepared.trainer.parallelism, prepared.simulation.world_size)
        signature = inspect.signature(dims.build_mesh)
        if "device_type" in signature.parameters:
            cast("Any", dims.build_mesh)(device_type="meta")
        else:
            owner = import_module(dims.build_mesh.__module__)
            if not hasattr(owner, "device_type"):
                raise RuntimeError("Native mesh construction no longer exposes a device selection seam")
            runtime.patches.replace(owner, "device_type", "meta")
            dims.build_mesh()
        yield dims
    finally:
        utils.set_spmd_backend(previous_backend)
        dist.destroy_process_group()
        if clear_native_cache is not None:
            clear_native_cache()
        if previous_mesh_context is not None:
            vars(mesh_tls).clear()
            vars(mesh_tls).update(previous_mesh_context)
