# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Use vendor shape inference instead of executing opaque metadata kernels."""

import inspect
from functools import wraps

import torch
from torch.overrides import TorchFunctionMode

METADATA_OPERATORS = (
    "lightning_indexer_metadata",
    "sparse_flash_mla_metadata",
    "sparse_flash_mla_grad_metadata",
    "sparse_lightning_indexer_kl_loss_grad_metadata",
)


class _MetadataDeviceMode(TorchFunctionMode):
    """Constrain explicit vendor Meta allocations to the Meta device.

    Some vendor Meta functions themselves request device='npu'. The mode
    exists only while running metadata shape inference, never model compute.
    """

    def __init__(self):
        self.redirected = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        device = kwargs.get("device")
        if device is not None and torch.device(device).type in {"npu", "privateuseone"}:
            self.redirected = True
            kwargs["device"] = torch.device("meta")
        return func(*args, **kwargs)


def _metadata_dispatch(original, operator):
    signature = inspect.signature(original)
    schema_names = {argument.name for argument in operator._schema.arguments}

    @wraps(original)
    def metadata(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        unknown = set(bound.arguments) - schema_names
        if unknown:
            raise RuntimeError(f"Metadata Python/dispatcher contract differs: {original.__name__}: {unknown}")
        values = {
            name: value.to("meta") if isinstance(value, torch.Tensor) else value
            for name, value in bound.arguments.items()
        }
        if not any(isinstance(value, torch.Tensor) for value in values.values()):
            raise ValueError("Metadata shape inference requires explicit tensor metadata")
        return operator(**values)

    return metadata


def install_metadata_dispatch(scope, owner):
    """Replace declared call-site bindings, retaining vendor schemas and shapes."""
    installed = {}
    library = torch.library.Library("cann_ops_transformer", "IMPL", "Meta")
    scope.defer(library._destroy)
    for name in METADATA_OPERATORS:
        original = getattr(owner, name)
        operator = getattr(torch.ops.cann_ops_transformer, name).default
        qualified = f"cann_ops_transformer::{name}"
        if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "Meta"):
            raise RuntimeError(f"Vendor metadata has no Meta implementation: {qualified}")
        native_meta = torch.library.get_kernel(operator, "Meta")
        required = {"num_heads_q": 8, "num_heads_k": 1, "num_heads_kv": 1, "head_dim": 128, "topk": 512}
        probe_arguments: dict[str, object] = {
            argument.name: required[argument.name]
            for argument in operator._schema.arguments
            if argument.name in required and not argument.has_default_value()
        }
        probe_arguments["cu_seqlens_q"] = torch.empty(2, device="meta", dtype=torch.int32)
        probe = _MetadataDeviceMode()
        with torch._C._EnableTorchFunction(), probe:
            native_meta.call_boxed(torch._C.DispatchKeySet(torch._C.DispatchKey.Meta), **probe_arguments)

        def corrected_meta(*args, _native=native_meta, **kwargs):
            # Operator dispatch disables TorchFunction modes on entry. Restore
            # them only inside this defective vendor Meta implementation.
            with torch._C._EnableTorchFunction(), _MetadataDeviceMode():
                return _native.call_boxed(torch._C.DispatchKeySet(torch._C.DispatchKey.Meta), *args, **kwargs)

        if probe.redirected:
            library.impl(name, corrected_meta)
        scope.replace(owner, name, _metadata_dispatch(original, operator))
        installed[name] = str(operator._schema)
    return installed
