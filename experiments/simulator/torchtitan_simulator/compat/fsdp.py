# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Check the private FSDP call boundaries used by capture, without installing patches."""

import inspect
import json
import sys
from importlib import import_module, metadata
from importlib.util import find_spec


def software_stack() -> dict[str, object]:
    """Describe installed versions and actual/resolvable module origins, without importing NPU code."""
    packages = {}
    for distribution, module_name in (
        ("torch", "torch"),
        ("torch-npu", "torch_npu"),
        ("torchtitan", "torchtitan"),
        ("torchtitan-npu", "torchtitan_npu"),
        ("torchtitan-simulator", "torchtitan_simulator"),
    ):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = None
        module = sys.modules.get(module_name)
        spec = find_spec(module_name) if module is None else None
        packages[distribution] = {
            "version": version,
            "path": getattr(module, "__file__", None) if module is not None else getattr(spec, "origin", None),
            "loaded": module is not None,
        }
    torch = sys.modules.get("torch")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
    }


def validate_fsdp_capture() -> dict[str, object]:
    """Reject missing/incompatible call seams; this is not a behavioral or version certification.

    Bind only calls made by the adapters. Defaults, annotations and additional
    optional parameters do not determine compatibility. Never invoke FSDP hooks
    here: model state, collective behavior and residency need capture regressions.
    """
    failures = []
    signatures = {}
    root = "torch.distributed.fsdp._fully_shard"
    # Counts include self for instance methods. Forward/backward observers
    # forward arbitrary arguments; these shapes are the native hook contracts.
    checks = (
        ("_fsdp_param_group", "FSDPParamGroup", "unshard", 2),
        ("_fsdp_param_group", "FSDPParamGroup", "wait_for_unshard", 1),
        ("_fsdp_param_group", "FSDPParamGroup", "reshard", 1),
        ("_fsdp_param_group", "FSDPParamGroup", "finalize_backward", 1),
        ("_fsdp_param_group", "FSDPParamGroup", "post_backward", 1),
        ("_fsdp_param_group", "FSDPParamGroup", "_prefetch_unshard", 2),
        ("_fsdp_param_group", "FSDPParamGroup", "_validate_no_meta_params", 1),
        ("_fsdp_state", "FSDPState", "_pre_forward", 4),
        ("_fsdp_state", "FSDPState", "_pre_backward", 2),
        ("_fsdp_state", "FSDPState", "_post_forward", 4),
        ("_fsdp_param", "FSDPParam", "_init_sharded_param", 5),
        ("_fsdp_param", "FSDPParam", "_init_sharded_post_forward_param_metadata", 2),
        ("_fsdp_init", None, "_get_device_from_mesh", 1),
    )
    for module_name, class_name, name, count in checks:
        symbol = ".".join(part for part in (root, module_name, class_name, name) if part)
        try:
            owner = import_module(f"{root}.{module_name}")
            if class_name is not None:
                owner = getattr(owner, class_name)
            method = getattr(owner, name)
            if name == "_prefetch_unshard" and not isinstance(inspect.getattr_static(owner, name), staticmethod):
                raise TypeError("capture requires the native staticmethod descriptor")
            signature = inspect.signature(method)
            signature.bind(*(object() for _ in range(count)))
            signatures[symbol] = str(signature)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
    # These are accessed directly by the residency observer, not optional fallbacks.
    for module_name, class_name, name in (
        ("_fsdp_param_group", "FSDPParamGroup", "is_unsharded"),
        ("_fsdp_param_group", "FSDPCommContext", "__setattr__"),
        ("_fsdp_common", None, "FSDPMeshInfo"),
    ):
        symbol = ".".join(part for part in (root, module_name, class_name, name) if part)
        try:
            owner = import_module(f"{root}.{module_name}")
            if class_name is not None:
                owner = getattr(owner, class_name)
            getattr(owner, name)
        except (ImportError, AttributeError) as exc:
            failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
    stack = software_stack()
    if failures:
        raise RuntimeError(
            "FSDP capture compatibility check failed before simulator patches were installed:\n"
            + "\n".join(failures)
            + "\nReview the native FSDP adapter against this software stack: "
            + json.dumps(stack, sort_keys=True)
        )
    return {"check": "fsdp-call-boundaries-v1", "signatures": signatures, "software": stack}


if __name__ == "__main__":
    print(json.dumps(validate_fsdp_capture(), indent=2, sort_keys=True))
