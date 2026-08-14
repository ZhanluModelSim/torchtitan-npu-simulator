# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)
_PATCHED = False
_ASCENDC_LOWERING_GUARD = "_LoweringGuard"
_ASCENDC_LOWERING_REGISTRY = "_data"
_INDUCTOR_LOWERING_MODULE = "torch._inductor.lowering"
_TORCH_NPU_ASCENDC_LOWERING_MODULE = "torch_npu._inductor.ascendc.lowering.common"
_TORCH_NPU_DYNAMO_MODULE = "torch_npu.utils._dynamo"
_TORCH_NPU_LOWERING_OP_LIST_MODULE = "torch_npu._inductor.lowering_op_list"
_NPU_BACKEND_SCOPE = "_NpuBackendScope"
_GENERATE_LIST = "GENERATE_LIST"
_LOWERING_OVERLOAD_OP = "LOWERING_OVERLOAD_OP"
_FALLBACK_LIST = "FALLBACK_LIST"

_LOWERINGS_TO_KEEP = (
    ("copy_", "default"),
    ("view", "default"),
    ("reshape", "default"),
    ("view_as_complex", "default"),
    ("split_with_sizes", "default"),
)


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import torch

    _enable_implicit_fallbacks()
    _wrap_torch_compile(torch)
    _PATCHED = True


def _enable_implicit_fallbacks() -> None:
    from torch._inductor import config

    config.implicit_fallbacks = True


def _prepare_inductor_bypass() -> None:
    # Importing torch_npu._inductor can lazily register AscendC lowerings.
    _fix_torch_npu_inductor_lowering()
    preserved = _collect_lowerings_to_keep()
    _clear_inductor_tables(preserved)
    _restore_lowerings(preserved)


def _collect_lowering_targets_to_keep() -> tuple[Any, ...]:
    import torch

    targets = []
    for packet_name, overload_name in _LOWERINGS_TO_KEEP:
        packet = getattr(torch.ops.aten, packet_name, None)
        if packet is None:
            continue
        target = getattr(packet, overload_name, None)
        if target is not None:
            targets.append(target)
    return tuple(targets)


def _collect_lowerings_to_keep() -> dict[Any, Any]:
    lowerings = importlib.import_module(_INDUCTOR_LOWERING_MODULE).lowerings

    preserved = {}
    for target in _collect_lowering_targets_to_keep():
        if target in lowerings:
            preserved[target] = lowerings[target]
    return preserved


def _clear_inductor_tables(preserved: dict[Any, Any]) -> None:
    from torch._inductor.decomposition import decompositions
    from torch._inductor.lowering import lowerings

    lowerings.clear()
    lowerings.update(preserved)
    decompositions.clear()


def _restore_lowerings(preserved: dict[Any, Any]) -> None:
    from torch._inductor.lowering import lowerings

    lowerings.update(preserved)


def _fix_torch_npu_inductor_lowering() -> None:
    try:
        lowering_op_list = importlib.import_module(_TORCH_NPU_LOWERING_OP_LIST_MODULE)
    except Exception as exc:
        logger.debug("Skip legacy torch_npu lowering cleanup for bypass smoke test: %r", exc)
    else:
        getattr(lowering_op_list, _GENERATE_LIST).clear()
        getattr(lowering_op_list, _LOWERING_OVERLOAD_OP).clear()
        getattr(lowering_op_list, _FALLBACK_LIST).clear()

    try:
        ascendc_lowering = importlib.import_module(_TORCH_NPU_ASCENDC_LOWERING_MODULE)
    except Exception as exc:
        logger.debug("Skip AscendC lowering cleanup for bypass smoke test: %r", exc)
    else:
        lowering_guard = getattr(
            ascendc_lowering,
            _ASCENDC_LOWERING_GUARD,
        )
        lowering_registry = getattr(
            lowering_guard,
            _ASCENDC_LOWERING_REGISTRY,
        )
        preserved_registry = {
            target: lowering_registry[target]
            for target in _collect_lowering_targets_to_keep()
            if target in lowering_registry
        }
        lowering_registry.clear()
        lowering_registry.update(preserved_registry)


def _npu_bypass_backend(gm: Any, example_inputs: Any):
    """Test-only compile backend: clear NPU codegen lowerings, then call Inductor."""
    from torch._dynamo.backends.registry import lookup_backend

    inductor_backend = lookup_backend("inductor")
    torch_npu_dynamo = importlib.import_module(_TORCH_NPU_DYNAMO_MODULE)
    npu_backend_scope = getattr(torch_npu_dynamo, _NPU_BACKEND_SCOPE)
    with npu_backend_scope("ascendc"):
        _prepare_inductor_bypass()
        return inductor_backend(gm, example_inputs)


def _wrap_torch_compile(torch_module) -> None:
    original_compile = torch_module.compile

    @wraps(original_compile)
    def compile_with_bypass(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["backend"] = _npu_bypass_backend
        kwargs.pop("options", None)
        return original_compile(*args, **kwargs)

    torch_module.compile = compile_with_bypass
