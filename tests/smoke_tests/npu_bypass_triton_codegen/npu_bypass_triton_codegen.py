# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import wraps
import logging
from typing import Any

logger = logging.getLogger(__name__)
_PATCHED = False

_LOWERINGS_TO_KEEP = (
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
    preserved = _collect_lowerings_to_keep()
    _clear_inductor_tables(preserved)
    _fix_torch_npu_inductor_lowering()
    _restore_lowerings(preserved)


def _collect_lowerings_to_keep() -> dict[Any, Any]:
    import torch
    from torch._inductor.lowering import lowerings

    preserved = {}
    for packet_name, overload_name in _LOWERINGS_TO_KEEP:
        packet = getattr(torch.ops.aten, packet_name, None)
        if packet is None:
            continue
        target = getattr(packet, overload_name, None)
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
        from torch_npu._inductor.lowering import _init_set
        from torch_npu._inductor.lowering_op_list import (
            FALLBACK_LIST,
            GENERATE_LIST,
            LOWERING_OVERLOAD_OP,
        )
    except Exception as exc:
        logger.debug("Skip torch_npu lowering cleanup for bypass smoke test: %r", exc)
        return

    _init_set(GENERATE_LIST, set())
    _init_set(LOWERING_OVERLOAD_OP, set())
    FALLBACK_LIST.clear()


def _npu_bypass_backend(gm: Any, example_inputs: Any):
    """Test-only compile backend: clear NPU codegen lowerings, then call Inductor."""
    _prepare_inductor_bypass()
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend("inductor")(gm, example_inputs)


def _wrap_torch_compile(torch_module) -> None:
    original_compile = torch_module.compile

    @wraps(original_compile)
    def compile_with_bypass(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["backend"] = _npu_bypass_backend
        kwargs.pop("options", None)
        return original_compile(*args, **kwargs)

    torch_module.compile = compile_with_bypass
