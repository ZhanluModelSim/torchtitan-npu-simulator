import importlib
from typing import Any

import torch

from torchtitan_npu.patches.workaround import smla_meta

_ops_module: Any | None = None


def get_cann_transformer_ops() -> Any:
    global _ops_module
    if _ops_module is None:
        try:
            module: Any = importlib.import_module("cann_ops_transformer")
        except ImportError as exc:
            raise RuntimeError(
                "This override requires the cann_ops_transformer package that "
                "matches the installed CANN/OPP version."
            ) from exc
        # Importing the package registers the meta kernels, so the workaround
        # has to override them here rather than at package import time.
        smla_meta.apply()
        # Dispatcher selects Fake/Meta kernels during ``torch.compile`` tracing.
        _ops_module = torch.ops.cann_ops_transformer
    return _ops_module
