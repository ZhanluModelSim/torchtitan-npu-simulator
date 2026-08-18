# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyPTO LI/LIG hooks for the shared DeepSeek-V4 attention bridge.

The metadata, input preparation and autograd bridge are implemented once by
:mod:`.ascendc`.  AscendC SMLA/SMLAG are reused eagerly; the PyPTO LI/LIG modules are
loaded lazily on their first NPU call.
"""

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch

from .ascendc import (
    AscCompressedSparseInnerAttention,
    _SparseAttentionHooks,
)

_LI_MODULE = "torchtitan_npu.ops.pypto.lightning_indexer.lightning_indexer"
_LIG_MODULE = "torchtitan_npu.ops.pypto.sparse_lightning_indexer_kl_loss_grad.sparse_lightning_indexer_kl_loss_grad"


@cache
def _load_pypto_op(module_name: str, op_name: str, device_index: int) -> Callable:
    os.environ.setdefault("TILE_FWK_DEVICE_ID", str(device_index))
    return getattr(importlib.import_module(module_name), op_name)


def _device_index(tensor: torch.Tensor) -> int:
    return 0 if tensor.device.index is None else tensor.device.index


def _pypto_lightning_indexer(q: torch.Tensor, *args, **kwargs):
    op = _load_pypto_op(_LI_MODULE, "lightning_indexer", _device_index(q))
    return op(q, *args, **kwargs)


def _pypto_sparse_lightning_indexer_kl_loss_grad(*, q: torch.Tensor, **kwargs):
    op = _load_pypto_op(
        _LIG_MODULE,
        "sparse_lightning_indexer_kl_loss_grad",
        _device_index(q),
    )
    return op(q=q, **kwargs)


def _pypto_sparse_flash_mla_grad(*args, **kwargs):
    """Use AscendC SMLAG while preserving the deterministic workaround."""

    deterministic_mode = torch.get_deterministic_debug_mode()
    if deterministic_mode:
        torch.set_deterministic_debug_mode(0)
    try:
        return torch.ops.cann_ops_transformer.sparse_flash_mla_grad(*args, **kwargs)
    finally:
        if deterministic_mode:
            torch.set_deterministic_debug_mode(deterministic_mode)


_PYPTO_SPARSEATTN_HOOK = _SparseAttentionHooks(
    lightning_indexer=_pypto_lightning_indexer,
    sparse_flash_mla=torch.ops.cann_ops_transformer.sparse_flash_mla,
    sparse_flash_mla_grad=_pypto_sparse_flash_mla_grad,
    sparse_lightning_indexer_kl_loss_grad=_pypto_sparse_lightning_indexer_kl_loss_grad,
)


class PyPTOCompressedSparseInnerAttention(AscCompressedSparseInnerAttention):
    """Use PyPTO LI/LIG with AscendC metadata and SMLA/SMLAG."""

    @dataclass(kw_only=True, slots=True)
    class Config(AscCompressedSparseInnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.hooks = _PYPTO_SPARSEATTN_HOOK
