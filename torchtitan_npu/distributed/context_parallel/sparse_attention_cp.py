# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V3.2 Context Parallel — ParallelStyle with pre-hook only.

Pre-hook:  all-gather K, V, and k_indexer across CP ranks using
           ``ft_c.all_gather_tensor_autograd``, causal-slice each.

No post-hook is needed — the output is rank-local (Q attends to global KV).

Assumes the NPU converter (``npu_dsa``) has already replaced the module's
forward with fused ``torch_npu`` ops.
"""

from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives as ft_c
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module

from torchtitan_npu.models.deepseek_v32.model import DSV32_SDPA

from .registry import register_cp_strategy


class SparseAttentionCP(ParallelStyle):
    """Context Parallel for DSA (DeepSeek Sparse Attention).

    All-gathers K, V, and k_indexer on the CP mesh before the
    (converter-replaced) forward, and causal-slices each to
    ``(rank + 1) × s_local`` so each rank's Q only attends to
    historical KV tokens.

    The module's forward must already be replaced by the NPU converter
    (``npu_dsa``), which provides ``npu_lightning_indexer`` and
    ``npu_sparse_flash_attention`` fused ops.
    """

    @staticmethod
    def _pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        q, k, v = args[0], args[1], args[2]

        # q/k/v are BNSD from PreAttention: [B, nheads, seq, dim]
        s_local = k.shape[2]
        rank = mesh.get_local_rank()
        group = mesh.get_group()
        slice_end = (rank + 1) * s_local

        def _ag(x: torch.Tensor, dim: int) -> torch.Tensor:
            r = ft_c.all_gather_tensor_autograd(x.contiguous(), gather_dim=dim, group=group)
            if isinstance(r, ft_c.AsyncCollectiveTensor):
                # See CompressorAttentionCP for reason
                # not using AsyncCollectiveTensor.wait().
                r = torch.ops._c10d_functional.wait_tensor(r)
            return r

        k = _ag(k, dim=2)[:, :, :slice_end, :]
        v = _ag(v, dim=2)[:, :, :slice_end, :]

        new_kwargs = dict(kwargs)
        k_indexer = new_kwargs.get("k_indexer")
        if k_indexer is not None and isinstance(k_indexer, torch.Tensor):
            idx_k = _ag(k_indexer, dim=1)[:, :slice_end, :, :]
            # Refresh 0-axis stride of the sliced tensor, otherwise LI may reject this.
            new_kwargs["k_indexer"] = idx_k.reshape(-1).view_as(idx_k)

        new_args = (q, k, v, *args[3:])
        return new_args, new_kwargs

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        module.register_forward_pre_hook(partial(self._pre_hook, mesh=device_mesh), with_kwargs=True)
        return module


def _detect_dsv32(module: torch.nn.Module) -> bool:
    return isinstance(module, DSV32_SDPA)


def _apply_dsv32(module: torch.nn.Module, cp_mesh: DeviceMesh) -> None:
    parallelize_module(module, cp_mesh, SparseAttentionCP())


register_cp_strategy(_detect_dsv32, _apply_dsv32)
