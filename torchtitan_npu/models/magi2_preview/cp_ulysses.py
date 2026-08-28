# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses context parallelism for MAGI-2-preview.

Fork reason: Upstream torchtitan has no MAGI-2 support. The packed-token
model sequence-shards in ORIGINAL token order across the CP mesh (chunk T
by the CP degree), and every attention sublayer swaps sequence for heads
with Ulysses all-to-all collectives so the attention core runs on the full
sequence with a head subset per rank. This mirrors the official inference
EP dispatch/undispatch head-swap layout
(/tmp/magi2-preview/inference/model/magi2_preview.py, ``ep_dispatch`` /
``ep_undispatch``, ~lines 3277-3347), applied here to attention with
funcol all-to-all primitives in the same spirit as the kimi_k3
parallelize.py funcol precedent.

Layout contract (cp = CP degree, S = T // cp local tokens, H heads):
- pre-hook  (on ``Magi2Attention.attn_core``): q/k/v ``(S, H, D)`` ->
  all_to_all -> ``(cp*S, H/cp, D)`` full sequence, local head group; the
  RoPE embed ``(S, rotary_dim)`` is all-gathered to the full sequence so
  RoPE is applied AFTER the swap.
- post-hook: attention output ``(cp*S, H/cp, D)`` -> all_to_all back ->
  ``(S, H, D)``, original local order, full heads.
- ``sinks`` parameter: sharded on the HEAD dim (each rank keeps the
  logits of its local head group); state-dict key unchanged.

Model-level entry/exit lives in model.py: forward slices the incoming
FULL batch tensors to the local sequence shard and all-gathers the
``(T_local, 64)`` prediction back to the full sequence with autograd
(``funcol.all_gather_tensor_autograd``) so the trainer's full-label MSE
loss runs unchanged. v1 loss decision: every CP rank computes the same
full-sequence loss and the all-gather backward is a reduce-scatter with
sum, so every rank-local gradient upstream of the exit gather would equal
``cp_degree`` times the shard's CP=1 contribution; the model exit applies
a gradient-only ``1 / cp_degree`` compensation
(``model._CpGradCompensation``), restoring CP=1 gradient scale with no
trainer/loss changes. The emulated and nightly CP=2-vs-CP=1 tests pin
these semantics.

Integration status:
- ``parallelize_magi2_preview`` calls ``apply_magi2_ulysses_cp`` when
  ``parallel_dims.cp_enabled`` (before MoE/AC/FSDP), passing the 1D "cp"
  mesh and the configured EP degree; torchtitan's "fsdp" mesh spans
  dp_shard x cp, so CP-replicated parameter gradients reduce during FSDP.
- Sequence divisibility (seq_len % cp_degree == 0) is asserted per batch
  in ``Magi2PreviewModel.forward``; the loaders emit full sequences on
  every rank and the model slices at entry (no loader changes needed).
- CP + EP combination (head mesh cp x ep, head-parallel MoE regime (b))
  is a later integration: ``apply_magi2_ulysses_cp`` raises
  NotImplementedError when both are requested.
- Checkpointing with sharded ``sinks``: keys are unchanged but shapes are
  head-sharded per rank; DTensor Shard(1)-aware save/restore is a
  follow-up.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module

from .attention import Magi2Attention

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CpContext:
    """Per-rank Ulysses CP state installed by the parallelize wiring.

    Args:
        mesh: 1D "cp" DeviceMesh. Only ``size()``/``get_local_rank()``/
            ``get_group()`` are used, which keeps the collective helpers
            testable with lightweight stand-ins.
        degree: CP world size.
        rank: this rank's coordinate in the CP mesh.
    """

    mesh: DeviceMesh
    degree: int
    rank: int


def _wait_if_async(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize funcol AsyncCollectiveTensor results before reshaping."""
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        return torch.ops._c10d_functional.wait_tensor(tensor)
    return tensor


def _dispatch_send_layout(x: torch.Tensor, cp_degree: int) -> torch.Tensor:
    """All-to-all send layout for the seq->head swap (official ep_dispatch).

    Args:
        x: (S, H, D) local-sequence, full-heads tensor.
        cp_degree: CP world size; must divide H.

    Returns:
        (cp_degree, S, H/cp, D): chunk ``i`` is the head group destined
        for CP rank ``i``.
    """
    S, H, D = x.shape
    return (
        x.view(S, cp_degree, H // cp_degree, D)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def _undispatch_send_layout(x: torch.Tensor, cp_degree: int) -> torch.Tensor:
    """All-to-all send layout for the head->seq swap (official ep_undispatch).

    Args:
        x: (cp_degree * S, H/cp, D) full-sequence, local-heads tensor.
        cp_degree: CP world size.

    Returns:
        (cp_degree, S, H/cp, D): chunk ``i`` is the token shard destined
        for CP rank ``i`` (rows ``[i*S, (i+1)*S)`` of the full sequence).
    """
    T, Hc, D = x.shape
    return x.view(cp_degree, T // cp_degree, Hc, D).contiguous()


def ulysses_dispatch(x: torch.Tensor, *, mesh: DeviceMesh) -> torch.Tensor:
    """Swap sequence for heads: (S, H, D) -> (cp*S, H/cp, D).

    Each rank sends one head group to every peer and receives every rank's
    local token shard for its own head group; received chunks arrive in
    rank order, and CP shards are contiguous slices of the original token
    order, so the result is the FULL sequence in original order.
    """
    cp = mesh.size()
    send = _dispatch_send_layout(x, cp)
    recv = funcol.all_to_all_single_autograd(
        send, None, None, mesh.get_group()
    )
    recv = _wait_if_async(recv)
    return recv.view(cp * x.shape[0], x.shape[1] // cp, x.shape[2])


def ulysses_undispatch(x: torch.Tensor, *, mesh: DeviceMesh) -> torch.Tensor:
    """Swap heads back for sequence: (cp*S, H/cp, D) -> (S, H, D).

    Inverse of ``ulysses_dispatch``: each rank returns every peer's token
    shard and concatenates the received head groups.
    """
    cp = mesh.size()
    T, Hc, D = x.shape
    send = _undispatch_send_layout(x, cp)
    recv = funcol.all_to_all_single_autograd(
        send, None, None, mesh.get_group()
    )
    recv = _wait_if_async(recv)
    return recv.view(cp, T // cp, Hc, D).permute(1, 0, 2, 3).reshape(
        T // cp, cp * Hc, D
    )


def gather_seq(x: torch.Tensor, *, mesh: DeviceMesh) -> torch.Tensor:
    """All-gather a sequence-sharded (S, ...) tensor to (cp*S, ...) on dim 0.

    Autograd-aware: the backward is a reduce-scatter with sum (torch
    funcol semantics); see the module docstring v1 loss decision.
    """
    gathered = funcol.all_gather_tensor_autograd(
        x.contiguous(), gather_dim=0, group=mesh.get_group()
    )
    return _wait_if_async(gathered)


def cp_pre_attention_swap(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rope: torch.Tensor,
    cp_context: CpContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-attention Ulysses swap: head-split q/k/v + full-sequence RoPE."""
    mesh = cp_context.mesh
    q = ulysses_dispatch(q, mesh=mesh)
    k = ulysses_dispatch(k, mesh=mesh)
    v = ulysses_dispatch(v, mesh=mesh)
    rope = gather_seq(rope, mesh=mesh)
    return q, k, v, rope


def cp_post_attention_swap(out: torch.Tensor, cp_context: CpContext) -> torch.Tensor:
    """Post-attention Ulysses swap back to the local sequence shard."""
    return ulysses_undispatch(out, mesh=cp_context.mesh)


class Magi2UlyssesAttentionCP(ParallelStyle):
    """Ulysses CP for one ``Magi2Attention`` module.

    Applied via ``parallelize_module`` (or ``apply_magi2_ulysses_cp``):
    head-shards the ``sinks`` parameter, stores ``module.cp_context`` and
    registers the all-to-all/all-gather hooks on the parameter-free
    ``module.attn_core`` submodule, so the swaps happen inside the
    attention path around the RoPE + attention core without touching the
    grouped projections or state-dict keys.
    """

    @staticmethod
    def _pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        ctx: CpContext,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        del module
        q, k, v, rope, cu_seqlens = args
        q, k, v, rope = cp_pre_attention_swap(q, k, v, rope, ctx)
        return (q, k, v, rope, cu_seqlens), kwargs

    @staticmethod
    def _post_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        output: torch.Tensor,
        ctx: CpContext,
    ) -> torch.Tensor:
        del module, args
        return cp_post_attention_swap(output, ctx)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, Magi2Attention):
            raise TypeError(
                "Magi2UlyssesAttentionCP expects Magi2Attention, got "
                f"{type(module).__name__}"
            )
        if getattr(device_mesh, "ndim", 1) != 1:
            raise ValueError(
                f"MAGI-2 Ulysses CP expects a 1D cp mesh, got "
                f"{device_mesh.ndim}D"
            )
        cp = device_mesh.size()
        if module.num_heads % cp != 0:
            raise ValueError(
                f"num_heads ({module.num_heads}) must be divisible by the "
                f"CP degree ({cp})"
            )
        if getattr(module, "cp_context", None) is not None:
            raise ValueError(
                "Magi2UlyssesAttentionCP applied twice on the same module"
            )

        # Head-shard the learned sink logits; the state-dict key stays
        # "attention.sinks", only its head dim shrinks to the local group.
        rank = device_mesh.get_local_rank()
        heads_per_rank = module.num_heads // cp
        with torch.no_grad():
            shard = (
                module.sinks[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
                .contiguous()
                .clone()
            )
        module.sinks = nn.Parameter(shard)

        ctx = CpContext(mesh=device_mesh, degree=cp, rank=rank)
        module.cp_context = ctx
        module.attn_core.register_forward_pre_hook(
            partial(self._pre_hook, ctx=ctx), with_kwargs=True
        )
        module.attn_core.register_forward_hook(partial(self._post_hook, ctx=ctx))
        logger.info(
            "Applied MAGI-2-preview Ulysses CP (degree=%d, rank=%d) to %d heads",
            cp,
            rank,
            module.num_heads,
        )
        return module


def apply_magi2_ulysses_cp(
    model: nn.Module, *, cp_mesh: DeviceMesh, ep_degree: int = 1
) -> nn.Module:
    """Model-level Ulysses CP wiring used by the parallelize function.

    Sets the ``Magi2PreviewModel`` CP attributes (``cp_context`` /
    ``cp_degree`` / ``cp_rank``) and applies ``Magi2UlyssesAttentionCP``
    to every layer's attention submodule.

    Args:
        model: a ``Magi2PreviewModel``.
        cp_mesh: 1D "cp" DeviceMesh.
        ep_degree: expert parallel degree requested alongside CP; the
            CP+EP combination (head mesh cp x ep) is a later integration.

    Returns:
        The same model, CP-enabled (in-place).
    """
    degree = cp_mesh.size()
    if degree > 1 and ep_degree > 1:
        raise NotImplementedError(
            "MAGI-2-preview does not support CP and EP together yet; the "
            "combined cp x ep head mesh (Item-4 regime b) is a later "
            "integration"
        )
    num_heads = model.config.hidden_size // model.config.head_dim
    if num_heads % degree != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by the CP degree "
            f"({degree})"
        )

    rank = cp_mesh.get_local_rank()
    ctx = CpContext(mesh=cp_mesh, degree=degree, rank=rank)
    model.cp_context = ctx
    model.cp_degree = degree
    model.cp_rank = rank

    styles = {
        f"block.layers.{layer_id}.attention": Magi2UlyssesAttentionCP()
        for layer_id in range(len(model.block.layers))
    }
    parallelize_module(model, cp_mesh, styles)
    return model
