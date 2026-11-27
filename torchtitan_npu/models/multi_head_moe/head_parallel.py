# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-parallel execution variant for the Multi-Head MoE layer.

MAGI-2 preview's Head Parallel (``Multi-Head LatentMoE and Head Parallel``,
sand.ai/blog/magi-2-preview §2.1/3.1) statically splits the projected token
representation along the MoE head axis: every rank projects all heads for its
own tokens, then two fixed-shape all-to-alls transpose ownership from tokens
to heads. Each rank routes and computes its owned heads for all tokens in the
ep group locally, sends the results back with the reverse all-to-all, and
finishes ``proj_out`` for its own tokens. Communication never grows with
top-k, unlike the token-choice all-to-all used by the default expert-parallel
path.

Layout facts this relies on (verified against torchtitan 0.3.0
``parallel_dims.py``): the ep group spans the data dimension — each rank of
an ep group holds a different ``dp_shard`` token shard, which is exactly the
token-distributed input the official head dispatch requires. The expert
weights keep the default expert-parallel slicing (``Shard(0)`` on the expert
axis, ``E_total/ep`` experts per rank); ``proj_in``/``proj_out`` weights and
the ``router`` stay ep-replicated — ``sharding.py`` wires the same replicated
placements for both flavors and has no ``head_parallel`` branch. The router
therefore deviates from the official head-parallel design (``Shard(0)`` on
the ep axis): this port declares it replicated (FSDP requires R on the
data-parallel storage axes, and the ep axis overlaps ``dp_shard`` in the
torchtitan mesh) and slices the owned-head range by ep rank at runtime inside
``forward`` — a deliberate **R + runtime slice** arrangement, not a
``Shard(0)`` placement.

The class reuses the layer's built submodules (``proj_in`` / ``router`` /
``routed_experts`` / ``proj_out`` / ``shared_experts``); only the forward
data path changes. The collectives mirror ``AllToAllTokenDispatcher``:
``functional_collectives.all_to_all_single`` with the ep ``DeviceMesh`` as
group — the only collective backend this variant uses (no ``spmd_types``
``all_to_all`` fallback).

Limitations: every rank of the ep group must hold the same token count for
the step. The static all-to-alls size every transfer block from the local
shard, so uneven token shards break the collectives themselves (mismatched
sizes error or hang long before the reassembly ``view`` could check) — this
is a caller precondition the forward cannot enforce locally, so every real
forward validates the ep-group token counts with one small all-gather per
MoE layer before the dispatch and raises on disagreement. The validation is
deliberately uncached: a skip keyed on the local token count alone lets
ranks take different collectives when only some ranks change shape (one
rank ``all_gather``s while another proceeds to the all-to-all, skewing the
group), so all ranks participate at the same point on every call. Meta
capture has no live peers (its transfer shapes are config-derived) and is
skipped. ``moe_heads`` must be divisible by ``expert_parallel_degree``
(validated in ``update_from_config``). The data all-to-alls stay
fixed-shape and their bytes do not grow with top-k.

Opt-in only: ``MultiHeadMoEModel.Config.head_parallel`` (set by the
``debugmodel-head-parallel`` flavor) swaps this variant in during
``update_from_config``; the default flavor keeps the token-dispatch EP path
byte-for-byte.
"""

import dataclasses

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_to_all_single
from torch.distributed.tensor import DeviceMesh, DTensor

from .model import MultiHeadMoE


def static_balanced_expert_ids(
    tokens_all: int, head_local: int, k_per_head: int, experts_per_head: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic load-balanced routing ids for the debug HP flavor.

    ``forward`` and the simulator's capture facts must agree on the exact
    pattern, so both call this helper (forward on the live device, capture
    on CPU) instead of restating the arithmetic. Returns the in-head expert
    index and the local-pool expert id per (token, head, slot).
    """
    indices_TSK = (
        torch.arange(tokens_all * head_local * k_per_head, device=device, dtype=torch.int64).reshape(
            tokens_all, head_local, k_per_head
        )
        % experts_per_head
    )
    head_offset = torch.arange(head_local, device=device, dtype=torch.int64) * experts_per_head
    return indices_TSK, indices_TSK + head_offset.view(1, -1, 1)


def head_dispatch(x_HTS: torch.Tensor, ep_mesh: DeviceMesh, head_local: int) -> torch.Tensor:
    """Transpose ``[Hm, T_local, hs]`` (token-owned, all heads) to
    ``[T_all, head_local, hs]`` (head-owned, all tokens in the ep group).

    Sends head block ``j`` to rank ``j`` and concatenates the received
    per-rank token blocks in rank order along the token axis.
    """
    ep = ep_mesh.size()
    splits = [head_local] * ep
    out = all_to_all_single(x_HTS, splits, splits, group=ep_mesh)
    t_local, hs = x_HTS.shape[1], x_HTS.shape[2]
    return out.view(ep, head_local, t_local, hs).permute(0, 2, 1, 3).reshape(-1, head_local, hs)


def head_combine(y_TSH: torch.Tensor, ep_mesh: DeviceMesh, t_local: int) -> torch.Tensor:
    """Transpose ``[T_all, head_local, hs]`` (head-owned) back to
    ``[T_local, Hm, hs]`` (token-owned), concatenating the received head
    blocks in rank order along the head axis.
    """
    ep = ep_mesh.size()
    splits = [t_local] * ep
    out = all_to_all_single(y_TSH, splits, splits, group=ep_mesh)
    h_local, hs = y_TSH.shape[1], y_TSH.shape[2]
    return out.view(ep, t_local, h_local, hs).permute(1, 0, 2, 3).reshape(t_local, -1, hs)


def _validate_ep_token_shards(ep_mesh: DeviceMesh, tokens: int, device: torch.device) -> None:
    """Reliable fail-fast for the even-token-shard precondition (PR #23 review).

    The static all-to-alls size each transfer block from the local shard, so
    uneven ``T_local`` across the ep group breaks the collectives themselves
    (mismatched sizes error or hang) before ``head_dispatch``'s reassembly
    ``view`` could check anything. Every real forward therefore gathers the
    per-rank token counts over the ep group and raises when they disagree.

    The validation deliberately runs on every call: caching keyed on the
    local token count lets ranks disagree about whether to participate when
    only some ranks change shape (one rank enters the ``all_gather`` while
    another proceeds to the data all-to-all, skewing the group into a hang),
    so all ranks take the same collective at the same point, every time. The
    payload is one int64 per rank. Meta capture has no live peers to ask
    (its transfer shapes are config-derived) and is skipped.
    """
    if device.type == "meta":
        return
    group = ep_mesh.get_group()
    assert group is not None
    counts = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(ep_mesh.size())]
    dist.all_gather(counts, torch.tensor([tokens], dtype=torch.int64, device=device), group=group)
    peer_counts = sorted({int(count.item()) for count in counts})
    if len(peer_counts) > 1:
        raise ValueError(
            f"head parallel requires equal token shards across the ep group: this rank holds "
            f"{tokens} tokens while the group holds {peer_counts}; the static all-to-alls "
            f"cannot transfer uneven shards (they error or hang on mismatched sizes), so "
            f"shard the tokens evenly across the ep group (e.g. drop_last)"
        )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _local_or_none(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """None-safe ``_local`` for the optional ``Linear`` bias."""
    return None if tensor is None else _local(tensor)


class HeadParallelMultiHeadMoE(MultiHeadMoE):
    """Head-parallel rewrite of :class:`MultiHeadMoE.forward`.

    The reference geometry is unchanged; owned-head ids come from the
    replicated router parameter sliced to the ep rank's head range at runtime
    (the R + runtime slice arrangement described in the module docstring), so
    the id formula is shared with EP mode and only the local sizes differ.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(MultiHeadMoE.Config):
        pass

    def _ep_mesh(self) -> DeviceMesh | None:
        """ep mesh for the head-ownership collectives.

        The routed-experts dispatcher has the ep mesh wired at parallelize
        time (the same object its all-to-alls use), which also exists under
        the simulator's meta capture where module parameters arrive as plain
        local tensors. Parameter DTensors are the fallback for unbuilt
        contexts.
        """
        if getattr(self, "_ep_mesh_override", None) is not None:
            override = self._ep_mesh_override
            assert isinstance(override, DeviceMesh)
            return override
        try:
            from torchtitan.distributed.spmd_types import spmd_sparse_mesh

            sparse = spmd_sparse_mesh()
            names = sparse.mesh_dim_names if sparse is not None else None
            if names and "ep" in names:
                return sparse["ep"]  # type: ignore[index]
        except Exception:
            pass
        dispatcher = getattr(self.routed_experts, "token_dispatcher", None)
        mesh = getattr(dispatcher, "ep_mesh", None)
        if isinstance(mesh, DeviceMesh):
            return mesh
        for _, parameter in self.named_parameters():
            if isinstance(parameter, DTensor):
                names = parameter.device_mesh.mesh_dim_names
                if names and "ep" in names:
                    return parameter.device_mesh["ep"]
        return None

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        B, L, D = x_BLD.shape
        tokens = B * L

        # Shared expert consumes the original token-owned activation; the
        # routed path transposes ownership away and back.
        shared_out = self.shared_experts(x_BLD)
        shared_out = _local(shared_out).reshape(tokens, D)

        x_TD = _local(x_BLD).reshape(tokens, D)
        w_in = _local(self.proj_in.weight)
        heads = w_in.shape[0] // self.head_hidden
        # _local_or_none passes the optional Linear bias through untouched, so
        # bias=True configurations stay equivalent to the default module call.
        x_TSH = F.linear(x_TD, w_in, _local_or_none(self.proj_in.bias)).view(tokens, heads, self.head_hidden)

        # The gate router stays a replicated parameter (FSDP requires R on
        # the data-parallel storage axes, and the ep axis overlaps dp_shard
        # in the torchtitan mesh); head ownership is a runtime slice. The
        # parameter is tiny next to the expert pool, so the memory cost of
        # replication is negligible and the communication facts (two static
        # all-to-alls per MoE layer) are unchanged.
        ep_mesh = self._ep_mesh()
        router = _local(self.router.router)
        if ep_mesh is None or ep_mesh.size() == 1:
            # Single-rank: no ownership transfer, route every head locally.
            head_local = heads
            tokens_all, head_local_ = x_TSH.shape[0], x_TSH.shape[1]
        else:
            ep = ep_mesh.size()
            assert ep_mesh is not None
            # The collectives below cannot recover from uneven token shards;
            # validate the ep-group counts before the first all-to-all on
            # every forward (uncached — a skip keyed on the local token count
            # skews the group when only some ranks change shape, PR #23
            # review).
            _validate_ep_token_shards(ep_mesh, tokens, x_TSH.device)
            head_local = heads // ep
            rank = ep_mesh.get_local_rank()
            router = router[rank * head_local : (rank + 1) * head_local]
            x_TSH = head_dispatch(x_TSH.permute(1, 0, 2).contiguous(), ep_mesh, head_local)
            tokens_all, head_local_ = x_TSH.shape[0], x_TSH.shape[1]

        experts_per_head = self.router.experts_per_head
        k_per_head = self.router.k_per_head

        logits = torch.einsum("tsh,she->tse", x_TSH, router)
        scores = F.softmax(logits, dim=-1)
        if self.router._debug_force_load_balance:
            indices_TSK, expert_ids_TSK = static_balanced_expert_ids(
                tokens_all, head_local_, k_per_head, experts_per_head, scores.device
            )
            weights_TSK = scores.gather(dim=-1, index=indices_TSK)
        else:
            weights_TSK, indices_TSK = torch.topk(scores, k=k_per_head, dim=-1)
            # Local expert id space: the grouped experts see their local
            # weight shard (E_local = E_total/ep, matching
            # ``w1_EFD.to_local()``), so the grouped-mm offsets span the
            # local pool only. Owned head s_local maps to local expert
            # s_local*Es + idx.
            head_offset = torch.arange(head_local, device=x_TSH.device, dtype=torch.int64) * experts_per_head
            expert_ids_TSK = indices_TSK + head_offset.view(1, -1, 1)

        routing_map_TSE = torch.zeros(
            tokens_all,
            head_local,
            experts_per_head * head_local,
            dtype=torch.bool,
            device=x_TSH.device,
        ).scatter_(-1, expert_ids_TSK, True)
        num_tokens_per_expert_E = routing_map_TSE.sum(dim=(0, 1))

        moe_out_TSH = self.routed_experts(x_TSH, weights_TSK, expert_ids_TSK, num_tokens_per_expert_E)

        if ep_mesh is not None and ep_mesh.size() > 1:
            assert ep_mesh is not None
            moe_out_TSH = head_combine(moe_out_TSH, ep_mesh, tokens)

        w_out = _local(self.proj_out.weight)
        moe_out_TD = F.linear(moe_out_TSH.reshape(tokens, -1), w_out, _local_or_none(self.proj_out.bias))
        out_TD = moe_out_TD + shared_out
        return out_TD.view(B, L, D)
