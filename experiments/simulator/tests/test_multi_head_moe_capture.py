# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head-MoE capture contracts: faithful SLA2 eager operators and memory plans."""

from collections import Counter

import pytest
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate
from torchtitan_simulator.validation import validate_workload

# Debug-model geometry (torchtitan_npu.models.multi_head_moe.config_registry):
# dim=256, vocab=102400, 8 layers (front-2/back-2 dense FFN, MoE in between),
# n_heads=4, head_dim=64, block=64, router proj 64, inter_dim=512,
# MoE heads S=4, head hidden hs=64, Es=16, Ks=4, expert mult 2 (inter=128),
# shared inter 128, FP32 parameters.
_SPARSE_FWD = "multi_head_moe_sim.sla2_sparse_attention.default"
_SPARSE_BWD = "multi_head_moe_sim.sla2_sparse_attention_grad.default"
_LINEAR_FWD = "multi_head_moe_sim.sla2_linear_attention.default"
_LINEAR_BWD = "multi_head_moe_sim.sla2_linear_attention_grad.default"
_ROUTER_FWD = "multi_head_moe_sim.router_soft_topk.default"
_ROUTER_BWD = "multi_head_moe_sim.router_soft_topk_backward.default"
_SWIGLU_FWD = "multi_head_moe_sim.swiglu.default"
_SWIGLU_BWD = "multi_head_moe_sim.swiglu_backward.default"

_D, _V, _HID = 256, 102400, 512
_H, _HD, _PROJ = 4, 64, 64
_HS, _ES, _MULT = 64, 16, 2
_SHARED = 128
_DENSE_LAYER = (
    4 * (_H * _HD * _D)  # to_q/k/v/out
    + 2 * _HD  # norm_q + norm_k
    + 2 * (_PROJ * _HD)  # block router proj_q/proj_k
    + _H  # alpha [H, 1]
    + 2 * _D  # attention_norm + ffn_norm
    + 3 * (_HID * _D)  # dense w1/w2/w3
)
_MOE_LAYER = (
    _DENSE_LAYER
    - 3 * (_HID * _D)
    + _D * (_H * _HS)  # proj_in
    + _H * _HS * _ES  # gate router [S, hs, Es]
    + 3 * (_ES * _H) * (_HS * _MULT) * _HS  # grouped experts w1/w2/w3
    + (_H * _HS) * _D  # proj_out
    + 3 * (_SHARED * _D)  # shared w1/w2/w3
)
_TOTAL_PARAMS = 2 * _V * _D + _D + 4 * _DENSE_LAYER + 4 * _MOE_LAYER
# Dense layer: qkv/out + 2 head norms + 2 router projs + alpha + 2 block norms
# + w1/w2/w3 = 14; MoE layer: 14 attention/norm entries + proj_in + gate
# router + 3 experts + proj_out + 3 shared = 20; plus embeddings, head and
# final norm.
_TOTAL_PARAM_TENSORS = 3 + 4 * 14 + 4 * 20


def test_multi_head_moe_param_formula_matches_debug_geometry():
    assert _TOTAL_PARAMS == 63_395_104
    assert _TOTAL_PARAM_TENSORS == 139


@pytest.mark.parametrize(
    "world,replicate,rank,ac",
    [
        (1, 1, 0, "none"),
        (2, 1, 0, "none"),
        (2, 1, 1, "none"),
        (4, 2, 0, "none"),
        (1, 1, 0, "full"),
        (1, 1, 0, "selective"),
    ],
    ids=["single", "fsdp-first", "fsdp-last", "hsdp", "full-ac", "selective-ac"],
)
def test_multi_head_moe_capture_reaches_iteration_with_sla2_eager_operators(world, replicate, rank, ac):
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=world,
            logical_rank=rank,
            data_parallel_replicate_degree=replicate,
            activation_checkpoint_mode=ac,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The SLA2 path is the reference's own eager operators; nothing is
    # substituted by virtual fused ops. 8 SLA2 layers. Full AC recomputes the
    # forward inside backward; the default selective policy saves only
    # upstream compute-intensive ops, so those are recomputed there as well.
    # Full AC and the default selective policy both recompute the SLA2 math
    # (only the topk mask output is saved by the selective policy). (The
    # linear branch einsums decompose into permute/bmm under capture;
    # masked_fill, topk and the RoPE complex views are the SLA2-only
    # signatures. Counts span the forward and backward graphs.)
    recompute = 2 if ac in ("full", "selective") else 1
    assert counts["aten.masked_fill.Scalar"] == 8 * recompute + 8
    assert counts["aten.view_as_complex.default"] == 16 * recompute + 16
    assert counts["aten.topk.default"] == 8 * (2 if ac == "full" else 1)
    assert not any(raw.startswith("multi_head_moe_sim.") for raw in counts)
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS
    degree = world // replicate
    memory = workload.iteration.schedule.annotations["memory_plan"]
    # The formulas below assume world is a power of two that divides the
    # parameter shard evenly (no padding imbalance); debug geometry guarantees
    # numel is even at every tested degree.
    local_params = _TOTAL_PARAMS if degree == 1 else _TOTAL_PARAMS // degree
    assert memory.persistent_param_bytes == local_params * 4
    # Native fused AdamW keeps two FP32 moments and one device FP32 step
    # scalar per participating parameter.
    assert memory.optimizer_state_bytes == local_params * 8 + _TOTAL_PARAM_TENSORS * 4
    # No virtual fusions remain: no node carries declared tensor semantics.
    assert all("tensor_semantics" not in n.attrs for n in nodes)


def test_multi_head_moe_ep2_dispatches_all_to_all_and_reaches_iteration():
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel",
            seq_len=16,
            world_size=2,
            logical_rank=0,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 4 MoE layers x (token dispatch + gradient combine) per dispatch pair.
    assert counts["_c10d_functional.all_to_all_single.default"] == 20
    comm = [n for n in nodes if n.op_type in ("all_to_all", "reduce_scatter")]
    assert len(comm) == 20 + 11
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS


def test_multi_head_moe_head_parallel_dispatches_static_all_to_all():
    workload = simulate(
        SimulationConfig(
            config="multi_head_moe_debugmodel_head_parallel",
            seq_len=16,
            world_size=2,
            logical_rank=0,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # 4 MoE layers x (head dispatch + head combine) forward plus their
    # backward: static shapes, so no token-count exchange precedes the data
    # all-to-alls (EP mode pays one extra count exchange per direction).
    assert counts["_c10d_functional.all_to_all_single.default"] == 16
    comm = [n for n in nodes if n.op_type in ("all_to_all", "reduce_scatter")]
    assert len(comm) == 16 + 11
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    assert len(gradients) == _TOTAL_PARAM_TENSORS


def _moe_layer_module_config(bias: bool):
    """Standalone small MultiHeadMoE geometry for flavor-level checks."""
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.moe import GroupedExperts, LocalTokenDispatcher, RoutedExperts

    from torchtitan_npu.models.multi_head_moe.model import MultiHeadGate, MultiHeadMoE

    def lin(in_features: int, out_features: int) -> Linear.Config:
        return Linear.Config(in_features=in_features, out_features=out_features, bias=bias)

    return MultiHeadMoE.Config(
        moe_heads=4,
        head_hidden=16,
        proj_in=lin(32, 64),
        proj_out=lin(64, 32),
        router=MultiHeadGate.Config(moe_heads=4, head_hidden=16, experts_per_head=8, k_per_head=2),
        routed_experts=RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(dim=16, hidden_dim=32, num_experts=32),
            token_dispatcher=LocalTokenDispatcher.Config(num_experts=32, top_k=2),
        ),
        shared_experts=FeedForward.Config(w1=lin(32, 64), w2=lin(64, 32), w3=lin(32, 64)),
    )


def test_multi_head_moe_head_parallel_keeps_linear_bias_and_gradients():
    """bias=True must stay equivalent to the default module-call path (PR #23 review).

    The head-parallel forward re-projects through bare ``F.linear``; the
    optional Linear bias has to ride along (outputs match the default flavor
    and both projection biases receive gradients).
    """
    import dataclasses

    import torch

    from torchtitan_npu.models.multi_head_moe.head_parallel import HeadParallelMultiHeadMoE

    base_cfg = _moe_layer_module_config(bias=True)
    torch.manual_seed(0)
    base = base_cfg.build()
    hp_cfg = HeadParallelMultiHeadMoE.Config(
        **{field.name: getattr(base_cfg, field.name) for field in dataclasses.fields(base_cfg) if field.init}
    )
    hp = hp_cfg.build()
    # GroupedExperts ships uninitialized ``torch.empty`` weights; give every
    # parameter a deterministic finite init so the comparison is stable.
    for parameter in base.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    hp.load_state_dict(base.state_dict())
    x_BLD = torch.randn(2, 5, base_cfg.proj_in.in_features)
    out_base = base(x_BLD)
    out_hp = hp(x_BLD)
    torch.testing.assert_close(out_hp, out_base)
    out_base.sum().backward()
    out_hp.sum().backward()
    torch.testing.assert_close(hp.proj_in.bias.grad, base.proj_in.bias.grad)
    torch.testing.assert_close(hp.proj_out.bias.grad, base.proj_out.bias.grad)


def _biased_head_parallel_preset(*, num_mtp_layers: int = 0):
    """Debug head-parallel preset with bias=True on both MoE projections.

    ``Linear.Config`` allows ``bias=True`` while every registered preset
    keeps it off, so the bias flavor rides on the simulator's documented
    ``module.factory`` escape hatch instead of a new production preset.
    """
    if num_mtp_layers:
        raise ValueError("multi_head_moe has no MTP modules; num_mtp_layers must be 0")
    from torchtitan_simulator.presets import multi_head_moe_debugmodel_head_parallel

    trainer = multi_head_moe_debugmodel_head_parallel()
    for layer in trainer.model_spec.model.layers:
        if layer.moe is not None:
            layer.moe.proj_in.bias = True
            layer.moe.proj_out.bias = True
    return trainer


def test_multi_head_moe_head_parallel_bias_capture_reaches_iteration():
    """End-to-end: a bias=True HP preset builds, captures and grads the biases (PR #23 review)."""
    workload = simulate(
        SimulationConfig(
            config=f"{__name__}._biased_head_parallel_preset",
            seq_len=16,
            world_size=2,
            logical_rank=0,
            expert_parallel_degree=2,
        )
    )
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The static all-to-all geometry is unchanged by the bias.
    assert counts["_c10d_functional.all_to_all_single.default"] == 16
    # The projections now run with bias: F.linear materializes addmm instead
    # of mm for the head-parallel proj_in/proj_out (4 MoE layers x 2 projections).
    assert counts["aten.addmm.default"] == 8
    gradients = workload.iteration.schedule.annotations["gradient_validation"]
    # 4 MoE layers x 2 projection biases join the gradient set.
    assert len(gradients) == _TOTAL_PARAM_TENSORS + 8


def test_multi_head_moe_head_parallel_rejects_indivisible_moe_heads_on_later_layer():
    """Divisibility must fail fast on every MoE layer, not only the first (PR #23 review)."""
    from torchtitan_simulator.presets import multi_head_moe_debugmodel_head_parallel

    trainer = multi_head_moe_debugmodel_head_parallel()
    trainer.parallelism.expert_parallel_degree = 4
    model = trainer.model_spec.model
    # Front-2/back-2 dense split: MoE layers are 2-5. Layer 2 keeps the
    # divisible geometry (4 % 4 == 0); layer 3 switches to a self-consistent
    # 6-head geometry that 4 does not divide.
    later = model.layers[3].moe
    assert later is not None
    later.moe_heads = 6
    later.proj_in.out_features = 6 * later.head_hidden
    later.proj_out.in_features = 6 * later.head_hidden
    later.router.moe_heads = 6
    later.routed_experts.inner_experts.num_experts = 6 * later.router.experts_per_head
    with pytest.raises(ValueError, match="MoE layer 3"):
        model.update_from_config(config=trainer)


def test_multi_head_moe_head_parallel_config_tolerates_dense_only_model():
    """A model without MoE layers must not crash the check with StopIteration (PR #23 review)."""
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.linear import Linear
    from torchtitan_simulator.presets import multi_head_moe_debugmodel_head_parallel

    trainer = multi_head_moe_debugmodel_head_parallel()
    trainer.parallelism.expert_parallel_degree = 2
    model = trainer.model_spec.model
    for layer in model.layers:
        if layer.moe is not None:
            dim = layer.attention.to_q.in_features
            layer.moe = None
            layer.feed_forward = FeedForward.Config(
                w1=Linear.Config(in_features=dim, out_features=128, bias=False),
                w2=Linear.Config(in_features=128, out_features=dim, bias=False),
                w3=Linear.Config(in_features=dim, out_features=128, bias=False),
            )
    # Must complete (no StopIteration from the divisibility check).
    model.update_from_config(config=trainer)


def _head_parallel_two_rank_worker(rank: int, rendezvous: str, queue) -> None:
    """Forked gloo worker: uneven token shards fail fast, even shards dispatch (PR #23 review)."""
    import dataclasses
    import traceback
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from torchtitan_npu.models.multi_head_moe.head_parallel import HeadParallelMultiHeadMoE

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        base_cfg = _moe_layer_module_config(bias=False)
        # Devices=[] keeps the fork CPU-only: this worker runs meta/cpu math
        # and a default fork_rng() would touch the NPU rng state, which
        # cannot re-initialize in a forked subprocess.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(313)
            module = HeadParallelMultiHeadMoE.Config(
                **{field.name: getattr(base_cfg, field.name) for field in dataclasses.fields(base_cfg) if field.init}
            ).build()
        # GroupedExperts ships uninitialized ``torch.empty`` weights; give
        # every parameter a deterministic finite init.
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        # Mirror the production EP expert placement: the grouped experts ship
        # Shard(0) on the expert axis, so rank r holds experts
        # [r * E/ep, (r+1) * E/ep) and the HP forward routes only the owned
        # heads' local expert pool (head_local * experts_per_head).
        experts = module.routed_experts.inner_experts
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            parameter = getattr(experts, name)
            setattr(
                experts,
                name,
                torch.nn.Parameter(parameter.detach().chunk(2, dim=0)[rank].clone()),
            )
        module._ep_mesh_override = mesh

        # Uneven token shards must raise before any collective: the static
        # all-to-alls size every transfer block from the local shard, so a
        # mismatch would error or hang inside the exchange instead of ever
        # reaching the reassembly view.
        tokens = 5 if rank == 0 else 3
        try:
            module(torch.randn(1, tokens, base_cfg.proj_in.in_features))
        except ValueError as error:
            assert "token shards" in str(error), str(error)
        else:
            raise AssertionError("uneven token shards must raise ValueError")

        # Regression sequence for the validation-cache skew (PR #23 review
        # round 3): even shards dispatch for real, then ONE rank's token
        # count changes. The validation is uncached, so both ranks enter the
        # same all-gather on the second call and both raise; a skip keyed on
        # the unchanged local count would send rank 1 straight to the data
        # all-to-all and skew the group into a hang.
        out = module(torch.randn(1, 4, base_cfg.proj_in.in_features))
        assert tuple(out.shape) == (1, 4, base_cfg.proj_out.out_features)

        tokens = 5 if rank == 0 else 4
        try:
            module(torch.randn(1, tokens, base_cfg.proj_in.in_features))
        except ValueError as error:
            assert "token shards" in str(error), str(error)
        else:
            raise AssertionError("uneven token shards after an even forward must raise ValueError")

        # Back to even shards: repeated forwards keep dispatching for real.
        out = module(torch.randn(1, 4, base_cfg.proj_in.in_features))
        assert tuple(out.shape) == (1, 4, base_cfg.proj_out.out_features)
        queue.put(("ok", rank, ""))
    except BaseException:
        queue.put(("error", rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def test_multi_head_moe_head_parallel_two_rank_token_shard_validation(tmp_path):
    """Two-rank gloo: uneven ep token shards fail fast before the collectives (PR #23 review).

    Meta capture has no live peers to ask, so the precondition's enforcement
    is exercised over a real two-rank process group instead: uneven per-rank
    token counts raise ``ValueError`` on both ranks before any all-to-all
    (both on the first forward and again after one rank's count changes from
    an even baseline — the uncached validation keeps both ranks in the same
    collective), and an even shard set dispatches for real, repeatedly.
    """
    import multiprocessing

    from torchtitan_npu.models.multi_head_moe.head_parallel import (
        HeadParallelMultiHeadMoE,  # noqa: F401 — loaded before the fork so workers inherit it
    )

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [
        context.Process(target=_head_parallel_two_rank_worker, args=(rank, str(tmp_path / "rendezvous"), queue))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=90) for _ in workers]
        for worker in workers:
            worker.join(timeout=15)
            assert worker.exitcode == 0
        failures = [text for status, _, text in sorted(results, key=lambda item: item[1]) if status == "error"]
        if failures:
            raise AssertionError("two-rank worker failed:\n" + "\n".join(failures))
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=5)
        queue.close()


def test_multi_head_moe_rejects_tensor_parallel_at_config_level():
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                tensor_parallel_degree=2,
            )
        )


def test_multi_head_moe_rejects_context_parallel():
    with pytest.raises(NotImplementedError, match="context parallel"):
        simulate(
            SimulationConfig(
                config="multi_head_moe_debugmodel",
                seq_len=16,
                world_size=2,
                context_parallel_degree=2,
            )
        )


def test_multi_head_moe_preset_rejects_mtp_modules():
    from torchtitan_simulator.presets import multi_head_moe_debugmodel

    with pytest.raises(ValueError, match="no MTP modules"):
        multi_head_moe_debugmodel(num_mtp_layers=1)


def test_multi_head_moe_opt_in_fusion_replaces_only_the_two_branch_seams():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=16, enable_sla2_fusion=True))
    result = validate_workload(workload)
    assert result["graph_acyclic"] and result["peak_storage_unique"]
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # The fusion scope is the four production seams (8 SLA2 layers): the two
    # SLA2 branch seams (8 forward + 8 backward virtual ops per branch), the
    # router seam (topk absorbed into the soft-topk virtual pair) and the
    # MoE FFN activation seam (swiglu virtual pair). Everything around them
    # stays native (the remaining sigmoid is still captured eagerly).
    assert counts[_SPARSE_FWD] == 8
    assert counts[_LINEAR_FWD] == 8
    assert counts[_SPARSE_BWD] == 8
    assert counts[_LINEAR_BWD] == 8
    assert counts[_ROUTER_FWD] == 8
    assert counts[_ROUTER_BWD] == 8
    assert counts[_SWIGLU_FWD] == 12
    assert counts[_SWIGLU_BWD] == 12
    assert counts["aten.sigmoid.default"] == 8
    assert not any(raw.startswith("aten.") and "topk" in raw for raw in counts)
    assert not any(
        raw.startswith("aten.") and "masked_fill" in raw for raw in counts
    )  # the score masking now lives inside the fused boundary

    # The fused ops carry declared tensor semantics; no eager node does.
    custom = [n for n in nodes if "tensor_semantics" in n.attrs]
    assert {n.annotations["raw_op_type"] for n in custom} == {
        _SPARSE_FWD,
        _SPARSE_BWD,
        _LINEAR_FWD,
        _LINEAR_BWD,
        _ROUTER_FWD,
        _ROUTER_BWD,
        _SWIGLU_FWD,
        _SWIGLU_BWD,
    }

    # Mask gradient routing matches the eager reference per branch (PR #19
    # review): the sparse mask enters the eager branch through the boolean
    # gate ``masked_fill(mask < 1e-6)``, so the fused sparse backward feeds
    # only the output mix, while the linear complement mask is a differentiable
    # einsum operand and its backward joins the ``M_not = 1 - M_c`` chain.
    sparse_bwd_ids = {n.op_id for n in nodes if n.annotations["raw_op_type"] == _SPARSE_BWD}
    sparse_consumers = {c.annotations["raw_op_type"] for c in nodes if sparse_bwd_ids & set(c.predecessors)}
    assert sparse_consumers == {"aten.add.Tensor"}
    linear_bwd_ids = {n.op_id for n in nodes if n.annotations["raw_op_type"] == _LINEAR_BWD}
    linear_consumers = {c.annotations["raw_op_type"] for c in nodes if linear_bwd_ids & set(c.predecessors)}
    assert {"aten.neg.default"} <= linear_consumers


def test_multi_head_moe_seq_len_128_exercises_real_block_selection():
    workload = simulate(SimulationConfig(config="multi_head_moe_debugmodel", seq_len=128))
    nodes = [n for template in workload.step_templates.values() for n in template.nodes.values()]
    counts = Counter(n.annotations["raw_op_type"] for n in nodes)
    # Two 64-token key blocks per head at seq 128; one topk block-selection
    # per layer forward (8 SLA2 layers), and the sparse branch materializes
    # the full [B, H, T, T] score matrix (forward + gradient fill).
    assert counts["aten.topk.default"] == 8
    score_nodes = [n for n in nodes if n.outputs and tuple(n.outputs[0].shape) == (1, 4, 128, 128)]
    sig = Counter(
        n.annotations["raw_op_type"]
        for n in score_nodes
        if n.annotations["raw_op_type"] in ("aten.div.Tensor", "aten.masked_fill.Scalar", "aten._softmax.default")
    )
    assert sig["aten.div.Tensor"] == 16
    assert sig["aten.masked_fill.Scalar"] == 16
    assert sig["aten._softmax.default"] == 8
    soft_mask = [
        n
        for n in nodes
        if n.annotations["raw_op_type"] == "aten.sigmoid.default" and tuple(n.outputs[0].shape) == (1, 4, 2, 2)
    ]
    assert len(soft_mask) == 8
