# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Acceptance checks for serializable workload semantics, not hardware timing."""

import json

import pytest
import torch
from model import BlockDiffusionModel, ModelConfig, RoutingPlan
from workload import (
    ParallelSpec,
    PhaseSpec,
    analyze_phase,
    analyze_schedule,
    build_schedule,
)


def config(**overrides):
    options = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        num_experts=0,
        num_shared_experts=0,
        dtype=torch.float32,
        kv_dtype=torch.float32,
    )
    options.update(overrides)
    return ModelConfig(**options)


def phase(**overrides):
    options = dict(phase="denoise", batch_size=2, query_length=4, prefix_length=3)
    options.update(overrides)
    return PhaseSpec(**options)


def test_multiblock_schedule_keeps_prefix_steps_and_head_work_distinct():
    schedule = build_schedule(
        prompt_length=3,
        block_lengths=(4, 2),
        commit_counts=((2, 2), (1, 1)),
        batch_size=2,
        commit_last=False,
        head_policy="unresolved",
    )
    assert [item.phase for item in schedule] == ["prefill", "denoise", "denoise", "commit", "denoise", "denoise"]
    denoise = [item for item in schedule if item.phase == "denoise"]
    assert [item.prefix_length for item in denoise] == [3, 3, 7, 7]
    assert [item.query_length for item in denoise] == [4, 4, 2, 2]
    assert [item.unresolved_tokens for item in denoise] == [4, 2, 2, 1]
    assert [item.head_tokens for item in denoise] == [4, 2, 2, 1]
    assert all(item.head_selection == "unresolved" and item.head_positions is None for item in denoise)
    assert all(item.head_positions == () for item in schedule if item.phase != "denoise")
    report = analyze_schedule(config(), schedule)
    assert report.generated_tokens == 12
    serialized = json.loads(json.dumps(report.to_dict()))
    assert serialized["generated_tokens"] == 12
    assert serialized["model_config"]["hidden_size"] == 16
    assert serialized["model_config"]["dtype"] == "float32"
    assert len(serialized["phases"]) == 6


def test_final_commit_is_explicit_and_changes_cache_not_generated_tokens():
    kwargs = dict(prompt_length=3, block_lengths=(4,), commit_counts=((2, 2),), batch_size=1)
    without = analyze_schedule(config(), build_schedule(**kwargs, commit_last=False))
    with_commit = analyze_schedule(config(), build_schedule(**kwargs, commit_last=True))
    assert without.generated_tokens == with_commit.generated_tokens == 4
    assert len(with_commit.phases) == len(without.phases) + 1
    assert with_commit.phases[-1].phase.phase == "commit"
    assert without.phases[-1].ranks[0].kv_after_bytes == without.phases[-1].ranks[0].kv_before_bytes
    assert with_commit.phases[-1].ranks[0].kv_after_bytes > with_commit.phases[-1].ranks[0].kv_before_bytes


@pytest.mark.parametrize("moe,tied", [(False, False), (False, True), (True, False), (True, True)])
def test_single_rank_parameter_and_kv_storage_match_real_model(moe, tied):
    cfg = config(
        num_experts=4 if moe else 0,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        num_shared_experts=1 if moe else 0,
        tie_word_embeddings=tied,
        kv_dtype=torch.bfloat16,
    )
    model = BlockDiffusionModel(cfg)
    report = analyze_phase(cfg, phase(phase="commit"))
    rank = report.ranks[0]
    assert rank.parameter_bytes == model.parameter_memory_bytes()
    with torch.no_grad():
        cache = model(torch.zeros((2, 3), dtype=torch.long), phase="prefill", logits_positions=()).cache
        after = model(torch.ones((2, 4), dtype=torch.long), cache=cache, phase="commit", logits_positions=()).cache
    assert rank.kv_before_bytes == cache.memory_bytes()
    assert rank.kv_after_bytes == after.memory_bytes()


def test_matrix_event_flops_match_independent_shape_arithmetic():
    cfg = config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12)
    description = analyze_phase(cfg, phase())
    matrix_count = 0
    for rank in description.ranks:
        for event in rank.events:
            if event.kind != "matmul":
                continue
            left, right = event.inputs[:2]
            assert len(left.shape) == len(right.shape) == 2
            assert left.shape[-1] == right.shape[-2]
            assert event.output.shape == (left.shape[0], right.shape[1])
            assert event.flops == 2 * left.shape[0] * left.shape[1] * right.shape[1]
            matrix_count += 1
    assert matrix_count >= cfg.num_hidden_layers * 7


def test_no_head_work_when_logits_are_not_requested():
    cfg = config()
    full = analyze_phase(cfg, phase(head_positions=None))
    selected = analyze_phase(cfg, phase(head_positions=(3,)))
    absent = analyze_phase(cfg, phase(head_positions=()))
    head_flops = lambda description: sum(
        event.flops or 0 for rank in description.ranks for event in rank.events if event.name.endswith("lm_head")
    )
    assert head_flops(full) == 2 * 2 * 4 * 16 * 32
    assert head_flops(selected) == 2 * 2 * 1 * 16 * 32
    assert head_flops(absent) == 0


def test_sparse_ep_conserves_pairs_and_has_zero_work_ranks():
    cfg = config(num_experts=16, num_experts_per_tok=2, moe_intermediate_size=12)
    replay = RoutingPlan(expert_indices=((0, 15),))
    description = analyze_phase(
        cfg,
        phase(batch_size=1, query_length=1, prefix_length=0, routing_plan=replay),
        ParallelSpec(tp_size=1, ep_size=16, ep_layout="batch_sharded"),
    )
    assert len(description.ranks) == 16
    assert sum(rank.local_batch_size for rank in description.ranks) == 1
    for layer in range(cfg.num_hidden_layers):
        assert sum(rank.routed_pairs_per_layer[layer] for rank in description.ranks) == 2
        assert sum(rank.routed_pairs_per_layer[layer] == 0 for rank in description.ranks) == 14
        matrix = description.routing_matrices[layer]
        assert sum(sum(row) for row in matrix) == 2
        assert matrix[0][0] == matrix[0][15] == 1
    for rank in description.ranks[1:]:
        assert rank.local_batch_size == 0
        assert rank.kv_before_bytes == rank.kv_after_bytes == 0
    remote_rank = next(rank for rank in description.ranks if rank.ep_rank == 15)
    assert remote_rank.routed_pairs_per_layer == (1, 1)


def test_ep_communication_payloads_conserve_sent_and_received_bytes():
    cfg = config(num_experts=4, num_experts_per_tok=1, moe_intermediate_size=12)
    replay = RoutingPlan(expert_indices=((0,), (3,), (0,), (1,), (2,), (3,)))
    description = analyze_phase(
        cfg,
        phase(batch_size=3, query_length=2, prefix_length=1, routing_plan=replay),
        ParallelSpec(tp_size=1, ep_size=4, ep_layout="batch_sharded"),
    )
    assert [rank.local_batch_size for rank in description.ranks] == [1, 1, 1, 0]
    for layer, matrix in enumerate(description.routing_matrices):
        assert matrix == ((1, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0))
        collectives = [
            event
            for rank in description.ranks
            for event in rank.events
            if event.layer == layer and (event.metadata or {}).get("operation") == "all_to_all"
        ]
        assert collectives
        assert sum(event.payload_bytes for event in collectives) == sum(
            event.received_payload_bytes for event in collectives
        )
        # Three remote pairs each dispatch and return a full hidden vector.
        assert sum(event.payload_bytes for event in collectives) >= 2 * 3 * 16 * 4
        assert all(len(event.metadata["group"]) == 4 for event in collectives)


def test_tp_projection_shapes_and_collective_groups():
    cfg = config()
    description = analyze_phase(cfg, phase(), ParallelSpec(tp_size=2))
    assert len(description.ranks) == 2
    for rank in description.ranks:
        q = [event for event in rank.events if event.name.endswith("q_proj")]
        kv = [event for event in rank.events if event.name.endswith("k_proj")]
        ff = [event for event in rank.events if event.name.endswith("gate_proj")]
        assert q and kv and ff
        assert all(event.output.shape[-1] == 8 for event in q)
        assert all(event.output.shape[-1] == 4 for event in kv)
        assert all(event.output.shape[-1] == 12 for event in ff)
        allreduces = [event for event in rank.events if (event.metadata or {}).get("operation") == "all_reduce"]
        assert len(allreduces) >= 2 * cfg.num_hidden_layers
        assert all(len(event.metadata["group"]) == 2 for event in allreduces)
        assert rank.kv_before_bytes == 2 * 2 * 3 * 1 * 4 * 2 * 4


@pytest.mark.parametrize(
    "parallel",
    [
        ParallelSpec(tp_size=3),
        ParallelSpec(tp_size=4),  # KV replication is not implicitly supported.
        ParallelSpec(ep_size=3),
    ],
)
def test_unsupported_sharding_fails_explicitly(parallel):
    with pytest.raises((ValueError, TypeError)):
        analyze_phase(config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12), phase(), parallel)


@pytest.mark.parametrize("counts", [((1, 1),), ((5,),), ((2, -1, 3),)])
def test_invalid_commit_schedule_cannot_silently_change_generated_work(counts):
    with pytest.raises((ValueError, TypeError)):
        build_schedule(prompt_length=3, block_lengths=(4,), commit_counts=counts)


def test_kv_dtype_conversion_is_explicit_on_both_read_and_write():
    cfg = config(dtype=torch.float32, kv_dtype=torch.bfloat16)
    report = analyze_phase(cfg, phase(phase="commit"))
    casts = [event for event in report.ranks[0].events if event.kind == "cast" and event.output is not None]
    for layer in range(cfg.num_hidden_layers):
        layer_casts = [event for event in casts if event.layer == layer]
        assert any(event.inputs[0].dtype == "bfloat16" and event.output.dtype == "float32" for event in layer_casts)
        assert any(event.inputs[0].dtype == "float32" and event.output.dtype == "bfloat16" for event in layer_casts)


def test_router_selection_uses_fp32_scores_with_topk_sized_outputs():
    cfg = config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12, dtype=torch.bfloat16)
    report = analyze_phase(cfg, phase())
    events = report.ranks[0].events
    topk = [event for event in events if "router" in event.name and "topk" in event.name]
    assert len(topk) == cfg.num_hidden_layers
    for event in topk:
        assert event.inputs[0].dtype == "float32"
        assert event.inputs[0].shape == (8, 4)
        assert event.output.dtype == "float32"
        assert event.output.shape == (8, 2)


def test_output_positions_match_model_nonnegative_index_contract():
    with pytest.raises(ValueError):
        phase(head_positions=(-1,))
    # Repeating a position is legal in the reference and costs repeated head rows.
    duplicate = analyze_phase(config(), phase(head_positions=(1, 1)))
    head = [event for event in duplicate.ranks[0].events if event.name.endswith("lm_head")]
    assert len(head) == 1
    assert head[0].output.shape[0] == 4


def test_combined_tp_ep_preserves_projection_work_and_global_kv_storage():
    cfg = config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12, num_shared_experts=1)
    first = RoutingPlan.balanced(6, 4, 2)
    second = RoutingPlan(expert_indices=((0, 1),) * 6)
    item = phase(batch_size=3, query_length=2, prefix_length=5, routing_plan=(first, second))
    reference = analyze_phase(cfg, item)
    sharded = analyze_phase(cfg, item, ParallelSpec(tp_size=2, ep_size=2))
    assert sum(rank.kv_before_bytes for rank in sharded.ranks) == reference.ranks[0].kv_before_bytes

    def projection_flops(description):
        return sum(
            event.flops
            for rank in description.ranks
            for event in rank.events
            if event.kind == "matmul" and event.name != "router"
        )

    assert projection_flops(sharded) == projection_flops(reference)
    for lane in range(2):
        lane_ranks = [rank for rank in sharded.ranks if rank.tp_rank == lane]
        assert sum(rank.local_batch_size for rank in lane_ranks) == 3
        for layer in range(2):
            assert sum(rank.routed_pairs_per_layer[layer] for rank in lane_ranks) == 12
    assert sharded.routing_matrices[0] != sharded.routing_matrices[1]
    for rank in sharded.ranks:
        for event in rank.events:
            metadata = event.metadata or {}
            if metadata.get("operation") == "all_reduce":
                assert tuple(metadata["group"]) == tuple(rank.ep_rank * 2 + lane for lane in range(2))
            if metadata.get("operation") == "all_to_all":
                assert tuple(metadata["group"]) == (rank.tp_rank, 2 + rank.tp_rank)


def test_sampling_uses_global_vocab_reductions_and_probability_confidence():
    cfg = config(dtype=torch.bfloat16)
    spec = phase(
        head_positions=(1, 3),
        unresolved_tokens=2,
        accepted_tokens=1,
        sampling_policy="greedy_confidence",
        block_index=0,
        step_index=0,
    )
    report = analyze_phase(cfg, spec, ParallelSpec(tp_size=2))
    for rank in report.ranks:
        events = {event.name: event for event in rank.events}
        assert events["sampling.logits_cast"].inputs[0].dtype == "bfloat16"
        assert events["sampling.logits_cast"].output.dtype == "float32"
        assert events["sampling.local_argmax"].inputs[0].shape == (2, 2, 16)
        assert events["sampling.global_argmax"].metadata["reduction"] == "maxloc"
        assert events["sampling.global_argmax"].payload_bytes == 2 * 2 * (4 + 8)
        assert events["sampling.global_normalizer"].metadata["reduction"] == "sum"
        assert events["sampling.global_normalizer"].payload_bytes == 2 * 2 * 4
        assert events["sampling.confidence"].inputs[0].dtype == "float32"
        assert events["sampling.select_positions"].output.shape == (2, 1)
        assert events["sampling.update_canvas"].output.shape == (2, 4)
        assert events["sampling.update_canvas"].metadata["accepted_tokens_per_request"] == 1
        assert events["logits.output"].metadata["sampling_included"] is True


def test_partial_schedule_output_denominator_counts_accepted_tokens_only():
    spec = phase(
        head_positions=None,
        unresolved_tokens=4,
        accepted_tokens=1,
        sampling_policy="greedy_confidence",
        block_index=0,
        step_index=0,
    )
    report = analyze_schedule(config(), (spec,))
    assert report.generated_tokens == 2  # B=2, one accepted token each, not B*C=8.


def test_sampling_rejects_missing_or_impossible_acceptance():
    with pytest.raises(ValueError):
        phase(sampling_policy="greedy_confidence")
    with pytest.raises(ValueError):
        phase(sampling_policy="greedy_confidence", accepted_tokens=3, head_positions=(0, 1))
    with pytest.raises(ValueError):
        phase(sampling_policy="greedy_confidence", accepted_tokens=2, unresolved_tokens=1)


def test_schedule_rejects_committing_a_known_incomplete_block():
    first = phase(block_index=0, step_index=0, unresolved_tokens=4, accepted_tokens=1)
    commit = phase(phase="commit", block_index=0, head_positions=(), unresolved_tokens=0)
    with pytest.raises(ValueError):
        analyze_schedule(config(), (first, commit))


def test_schedule_rejects_duplicate_commit_and_reopened_block():
    first = phase(block_index=0, step_index=0, unresolved_tokens=4, accepted_tokens=4)
    commit = phase(phase="commit", block_index=0, head_positions=(), unresolved_tokens=0)
    duplicate = phase(phase="commit", block_index=0, prefix_length=7, head_positions=())
    reopened = phase(block_index=0, step_index=1, prefix_length=7, unresolved_tokens=0, accepted_tokens=0)
    for later in (duplicate, reopened):
        with pytest.raises(ValueError):
            analyze_schedule(config(), (first, commit, later))


def test_schedule_rejects_inconsistent_remaining_token_counts():
    first = phase(block_index=0, step_index=0, unresolved_tokens=4, accepted_tokens=2)
    wrong_remaining = phase(block_index=0, step_index=1, unresolved_tokens=4, accepted_tokens=2)
    with pytest.raises(ValueError):
        analyze_schedule(config(), (first, wrong_remaining))


def test_stalled_denoising_steps_preserve_compute_and_accept_nothing():
    schedule = build_schedule(
        prompt_length=1,
        block_lengths=(4,),
        commit_counts=((0, 2, 0, 2),),
        batch_size=2,
        commit_last=False,
    )
    steps = [item for item in schedule if item.phase == "denoise"]
    assert [item.accepted_tokens for item in steps] == [0, 2, 0, 2]
    assert [item.unresolved_tokens for item in steps] == [4, 4, 2, 2]
    report = analyze_schedule(config(), schedule)
    assert report.generated_tokens == 8
    for described in report.phases:
        if described.phase.phase != "denoise":
            continue
        assert described.ranks[0].matrix_flops > 0
        assert any(event.name == "sampling.local_argmax" for event in described.ranks[0].events)
    for previous, current in ((report.phases[1], report.phases[2]), (report.phases[3], report.phases[4])):
        assert previous.ranks[0].matrix_flops == current.ranks[0].matrix_flops
        assert previous.ranks[0].kv_after_bytes == current.ranks[0].kv_before_bytes


@pytest.mark.parametrize("head_policy", ["all", "unresolved"])
def test_sampling_tracks_unresolved_state_across_partial_and_stalled_rounds(head_policy):
    schedule = build_schedule(
        prompt_length=1,
        block_lengths=(4,),
        commit_counts=((2, 0, 2),),
        batch_size=2,
        head_policy=head_policy,
    )
    report = analyze_schedule(config(), schedule)
    steps = [description for description in report.phases if description.phase.phase == "denoise"]
    first = {event.name: event for event in steps[0].ranks[0].events}
    initialized = first["sampling.unresolved_init"]
    assert initialized.kind == "state_init"
    assert initialized.output.shape == (2, 4)
    assert initialized.output.dtype == "bool"
    for index, described in enumerate(steps):
        events = {event.name: event for event in described.ranks[0].events}
        selected = events["sampling.select_positions"]
        masks = [item for item in selected.inputs if item.dtype == "bool"]
        assert len(masks) == 1
        assert masks[0].shape == (2, 4)
        if head_policy == "all":
            assert tuple(selected.metadata["head_positions"]) == tuple(range(4))
        else:
            unresolved = described.phase.unresolved_tokens
            assert described.phase.head_positions is None
            assert described.phase.head_selection == "unresolved"
            assert "head_positions" not in selected.metadata
            assert selected.metadata["head_position_source"] == "unresolved_mask"
            compact = events["head.compact_unresolved"]
            assert compact.inputs[0].shape == (2, 4)
            assert compact.inputs[0].dtype == "bool"
            assert compact.output.shape == (2, unresolved)
            assert compact.output.dtype == "int64"
            gather = events["head.select"]
            assert gather.inputs[1].shape == (2, unresolved)
            assert gather.inputs[1].dtype == "int64"
            assert gather.output.shape == (2, unresolved, 16)
            assert selected.inputs[-1].shape == (2, unresolved)
            assert selected.inputs[-1].dtype == "int64"
            assert selected.metadata["head_index_map"]["shape"] == [2, unresolved]
        assert selected.metadata["output_indices"] == "block_positions"
        if index:
            assert "sampling.unresolved_init" not in events
            assert events["sampling.unresolved_read"].output.shape == (2, 4)
            assert described.phase.unresolved_tokens == 2
        if described.phase.accepted_tokens == 0:
            assert "sampling.local_argmax" in events
            assert "sampling.unresolved_update" not in events
            assert "sampling.update_canvas" not in events
        else:
            updated = events["sampling.unresolved_update"]
            assert updated.kind == "state_write"
            assert updated.output.dtype == "bool"
            assert updated.output.shape == (2, 4)
            assert updated.inputs[1].shape == (2, described.phase.accepted_tokens)
            if head_policy == "unresolved":
                canvas_update = events["sampling.update_canvas"]
                assert len(canvas_update.inputs) == 4
                assert canvas_update.inputs[-1].shape == (2, described.phase.unresolved_tokens)
                assert canvas_update.inputs[-1].dtype == "int64"


def test_dynamic_unresolved_selection_rejects_fabricated_positions():
    with pytest.raises(ValueError):
        phase(head_selection="unresolved", unresolved_tokens=2, head_positions=(0, 1))
    with pytest.raises(ValueError):
        phase(head_selection="unresolved", unresolved_tokens=None)


def test_unresolved_head_compaction_exists_without_sampling_consumer():
    spec = phase(head_selection="unresolved", unresolved_tokens=2, head_positions=None, sampling_policy="none")
    description = analyze_phase(config(), spec)
    events = {event.name: event for event in description.ranks[0].events}
    assert events["head.compact_unresolved"].output.shape == (2, 2)
    assert events["head.select"].output.shape == (2, 2, 16)
    assert events["lm_head"].output.shape[0] == 4
    assert not any(name.startswith("sampling.update") for name in events)
