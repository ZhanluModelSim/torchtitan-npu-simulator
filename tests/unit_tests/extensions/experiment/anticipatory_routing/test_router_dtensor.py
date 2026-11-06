# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real token-sharded Gloo routing through initialization, capture and replay."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from torchtitan_npu.extensions.experiment.anticipatory_routing.router import build_routing_cache
from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router


def _router():
    from torchtitan.models.common.activation import Sigmoid
    from torchtitan.models.common.linear import RouterGateLinear

    config = DeepSeekV41Router.Config(
        num_experts=4,
        gate=RouterGateLinear.Config(in_features=4, out_features=4),
        score_func=Sigmoid.Config(),
        top_k=2,
        sorted_topk=True,
        route_norm=True,
        route_scale=1.4,
    )
    with torch.random.fork_rng(devices=[]):
        result = config.build()
    assert isinstance(result, DeepSeekV41Router)
    return result


def _replay_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    cache = None
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        inputs = torch.tensor(
            [
                [1.0, 0.2, -0.5, 0.7],
                [-0.3, 1.4, 0.6, -0.2],
                [0.8, -0.7, 1.1, 0.4],
                [-0.9, 0.1, 0.3, 1.5],
                [0.4, 0.9, -1.2, 0.5],
                [1.3, -0.4, 0.2, -0.6],
            ]
        )
        capture_weight = torch.tensor(
            [[0.7, -0.2, 0.5, 0.1], [-0.4, 0.8, 0.2, -0.6], [0.3, 0.1, -0.7, 0.9], [-0.8, -0.5, 0.6, 0.4]]
        )
        router = _router()
        router.gate.weight = torch.nn.Parameter(distribute_tensor(capture_weight.clone(), mesh, [Replicate()]))
        # Token statistics are rank-local partial sums, as in the upstream router sharding plan.
        router.tokens_per_expert_E = DTensor.from_local(torch.zeros(4), mesh, [Partial()], run_check=False)
        sharded_inputs = distribute_tensor(inputs.clone(), mesh, [Shard(0)]).requires_grad_()
        cache = build_routing_cache([router], index_store_dtype="int16", device=torch.device("cpu"))
        key = router._anticipatory_key
        buffer = cache.replay_buffers[key]
        dense_capture = F.linear(inputs, capture_weight).sigmoid()
        expected_ids = torch.tensor(
            [sorted(range(4), key=lambda expert: row[expert], reverse=True)[:2] for row in dense_capture.tolist()]
        )
        row = slice(rank * 3, (rank + 1) * 3)
        with torch.no_grad():
            cache.begin_initialization()
            try:
                _, initial_ids, _ = router(sharded_inputs)
            finally:
                cache.reset()
            assert isinstance(initial_ids, DTensor)
            assert initial_ids.shape == (6, 2) and initial_ids.placements == (Shard(0),)
            torch.testing.assert_close(initial_ids.full_tensor(), expected_ids, rtol=0, atol=0)
            assert buffer.indices.shape == buffer.capture_indices.shape == (3, 2)
            assert not isinstance(buffer.indices, DTensor)
            slot = {}
            cache.begin_capture(slot)
            try:
                router(sharded_inputs)
                cache.save_captured_routes()
            finally:
                cache.reset()
            assert slot[key].shape == (3, 2) and slot[key].dtype == torch.int16
            torch.testing.assert_close(slot[key].long(), expected_ids[row], rtol=0, atol=0)
            # The live top-2 changes to its complement, so replay cannot pass by
            # accidentally returning the newly computed IDs or old gate scores.
            router.gate.weight.to_local().copy_(-capture_weight)

        cache.begin_training([slot])
        cache.select(0)
        selected, replay_ids, routing_map = router(sharded_inputs)
        assert isinstance(replay_ids, DTensor)
        assert replay_ids.device_mesh == mesh and replay_ids.placements == (Shard(0),)
        assert replay_ids.shape == (6, 2) and replay_ids.to_local().shape == (3, 2)
        torch.testing.assert_close(replay_ids.full_tensor(), expected_ids, rtol=0, atol=0)
        live_scores = F.linear(inputs, -capture_weight).sigmoid()
        assert not torch.equal(live_scores.topk(2, -1).indices, expected_ids)
        reference_selected = live_scores.gather(-1, expected_ids)
        reference_selected = reference_selected / (reference_selected.sum(-1, keepdim=True) + 1e-20) * 1.4
        expected_map = torch.zeros_like(live_scores, dtype=torch.bool).scatter_(-1, expected_ids, True)
        torch.testing.assert_close(routing_map.full_tensor(), expected_map, rtol=0, atol=0)
        torch.testing.assert_close(selected.full_tensor(), reference_selected, rtol=1e-6, atol=1e-7)
        upstream = torch.tensor([[0.3, -0.6], [1.2, 0.7], [-0.4, 1.1], [0.8, -0.2], [-0.9, 0.5], [1.4, -1.0]])
        (selected.to_local() * upstream[row]).sum().backward()

        reference_inputs = inputs.double().requires_grad_()
        reference_weight = (-capture_weight).double().requires_grad_()
        reference_scores = F.linear(reference_inputs, reference_weight).sigmoid().gather(-1, expected_ids)
        reference_scores = reference_scores / (reference_scores.sum(-1, keepdim=True) + 1e-20) * 1.4
        (reference_scores * upstream.double()).sum().backward()
        assert sharded_inputs.grad.placements == (Shard(0),)
        torch.testing.assert_close(
            sharded_inputs.grad.full_tensor(), reference_inputs.grad.float(), rtol=2e-6, atol=2e-7
        )
        torch.testing.assert_close(
            router.gate.weight.grad.full_tensor(), reference_weight.grad.float(), rtol=2e-6, atol=2e-7
        )
    finally:
        if cache is not None:
            cache.reset()
        dist.destroy_process_group()


def test_anticipatory_replay_preserves_dtensor_ids_and_current_score_gradients(tmp_path):
    mp.spawn(_replay_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_cache_wraps_dynamic_router_but_not_v4_hash_router():
    from torchtitan.models.common.activation import Sigmoid
    from torchtitan.models.common.linear import RouterGateLinear
    from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

    gate = RouterGateLinear.Config(in_features=4, out_features=4)
    with torch.random.fork_rng(devices=[]):
        hash_router = DeepSeekV4Router.Config(
            num_experts=4, gate=gate, score_func=Sigmoid.Config(), top_k=2, vocab_size=8, layer_id=0, n_hash_layers=1
        ).build()
        dynamic_router = _router()
    model = torch.nn.ModuleList([hash_router, dynamic_router])
    hash_selection = hash_router._select_experts
    dynamic_selection = dynamic_router._select_experts

    cache = build_routing_cache([model], index_store_dtype="int16", device=torch.device("cpu"))

    assert hash_router._select_experts == hash_selection
    assert not hasattr(hash_router, "_anticipatory_cache")
    assert dynamic_router._select_experts is not dynamic_selection
    assert dynamic_router._anticipatory_cache is cache
    assert list(cache.replay_buffers) == [dynamic_router._anticipatory_key]
