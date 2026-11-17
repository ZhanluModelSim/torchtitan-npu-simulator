# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host Engram sparse updates, clipping and checkpoint restoration."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint, DefaultDeviceType

from torchtitan_npu.extensions.torchtitan.components.optimizer import optimizer as host_mod
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

pytestmark = pytest.mark.cpu
_DIM = 128


def table_config():
    return HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(11,),
        embedding_dim=4,
        num_embeddings=12,
        require_token_id_map=False,
        pin_memory=False,
    )


@pytest.mark.parametrize("recompute", [False, True], ids=["eager", "full_ac"])
def test_local_sparse_training_and_checkpoint_match_embedding(monkeypatch, recompute):
    # CPU-only inputs otherwise inherit the registered NPU checkpoint backend.
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    table = table_config().build()
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
    ref_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    for step, rows in enumerate(([0, 7, 7, 11], [1, 7, 1], [0, 7, 2])):
        ids = torch.tensor(rows)
        output = (
            checkpoint(table._distributed_lookup, ids, use_reentrant=False)
            if recompute
            else table._distributed_lookup(ids)
        )
        expected = reference(ids)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        grad = torch.arange(output.numel()).reshape_as(output).float() + 1
        output.backward(grad)
        expected.backward(grad)
        pending = table.pending_sparse_grad()
        assert table.weight.grad is None and pending.is_sparse
        torch.testing.assert_close(pending.to_dense(), reference.weight.grad.to_dense(), rtol=0, atol=0)
        table.prepare_sparse_optimizer_step()
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
        table.clear_sparse_gradient()
        ref_optimizer.zero_grad(set_to_none=True)
        if step == 1:
            weights, states = copy.deepcopy(table.state_dict()), copy.deepcopy(optimizer.state_dict())
            table = table_config().build()
            table.load_state_dict(weights)
            optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
            optimizer.load_state_dict(states)


def _ramp(rows):
    return torch.arange(rows * _DIM, dtype=torch.float32).reshape(rows, _DIM)


def _host_table_config():
    return HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(7,),
        embedding_dim=_DIM,
        num_embeddings=10,
        require_token_id_map=False,
        pin_memory=False,
    )


def test_host_offload_model_state_uses_rank_specific_shard_key():
    table = HostEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table._ep_rank = 1
    table._ep_size = 2
    table._mark_host_weight()

    state = table.state_dict()
    assert "weight" not in state
    assert list(state) == ["weight.ep_shard_00001_of_00002"]

    restored = HostEngramTable(_host_table_config())
    restored.weight = torch.nn.Parameter(torch.zeros(5, _DIM))
    restored._ep_rank = 1
    restored._ep_size = 2
    restored._mark_host_weight()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.weight, table.weight)


def test_trainer_clip_owns_only_its_tables(monkeypatch):
    from torchtitan.distributed import utils as dist_utils

    from torchtitan_npu.models.deepseek_v4_1.config_registry import DeepSeekV41Trainer

    table = HostEngramTable(_host_table_config())
    values = torch.zeros(1, _DIM)
    values[0, 0] = 4
    table.accumulate_sparse_gradient(torch.tensor([0]), values)
    model = torch.nn.Module()
    model.add_module("table", table)
    optimizer = object.__new__(host_mod.HostSparseOptimizersContainer)
    optimizer.model_parts = [model]
    trainer = object.__new__(DeepSeekV41Trainer)
    trainer.optimizers = optimizer
    other = object.__new__(DeepSeekV41Trainer)
    other.optimizers = object()

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)
    dense = torch.nn.Parameter(torch.zeros(1))
    dense.grad = torch.tensor([3.0])
    assert trainer.clip_grad_norm([dense], 1.0).item() == pytest.approx(5.0)
    assert dense.grad.item() == pytest.approx(0.6, rel=1e-5)
    assert table.pending_sparse_grad().values()[0, 0].item() == pytest.approx(0.8, rel=1e-5)
    dense.grad = torch.tensor([3.0])
    assert other.clip_grad_norm([dense], 1.0).item() == pytest.approx(3.0)
    assert dense.grad.item() == pytest.approx(1.0, rel=1e-5)
    assert dist_utils.clip_grad_norm_ is dense_clip


@pytest.mark.parametrize("overflow_source", ["dense", "sparse"])
def test_trainer_clip_aborts_before_nonfinite_correction(monkeypatch, overflow_source):
    from torchtitan.distributed import utils as dist_utils

    from torchtitan_npu.models.deepseek_v4_1.config_registry import DeepSeekV41Trainer

    table = HostEngramTable(_host_table_config())
    values = torch.zeros(1, _DIM)
    values[0, 0] = 3e32 if overflow_source == "sparse" else 4.0
    table.accumulate_sparse_gradient(torch.tensor([0]), values)
    model = torch.nn.Module()
    model.add_module("table", table)
    dense = torch.nn.Parameter(torch.ones(2))
    model.register_parameter("dense", dense)
    dense.grad = torch.tensor([3e32 if overflow_source == "dense" else 3.0, 1.0])
    optimizer = object.__new__(host_mod.HostSparseOptimizersContainer)
    optimizer.model_parts = [model]
    trainer = object.__new__(DeepSeekV41Trainer)
    trainer.optimizers = optimizer

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)

    with pytest.raises(RuntimeError, match="nonfinite .* gradient norm; optimizer update aborted"):
        trainer.clip_grad_norm(model.parameters(), 1.0)

    # Clipping may have scaled dense gradients, but must not introduce NaNs
    # or modify parameters/the pending sparse gradient before raising.
    assert torch.isfinite(dense.grad).all()
    torch.testing.assert_close(dense, torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(table.pending_sparse_grad().values(), values, rtol=0, atol=0)


@pytest.mark.parametrize("anticipatory", [False, True], ids=["standard", "anticipatory"])
def test_training_step_preserves_parent_context_and_sparse_clipping(monkeypatch, anticipatory):
    from torchtitan.components.optimizer import ParamGroupConfig
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.trainer import Trainer
    from torchtitan.models.common.linear import Linear

    from torchtitan_npu.extensions.experiment.anticipatory_routing.config import AnticipatoryRoutingConfig
    from torchtitan_npu.extensions.experiment.anticipatory_routing.router import AnticipatoryHashRouter
    from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import AnticipatorySchedule
    from torchtitan_npu.models.deepseek_v4_1.config_registry import DeepSeekV41Trainer
    from torchtitan_npu.models.deepseek_v4_1.indexer import FULL
    from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import AscSelector

    class RoutingModel(torch.nn.Module):
        """CPU fixture whose initialization forward visits the real router."""

        def forward(self, input_ids):
            return self.router((input_ids.float() + 1).unsqueeze(-1))[0]

    model = RoutingModel()
    model.add_module("table", table_config().build())
    model.register_parameter("dense", torch.nn.Parameter(torch.ones(1)))
    with torch.no_grad():
        model.table.weight.fill_(1)
    optimizer = host_mod.HostSparseOptimizersContainer.Config(
        lr=0.1, weight_decay=0.0, implementation="for-loop",
        param_groups=[
            ParamGroupConfig(pattern="table.weight", optimizer_name="SparseAdam"),
            ParamGroupConfig(pattern="dense", optimizer_name="AdamW"),
        ],
    ).build(model_parts=[model])
    trainer = object.__new__(DeepSeekV41Trainer)
    trainer.model_parts, trainer.optimizers = [model], optimizer
    model.selector = AscSelector.Config(
        mode=FULL, compress_ratio=1, num_index_heads=1, index_head_dim=4, index_topk=1,
    ).build()
    model.router = AnticipatoryHashRouter.Config(
        num_experts=2, top_k=1, gate=Linear.Config(in_features=1, out_features=2),
    ).build()
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    trainer._indexer_selectors = [model.selector]
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(disable_cuda_graphs=True, max_norm=1.0, steps=2),
        anticipatory=AnticipatoryRoutingConfig(enable=anticipatory),
        checkpoint=SimpleNamespace(interval=1, keep_latest_k=0),
    )
    trainer.parallel_dims = SimpleNamespace(
        dp_enabled=False, pp_enabled=False, ep_enabled=False, dp_cp_enabled=False,
        get_optional_mesh=lambda name: None,
    )
    trainer.device = torch.device("cpu")
    trainer.step = trainer.num_pipeline_parallel_microbatches = 1
    trainer.gradient_accumulation_steps = 2
    trainer.ntokens_seen = 1
    trainer.train_context = nullcontext

    def preprocess(inputs, labels):
        trainer.ntokens_seen += labels.numel()
        return inputs["input"], labels, {}

    trainer.post_dataloading_process = preprocess
    events, metrics = [], []
    trainer.lr_schedulers = SimpleNamespace(get_metrics=lambda: {}, step=lambda: events.append("lr"))
    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: events.append("staging"), save=lambda *a: True)
    trainer.metrics_processor = SimpleNamespace(
        should_log=lambda step: True,
        log=lambda *args, **kwargs: metrics.append(args),
        ntokens_since_last_log=0, data_loading_times=[],
    )
    trainer._sdc = SimpleNamespace(finalize_sdc_step=lambda: events.append("sdc"))

    class Loader:
        """Two fixed CPU batches for the schedule's initial training step."""

        def __iter__(self):
            yield {"input": torch.tensor([0])}, torch.tensor([-100])
            yield {"input": torch.tensor([0, 1])}, torch.tensor([1, 2])

        def state_dict(self):
            return {}

    trainer.dataloader = Loader()
    trainer.anticipatory_schedule = AnticipatorySchedule(trainer) if anticipatory else None

    def forward_backward(self, input_dict, **kwargs):
        events.append("forward_backward")
        # Both microbatches, including unsupervised positions, must be counted
        # before the first fused selector consumes the step denominator.
        assert model.selector.num_global_queries.item() == 3
        if anticipatory:
            assert trainer.anticipatory_schedule.engine.in_step
        loss = (3 * model.dense.sum() + 4 * model.table._distributed_lookup(input_dict["input"])[0, 0]) / 2
        loss.backward()
        return loss

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        events.append("clip")
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    # Keep the real parent training loop and TrainerEx hooks; only replace the
    # NPU forward/backward boundary and the distributed dense norm on CPU.
    monkeypatch.setattr(Trainer, "forward_backward_step", forward_backward)
    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)
    original_step = Trainer.train_step
    batches = trainer.anticipatory_schedule.data if anticipatory else iter(trainer.dataloader)
    trainer.train_step(batches)

    expected_events = ["forward_backward", "sdc"] * 2 + ["clip", "staging", "lr"]
    if anticipatory:
        engine = trainer.anticipatory_schedule.engine
        assert engine.last_global_loss == 7.0 and not engine.in_step and not engine.failed
        assert trainer.metrics_processor.ntokens_since_last_log == 3
        assert trainer.ntokens_seen == 1  # Initialization must not credit training tokens.
        buffer = model.router._anticipatory_replay_buffer
        torch.testing.assert_close(buffer.indices, torch.zeros(1, 1, dtype=torch.int64))
    assert events == expected_events
    assert metrics == [(1, 7.0, 7.0, 5.0)]
    torch.testing.assert_close(model.dense.grad, torch.tensor([3 / (5 + 1e-6)]))
    sparse = next(opt for opt in optimizer.optimizers if isinstance(opt, torch.optim.SparseAdam))
    expected_moment = torch.zeros_like(model.table.weight)
    expected_moment[0, 0] = (1 - sparse.param_groups[0]["betas"][0]) * 4 / (5 + 1e-6)
    torch.testing.assert_close(sparse.state[model.table.weight]["exp_avg"], expected_moment)
    assert model.table.weight[0, 0] < 1 and model.dense.item() < 1
    torch.testing.assert_close(model.table.weight[1:], torch.ones_like(model.table.weight[1:]))
    assert model.table.pending_sparse_grad() is None
    assert Trainer.train_step is original_step
    assert Trainer.train_step.__globals__["dist_utils"] is dist_utils
    assert dist_utils.clip_grad_norm_ is dense_clip
