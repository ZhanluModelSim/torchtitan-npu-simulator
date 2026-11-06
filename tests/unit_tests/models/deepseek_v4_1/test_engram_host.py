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
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from torchtitan.components.optim import AdamW

from torchtitan_npu.extensions.torchtitan.components.optimizer import optimizer as host_mod
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import SparseAdam
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


def test_token_forward_microbatch_clipping_and_sparse_adam_match_embedding():
    from torchtitan.components.optim import AdamW

    table = table_config().build()
    table._init_self_buffers(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    initial_weight = table.weight.detach().clone()
    reference = torch.nn.Embedding.from_pretrained(initial_weight.clone(), freeze=False, sparse=True)
    model = torch.nn.Module()
    model.add_module("table", table)
    optimizer = host_mod.HostSparseOptimizersContainer.Config(
        lr=0.05,
        weight_decay=0.0,
        optimizers=[
            SparseAdam.Config(pattern="table.weight", betas=(0.9, 0.999), eps=1e-08, lr=0.05),
        ],
    ).build(model_parts=[model])
    ref_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)

    for step in range(2):
        for microbatch, tokens in enumerate(([4, 4, 4, 5], [4, 5, 4, 4])):
            ids, positions = torch.tensor(tokens), torch.arange(4)
            row_ids = table.hash(ids, positions)
            output = table(ids, positions)
            expected = reference(row_ids).flatten(1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            upstream = torch.arange(output.numel()).reshape_as(output).float() / 8 + step + microbatch
            output.backward(upstream)
            expected.backward(upstream)
            pending = table.pending_sparse_grad()
            ref_grad = reference.weight.grad.coalesce()
            assert table.weight.grad is None
            torch.testing.assert_close(pending.indices(), ref_grad.indices(), rtol=0, atol=0)
            torch.testing.assert_close(pending.values(), ref_grad.values(), rtol=0, atol=0)

        expected_norm = ref_grad.values().square().sum().sqrt()
        expected_scale = torch.clamp(1.0 / (expected_norm + 1e-6), max=1.0)
        norm = host_mod.clip_grad_norm_with_host_sparse_tables(
            [],
            1.0,
            tables=[table],
            clipper=torch.nn.utils.clip_grad_norm_,
        )
        reference.weight.grad = torch.sparse_coo_tensor(
            ref_grad.indices(),
            ref_grad.values() * expected_scale,
            ref_grad.shape,
        ).coalesce()
        torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(
            table.pending_sparse_grad().values(),
            reference.weight.grad.values(),
            rtol=0,
            atol=0,
        )
        optimizer.step()
        ref_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
        actual_state = optimizer.optimizers[0].state[table.weight]
        expected_state = ref_optimizer.state[reference.weight]
        assert actual_state["step"] == expected_state["step"] == step + 1
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
        untouched = torch.ones(table.num_embeddings, dtype=torch.bool)
        untouched[ref_grad.indices()[0]] = False
        torch.testing.assert_close(table.weight[untouched], initial_weight[untouched], rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        assert table.pending_sparse_grad() is None and table.weight.grad is None


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


@pytest.mark.skipif(
    not hasattr(torch.distributed.checkpoint, "CheckpointableTensor"), reason="CheckpointableTensor is required"
)
def test_host_offload_weight_and_moments_describe_global_row_shard():
    table = HostEngramTable(_host_table_config())
    table.weight = torch.nn.Parameter(_ramp(5))
    table._ep_rank = 1
    table._ep_size = 2
    table._mark_host_weight()

    state = table.state_dict()
    assert list(state) == ["weight"]
    assert state["weight"].global_shape == (10, _DIM)
    assert state["weight"].global_offsets == ((5, 0),)
    assert state["weight"].local_offsets == ((0, 0),)
    assert state["weight"].local_sizes == ((5, _DIM),)

    model = torch.nn.Module()
    model.add_module("table", table)
    optimizer = host_mod.HostSparseOptimizersContainer.Config(
        optimizers=[SparseAdam.Config(pattern="table\\.weight$", betas=(0.9, 0.999), eps=1e-08)],
    ).build(model_parts=[model])

    optim_state = optimizer.state_dict()
    for name in ("exp_avg", "exp_avg_sq"):
        moment = optim_state[f"state.table.weight.{name}"]
        assert moment.global_shape == (10, _DIM)
        assert moment.global_offsets == ((5, 0),)
        assert moment.local_offsets == ((0, 0),)
        assert moment.local_sizes == ((5, _DIM),)
        assert moment.data_ptr() == optimizer.optimizers[0].state[table.weight][name].data_ptr()

    restored = HostEngramTable(_host_table_config())
    restored.weight = torch.nn.Parameter(torch.zeros(5, _DIM))
    restored._ep_rank = 1
    restored._ep_size = 2
    restored._mark_host_weight()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.weight, table.weight)


def test_container_clip_owns_only_its_tables(monkeypatch):
    from torchtitan.distributed import utils as dist_utils

    table = HostEngramTable(_host_table_config())
    values = torch.zeros(1, _DIM)
    values[0, 0] = 4
    table.accumulate_sparse_gradient(torch.tensor([0]), values)
    model = torch.nn.Module()
    model.add_module("table", table)
    optimizer = object.__new__(host_mod.HostSparseOptimizersContainer)
    optimizer.model_parts = [model]

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)
    dense = torch.nn.Parameter(torch.zeros(1))
    dense.grad = torch.tensor([3.0])
    assert optimizer.clip_grad_norm([dense], 1.0).item() == pytest.approx(5.0)
    assert dense.grad.item() == pytest.approx(0.6, rel=1e-5)
    assert table.pending_sparse_grad().values()[0, 0].item() == pytest.approx(0.8, rel=1e-5)
    dense.grad = torch.tensor([3.0])
    assert dense_clip([dense], 1.0).item() == pytest.approx(3.0)
    assert dense.grad.item() == pytest.approx(1.0, rel=1e-5)
    assert dist_utils.clip_grad_norm_ is dense_clip


@pytest.mark.parametrize("overflow_source", ["dense", "sparse"])
def test_container_clip_aborts_before_nonfinite_correction(monkeypatch, overflow_source):
    from torchtitan.distributed import utils as dist_utils

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

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)

    with pytest.raises(RuntimeError, match=r"nonfinite .* gradient norm; optimizer update aborted"):
        optimizer.clip_grad_norm(model.parameters(), 1.0)

    # Clipping may have scaled dense gradients, but must not introduce NaNs
    # or modify parameters/the pending sparse gradient before raising.
    assert torch.isfinite(dense.grad).all()
    torch.testing.assert_close(dense, torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(table.pending_sparse_grad().values(), values, rtol=0, atol=0)


@pytest.mark.parametrize("anticipatory", [False, True], ids=["standard", "anticipatory-hooks"])
def test_engine_accumulation_preserves_sparse_clipping_and_microbatch_hooks(monkeypatch, anticipatory):
    from functools import partial

    from torchtitan.components.data import TokenizedTrainingMicrobatch
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.training_engine import TrainingEngine

    from torchtitan_npu.models.deepseek_v4_1.config_registry import (
        DeepSeekV41TrainingEngine,
        deepseek_v4_1_debugmodel_text,
    )
    from torchtitan_npu.models.deepseek_v4_1.indexer import FULL
    from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import AscSelector

    class SparseModel(torch.nn.Module):
        """A CPU loss fixture with real Host Engram autograd and sparse state."""

        def preprocess_inputs(self, batch, **kwargs):
            return batch["input"], batch["labels"], {}

        def forward(self, input_ids):
            return (3 * self.dense.sum() + 4 * self.table._distributed_lookup(input_ids)[0, 0]) / 2

    model = SparseModel()
    model.add_module("table", table_config().build())
    model.register_parameter("dense", torch.nn.Parameter(torch.ones(1)))
    with torch.no_grad():
        model.table.weight.fill_(1)
    optimizer_config = host_mod.HostSparseOptimizersContainer.Config(
        optimizers=[
            SparseAdam.Config(pattern="table.weight", lr=0.1, betas=(0.9, 0.999), eps=1e-8),
            AdamW.Config(pattern="dense", lr=0.1, weight_decay=0, fused=False, foreach=False),
        ],
    )
    model.selector = AscSelector.Config(
        mode=FULL,
        compress_ratio=1,
        num_index_heads=1,
        index_head_dim=4,
        index_topk=1,
    ).build()
    engine = object.__new__(DeepSeekV41TrainingEngine)
    engine.model_parts = [model]
    engine.device = torch.device("cpu")
    engine.parallelism_context = SimpleNamespace(
        cp=1,
        dp_enabled=False,
        pp_enabled=False,
        ep_enabled=False,
        dp_replicate_enabled=False,
        get_optional_mesh=lambda name: None,
        activate_spmd=lambda **kwargs: nullcontext(),
    )
    engine.config = SimpleNamespace(
        training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=2, max_context_length=2),
        parallelism=SimpleNamespace(),
        debug=SimpleNamespace(spmd_typechecking=False),
    )
    engine.num_completed_steps = engine.ntokens_seen = 0
    engine.max_num_documents = None
    engine.preprocess_inputs_kwargs = {}
    engine.sdc_replayer = None
    optim_config = deepseek_v4_1_debugmodel_text().optim
    optim_config.optimizer = optimizer_config
    optim_config.max_norm = 1.0
    engine.optim = optim_config.build(
        model_parts=[model],
        parallelism_context=engine.parallelism_context,
        training_steps=2,
        pp_has_last_stage=True,
    )
    optimizer = engine.optim.optimizers
    engine.gc_handler = SimpleNamespace(run=lambda step: None)
    engine._run_forward_backward = partial(engine._forward_backward_body, defer_fsdp_gradient_reduction=False)
    events = []
    engine._sdc = SimpleNamespace(finalize_sdc_step=lambda: events.append("sdc"))
    engine.anticipatory_schedule = (
        SimpleNamespace(
            prepare_microbatch=lambda: events.append("prepare"),
            accumulate_microbatch_loss=lambda loss: events.append(("loss", loss.item())),
        )
        if anticipatory
        else None
    )

    def loss_fn(output, labels, **kwargs):
        assert model.selector.num_global_queries.item() == 3
        assert kwargs["global_valid_tokens"].item() == 2
        return output, {}

    engine.loss_fn = loss_fn
    groups = [
        [
            TokenizedTrainingMicrobatch(
                input=torch.tensor(tokens),
                labels=torch.tensor(labels),
                positions=torch.arange(len(tokens)),
                padding_mask=torch.ones(len(tokens), dtype=torch.bool),
                num_valid_tokens=valid,
            )
        ]
        for tokens, labels, valid in (([0], [-100], 0), ([0, 1], [1, 2], 2))
    ]
    result = engine.forward_backward(microbatch_groups=groups, global_valid_tokens=2)
    assert result.loss.item() == 7.0
    assert engine.loss is result.loss
    assert engine.num_accumulation_steps == 2 and engine.ntokens_seen == 4
    assert events == (["prepare", ("loss", 3.5), "sdc"] * 2 if anticipatory else ["sdc"] * 2)
    assert (
        DeepSeekV41TrainingEngine._non_pp_forward_backward_microbatch
        is not TrainingEngine._non_pp_forward_backward_microbatch
    )

    def dense_clip(parameters, max_norm, norm_type=2.0, **kwargs):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, foreach=False)

    monkeypatch.setattr(dist_utils, "clip_grad_norm_", dense_clip)
    assert engine.optim_step().item() == pytest.approx(5.0)
    assert engine.num_completed_steps == 1
    torch.testing.assert_close(model.dense.grad, torch.tensor([3 / (5 + 1e-6)]))
    sparse = next(opt for opt in optimizer.optimizers if isinstance(opt, torch.optim.SparseAdam))
    expected_moment = torch.zeros_like(model.table.weight)
    expected_moment[0, 0] = (1 - sparse.param_groups[0]["betas"][0]) * 4 / (5 + 1e-6)
    torch.testing.assert_close(sparse.state[model.table.weight]["exp_avg"], expected_moment)
    assert model.table.weight[0, 0] < 1 and model.dense.item() < 1
    torch.testing.assert_close(model.table.weight[1:], torch.ones_like(model.table.weight[1:]))
    assert model.table.pending_sparse_grad() is None


def test_lora_host_table_accepts_folded_parent_keywords_without_frozen_autograd():
    table = HostEngramTable.Config(
        vocab_size=16,
        layer_id=0,
        ngram_orders=(2,),
        num_heads=1,
        head_vocab_sizes=(11,),
        embedding_dim=4,
        num_embeddings=12,
        require_token_id_map=False,
        pin_memory=False,
    ).build()
    table._init_self_buffers(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    table.weight.requires_grad_(False)
    tokens, positions = torch.tensor([3, 4, 3, 4]), torch.arange(4)

    output = table(input_ids_T=tokens, positions_T=positions)

    expected = table.weight[table.hash(tokens, positions)].flatten(1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not output.requires_grad


@pytest.mark.parametrize("nonfinite_loss", [float("nan"), float("inf")], ids=["nan", "infinity"])
def test_host_sparse_optim_rejects_nonfinite_loss_before_parameter_update(nonfinite_loss):
    model = torch.nn.Module()
    model.table = table_config().build()
    model.dense = torch.nn.Parameter(torch.ones(1))
    with torch.no_grad():
        model.table.weight.fill_(1)
    optimizer_config = host_mod.HostSparseOptimizersContainer.Config(
        optimizers=[
            SparseAdam.Config(pattern="table.weight", lr=0.1),
            AdamW.Config(pattern="dense", lr=0.1, fused=False, foreach=False),
        ],
    )
    context = SimpleNamespace(pp_enabled=False, ep_enabled=False, get_optional_mesh=lambda name: None)
    optim = host_mod.HostSparseOptim.Config(optimizer=optimizer_config).build(
        model_parts=[model],
        parallelism_context=context,
        training_steps=2,
        pp_has_last_stage=True,
    )
    (model.dense.sum() + model.table._distributed_lookup(torch.tensor([0])).sum()).backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with pytest.raises(RuntimeError, match="Loss or gradient norm is not finite"):
        optim.step(torch.tensor(nonfinite_loss), current_step=1)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert all(not optimizer.state for optimizer in optim.optimizers.optimizers)
