# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host Engram sparse updates, clipping and checkpoint restoration."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torchtitan.components.optim import AdamW, LRSchedulersContainer

from torchtitan_npu.config import OptimConfig
from torchtitan_npu.extensions.components.gradient_clipping import GradientClippingOptim
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


def test_local_sparse_training_and_checkpoint_match_embedding():
    table = table_config().build()
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
    ref_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    for step, rows in enumerate(([0, 7, 7, 11], [1, 7, 1], [0, 7, 2])):
        ids = torch.tensor(rows)
        output = table._distributed_lookup(ids)
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
    table = table_config().build()
    table._init_self_buffers(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        table.weight.copy_(torch.arange(48).view(12, 4) / 16)
    initial_weight = table.weight.detach().clone()
    reference = torch.nn.Embedding.from_pretrained(initial_weight.clone(), freeze=False, sparse=True)
    model = torch.nn.Module()
    model.add_module("table", table)
    context = _parallelism_context()
    optim = OptimConfig(
        optimizer=host_mod.HostSparseOptimizersContainer.Config(
            optimizers=[host_mod.SparseAdam.Config(pattern=r"table\.weight$", lr=0.05)]
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0),
    ).build(model_parts=[model], parallelism_context=context, training_steps=2, pp_has_last_stage=True)
    assert isinstance(optim, GradientClippingOptim)
    optimizer = optim.optimizers
    ref_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05, betas=(0.9, 0.95), eps=1e-6)

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
        norm = optim.gradient_clipper.clip_grad_norm_(optim.parameters, 1.0, parallelism_context=context)
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
        optimizers=[host_mod.SparseAdam.Config(pattern=r"table\.weight$")],
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


def _parallelism_context():
    return SimpleNamespace(get_optional_mesh=lambda _name: None, pp_enabled=False, ep_enabled=False)


@pytest.mark.parametrize("overflow_source", ["dense", "sparse"])
def test_optimizer_aborts_before_nonfinite_host_gradient_update(overflow_source):
    table = _host_table_config().build()
    with torch.no_grad():
        table.weight.copy_(_ramp(table.num_embeddings) / 128)
    values = torch.zeros(1, _DIM)
    values[0, 0] = 3e32 if overflow_source == "sparse" else 4.0
    table.accumulate_sparse_gradient(torch.tensor([0]), values)
    model = torch.nn.Module()
    model.add_module("table", table)
    dense = torch.nn.Parameter(torch.ones(2))
    model.register_parameter("dense", dense)
    dense.grad = torch.tensor([3e32 if overflow_source == "dense" else 3.0, 1.0])
    original_table = table.weight.detach().clone()
    optim = OptimConfig(
        optimizer=host_mod.HostSparseOptimizersContainer.Config(
            optimizers=[
                host_mod.SparseAdam.Config(pattern=r"table\.weight$", lr=0.05),
                AdamW.Config(pattern=r".*", lr=0.1, fused=False),
            ]
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0),
    ).build(model_parts=[model], parallelism_context=_parallelism_context(), training_steps=2, pp_has_last_stage=True)
    assert isinstance(optim, GradientClippingOptim)

    with pytest.raises(RuntimeError, match=r"nonfinite .* gradient norm; optimizer update aborted"):
        optim.step(torch.tensor(1.0), current_step=1)

    assert torch.isfinite(dense.grad).all()
    torch.testing.assert_close(dense, torch.ones(2), rtol=0, atol=0)
    torch.testing.assert_close(table.weight, original_table, rtol=0, atol=0)
    torch.testing.assert_close(table.pending_sparse_grad().values(), values, rtol=0, atol=0)
    assert table.weight.grad is None


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
