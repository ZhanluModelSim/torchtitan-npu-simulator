# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU adapter checks; the fake buffer does not exercise pinned memory or HCCL."""

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint, DefaultDeviceType
from torchtitan.config import Configurable, OverrideConfig, TrainingConfig, apply_overrides
from torchtitan.components.optimizer import ParamGroupConfig

from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable, _HAS_OPAQUE_OBJECT_API
from torchtitan_npu.models.deepseek_v4_1.parallelize import _shard_engram_tables
from torchtitan_npu.override.deepseek_v4_1.engram import ascendc, mxfp8

pytestmark = pytest.mark.cpu


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        table: HostEngramTable.Config


class _FakeBuffer:
    """Public buffer boundary with local CPU rows, without cross-rank routing."""

    def __init__(self, group, **kwargs):
        self.group = group
        self.kwargs = kwargs
        self.writes = 0

    @staticmethod
    def get_engram_storage_size_hint(rows, width, dtype):
        return rows * width * dtype.itemsize

    def engram_write(self, storage, scale=None):
        self.storage = storage
        self.scale = scale
        self.writes += 1

    def engram_fetch(self, indices):
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        fetched = self.storage.index_select(0, indices.long())
        # This local fake models the tests' two-rank EngramFetchCtx layout.
        rank_size = 2
        entries = torch.zeros(self.kwargs["num_max_tokens_per_rank"] * rank_size, dtype=torch.int32)
        entries[:indices.numel()] = indices
        context = SimpleNamespace(
            perm=torch.arange(indices.numel(), dtype=torch.int32),
            send_counts=torch.zeros(rank_size * 8, dtype=torch.int32),  # 32-byte slots per rank.
            recv_counts=torch.zeros(rank_size, dtype=torch.int32),
            recv_local_entry=entries,
            num_recv=torch.tensor([indices.numel()], dtype=torch.int32),
        )
        if self.scale is not None:
            fetched_scale = self.scale.index_select(0, indices.long())
            return lambda: (fetched, fetched_scale, context)
        return lambda: (fetched, context)

    def engram_fetch_grad(self, grad, context):
        assert grad.dtype == torch.float32 and grad.is_contiguous()
        indices = context.recv_local_entry[:int(context.num_recv.item())]
        unique, inverse = indices.unique(return_inverse=True)
        reduced = torch.zeros(unique.numel(), grad.shape[1])
        reduced.index_add_(0, inverse, grad)
        return reduced, unique


@pytest.mark.parametrize("recompute", [False, True], ids=["eager", "full_ac"])
def test_override_sparse_update_reuses_registered_storage(monkeypatch, recompute):
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    root = _Root.Config(table=HostEngramTable.Config(
        vocab_size=16, layer_id=0, ngram_orders=(2,), num_heads=1,
        head_vocab_sizes=(11,), embedding_dim=128, num_embeddings=12,
        require_token_id_map=False, param_init={"weight": torch.nn.init.ones_},
    ))
    apply_overrides(OverrideConfig(imports=[(
        "torchtitan_npu.override.deepseek_v4_1.engram.host_offload",
        {"num_max_tokens_per_rank": 4},
    )]), root)
    table = root.table.build()
    assert isinstance(table, ascendc.HostOffloadEngramTable) and table.pin_memory
    # Simulate an already-sharded CPU weight; actual EP/HCCL needs NPU validation.
    table.weight = torch.nn.Parameter(torch.zeros(6, 128))
    table._ep_size = 2
    table.ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "ep")
    monkeypatch.setattr(ascendc, "_dedicated_engram_group", lambda group: "engram")
    monkeypatch.setattr(table, "_elastic_buffer_type", lambda: _FakeBuffer)
    model = torch.nn.Module()
    model.add_module("table", table)
    ignored = _shard_engram_tables(
        model, edp_mesh=SimpleNamespace(size=lambda: 1), edp_mesh_dims=None,
        training=TrainingConfig(mixed_precision_param="bfloat16"),
    )
    assert ignored == {table.weight}
    table.init_states(buffer_device=torch.device("cpu"))
    buffer = table._elastic_buffer
    assert buffer.group == "engram" and buffer.kwargs["with_grad"]
    assert buffer.kwargs["num_cpu_bytes"] == 6 * 128 * 4
    assert buffer.kwargs["num_max_tokens_per_rank"] == 4
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = torch.optim.SparseAdam([table.weight], lr=0.05)
    expected_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    for rows in ([0, 3, 3, 5], [1, 3, 1]):
        ids = torch.tensor(rows)
        actual = checkpoint(table._distributed_lookup, ids, use_reentrant=False) if recompute else table._distributed_lookup(ids)
        expected = reference(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad = torch.arange(actual.numel()).reshape_as(actual).float() / 128
        actual.backward(grad)
        expected.backward(grad)
        assert table.weight.grad is None
        torch.testing.assert_close(table.pending_sparse_grad().to_dense(), reference.weight.grad.to_dense())
        table.prepare_sparse_optimizer_step()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
        table.clear_sparse_gradient()
        expected_optimizer.zero_grad()
    assert buffer.writes == 1
    assert buffer.storage.data_ptr() == table.weight.data_ptr()
    with pytest.raises(ValueError, match="capacity"):
        table._distributed_lookup(torch.zeros(5, dtype=torch.long))
    table.weight = torch.nn.Parameter(table.weight.detach().clone())
    with pytest.raises(RuntimeError, match="storage was replaced"):
        table._distributed_lookup(torch.tensor([0]))


def test_mxfp8_override_refreshes_derived_storage(monkeypatch):
    root = _Root.Config(table=HostEngramTable.Config(
        vocab_size=16, layer_id=0, ngram_orders=(2,), num_heads=1,
        head_vocab_sizes=(11,), embedding_dim=128, num_embeddings=12,
        require_token_id_map=False, param_init={"weight": torch.nn.init.ones_},
    ))
    apply_overrides(OverrideConfig(imports=[(
        "torchtitan_npu.override.deepseek_v4_1.engram.host_offload_mxfp8",
        {"num_max_tokens_per_rank": 4, "quantization_chunk_rows": 2},
    )]), root)
    table = root.table.build()
    assert isinstance(table, mxfp8.MXFP8HostOffloadEngramTable)
    table.weight = torch.nn.Parameter(torch.zeros(6, 128))
    table._ep_size = 2
    table.ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "ep")
    monkeypatch.setattr(ascendc, "_dedicated_engram_group", lambda group: "engram")
    monkeypatch.setattr(table, "_elastic_buffer_type", lambda: _FakeBuffer)
    table.init_elastic_buffer(param_dtype=torch.bfloat16)
    table.init_states(buffer_device=torch.device("cpu"))

    storage, scale = table._require_quantized_storage()
    assert storage.dtype == torch.float8_e4m3fn
    assert scale.shape == (6, 4) and scale.dtype == torch.float8_e8m0fnu
    assert table._elastic_buffer.kwargs["num_cpu_bytes"] == 6 * 128
    ids = torch.tensor([0, 3, 3, 5])
    actual = table._distributed_lookup(ids)
    expected = mxfp8._dequantize_mxfp8_rows(storage[ids], scale[ids])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.backward(torch.arange(actual.numel()).reshape_as(actual).bfloat16() / 128)
    untouched_storage = storage[1].clone()
    untouched_scale = scale[1].clone()
    container = HostSparseOptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="SparseAdam", optimizer_kwargs={"lr": 0.05})],
    ).build(model_parts=[table])
    container.step()
    assert table.pending_sparse_grad() is None
    expected_storage, expected_scale = mxfp8._quantize_mxfp8_rows(table.weight)
    torch.testing.assert_close(storage.float(), expected_storage.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale.float(), expected_scale.float(), rtol=0, atol=0)
    torch.testing.assert_close(storage[1].float(), untouched_storage.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale[1].float(), untouched_scale.float(), rtol=0, atol=0)

    loaded = {name: torch.full_like(value, 0.125) for name, value in table.state_dict().items()}
    table.load_state_dict(loaded)
    expected_storage, expected_scale = mxfp8._quantize_mxfp8_rows(table.weight)
    torch.testing.assert_close(storage.float(), expected_storage.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale.float(), expected_scale.float(), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["torch", "fp32", "mxfp8"])
def test_joint_sharding_override_sets_storage_and_lookup_group(monkeypatch, backend):
    from torchtitan_npu.models.deepseek_v4_1.engram import host

    ep_group, joint_group = object(), object()
    ep = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1, get_group=lambda: ep_group)
    joint = SimpleNamespace(size=lambda: 4, get_local_rank=lambda: 3, get_group=lambda: joint_group)
    monkeypatch.setattr(host, "_joint_table_mesh", lambda dims: joint)
    root = _Root.Config(table=HostEngramTable.Config(
        vocab_size=16, layer_id=0, ngram_orders=(2,), num_heads=1,
        head_vocab_sizes=(11,), embedding_dim=128, num_embeddings=12,
        require_token_id_map=False,
    ))
    if backend == "torch":
        name, kwargs = "shard_over_efsdp", {}
    else:
        name = "host_offload" if backend == "fp32" else "host_offload_mxfp8"
        kwargs = {"num_max_tokens_per_rank": 4, "shard_over_efsdp": True}
    apply_overrides(OverrideConfig(imports=[(
        f"torchtitan_npu.override.deepseek_v4_1.engram.{name}", kwargs,
    )]), root)
    table = root.table.build()
    table.parallelize(SimpleNamespace(get_optional_mesh=lambda name: ep))
    assert table.weight.shape == (3, 128)
    assert table.ep_mesh is ep and table.lookup_mesh is joint
    assert list(table.state_dict()) == ["weight.efsdp_ep_shard_00003_of_00004"]
    if backend != "torch":
        monkeypatch.setattr(ascendc, "_dedicated_engram_group", lambda group: group)
        monkeypatch.setattr(table, "_elastic_buffer_type", lambda: _FakeBuffer)
        table.init_elastic_buffer(param_dtype=torch.float32)
        assert table._elastic_buffer.group is joint_group
        assert table._elastic_buffer_spec[:2] == (3, 128)


@pytest.fixture
def compiler_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.skipif(not _HAS_OPAQUE_OBJECT_API, reason="Compiled Host Engram requires the PyTorch opaque-object API")
@pytest.mark.parametrize("backend", ["fp32", "mxfp8"])
def test_compiled_lookup_preserves_sparse_updates(monkeypatch, compiler_cache, backend):
    """Real compiler/autograd with the CANN boundary replaced by local CPU rows."""
    from torch._dynamo.backends.registry import lookup_backend

    # Match the V4.1 compile assembly's treatment of effectful ops under AC.
    monkeypatch.setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    elastic_module = types.ModuleType("cann_ops_transformer.ops.mc2.common.elastic_buffer")
    elastic_module.EngramFetchCtx = SimpleNamespace
    monkeypatch.setitem(sys.modules, elastic_module.__name__, elastic_module)
    monkeypatch.setattr(ascendc, "_dedicated_engram_group", lambda group: group)
    monkeypatch.setattr(ascendc.HostOffloadEngramTable, "_elastic_buffer_type", staticmethod(lambda: _FakeBuffer))
    root = _Root.Config(table=HostEngramTable.Config(
        vocab_size=16, layer_id=0, ngram_orders=(2,), num_heads=1,
        head_vocab_sizes=(11,), embedding_dim=128, num_embeddings=12,
        require_token_id_map=False, param_init={"weight": torch.nn.init.ones_},
    ))
    factory = "host_offload" if backend == "fp32" else "host_offload_mxfp8"
    apply_overrides(OverrideConfig(imports=[(
        f"torchtitan_npu.override.deepseek_v4_1.engram.{factory}", {"num_max_tokens_per_rank": 4},
    )]), root)
    table = root.table.build()
    table.weight = torch.nn.Parameter(torch.ones(6, 128))
    table._ep_size = 2
    table.ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "ep")
    table.init_elastic_buffer(param_dtype=torch.float32)
    table.init_states(buffer_device=torch.device("cpu"))
    with torch.no_grad():
        table.weight.copy_(torch.arange(6 * 128).reshape(6, 128) / 1024 + 0.25)
        table.refresh_lookup_storage(torch.arange(6))
    reference = torch.nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False, sparse=True)
    optimizer = HostSparseOptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[ParamGroupConfig(pattern=".*", optimizer_name="SparseAdam", optimizer_kwargs={"lr": 0.05})],
    ).build(model_parts=[table])
    reference_optimizer = torch.optim.SparseAdam(reference.parameters(), lr=0.05)
    gain = torch.nn.Parameter(torch.tensor(0.5))
    reference_gain = torch.nn.Parameter(gain.detach().clone())
    graphs = []

    def backend_fn(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    def block(ids):
        return table._distributed_lookup(ids).float() * gain

    def recomputed_block(ids):
        return checkpoint(block, ids, use_reentrant=False)

    for recompute in (False, True):
        compiled = torch.compile(recomputed_block if recompute else block, backend=backend_fn, fullgraph=True)
        for step in range(2):
            ids_a = torch.tensor([0, 3, 3, 5]) if step == 0 else torch.tensor([2, 1, 2, 4])
            ids_b = torch.tensor([1, 1, 5, 0])
            actual_a, actual_b = compiled(ids_a), compiled(ids_b)

            def reference_rows(ids):
                rows = reference(ids)
                if backend == "fp32":
                    return rows
                # Independent arithmetic oracle; straight-through gradient to master rows.
                blocks = rows.detach().unflatten(-1, (-1, 32))
                scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1) / 448)))
                values = (blocks / scale[..., None]).to(torch.float8_e4m3fn).float()
                decoded = (values * scale[..., None]).flatten(-2).bfloat16().float()
                return rows + (decoded - rows).detach()

            expected_a = reference_rows(ids_a) * reference_gain
            expected_b = reference_rows(ids_b) * reference_gain
            torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
            torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
            # Two outstanding forwards, reversed backward order and accumulation.
            actual_b.sum().backward()
            actual_a.sum().backward()
            expected_b.sum().backward()
            expected_a.sum().backward()
            torch.testing.assert_close(gain.grad, reference_gain.grad, rtol=1e-6, atol=1e-6)
            assert table.weight.grad is None
            torch.testing.assert_close(table.pending_sparse_grad().to_dense(), reference.weight.grad.to_dense(), rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
            assert table.pending_sparse_grad() is None
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            gain.grad = reference_gain.grad = None
    assert graphs and all(
        any("ascendc_engram_lookup" in str(module.graph) for module in graph.modules() if isinstance(module, torch.fx.GraphModule))
        for graph in graphs
    )
    assert table._elastic_buffer.writes == 1
