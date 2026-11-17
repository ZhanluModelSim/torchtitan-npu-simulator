# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU oracle at the CANN boundary, with real configs and document metadata."""

import copy
import importlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import _debugmodel
from torchtitan_npu.models.deepseek_v4.compressor import Compressor, CompressorImplementation
from torchtitan_npu.models.deepseek_v4.metadata import build_compressed_varlen_metadata
from torchtitan_npu.override.deepseek_v4.compressor import ascendc


@pytest.mark.parametrize("reverse", [False, True], ids=["compressor-first", "compressor-last"])
def test_compressor_override_composes_with_norm_and_rope(monkeypatch, reverse):
    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.rope import AscComplexRoPE

    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    cfg = _debugmodel()
    imports = [
        "torchtitan_npu.override.deepseek_v4.compressor.asc",
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_complex",
    ]
    apply_overrides(OverrideConfig(imports=imports[::-1] if reverse else imports), cfg)
    compressors = [node for _, node, _, _ in cfg.traverse(Compressor.Config)]
    assert compressors
    for node in compressors:
        assert type(node) is Compressor.Config
        assert isinstance(node.implementation, ascendc.AscCompressor.Config)
        assert isinstance(node.norm, AscRMSNorm.Config)
        assert isinstance(node.rope, AscComplexRoPE.Config)
    assert all(
        type(node.implementation) is CompressorImplementation.Config
        for _, node, _, _ in _debugmodel().traverse(Compressor.Config)
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("batch_chunk", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_compressor_metadata_selection(dsv4, monkeypatch, fused, batch_chunk, reverse):
    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    cfg = _debugmodel()
    metadata_target = (
        "torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_metadata.asc_metadata"
        if batch_chunk else "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata"
    )
    imports = [metadata_target, "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata"]
    if fused:
        imports.extend([
            "torchtitan_npu.override.deepseek_v4.compressor.asc_metadata",
            "torchtitan_npu.override.deepseek_v4.compressor.asc",
        ])
    apply_overrides(OverrideConfig(imports=imports[::-1] if reverse else imports), cfg)
    provider = cfg.metadata_extension.build()
    li_provider = cfg.lightning_indexer_metadata.build()
    captured = []
    if batch_chunk:
        from torchtitan_npu.extensions.experiment.cv_parallel import batch_chunk_metadata as cv

        provider._chunk_li_metadata = li_provider
        monkeypatch.setattr(cv, "_tag_batch_chunk_metadata", lambda full, chunks: captured.extend((full, *chunks)))
    boundaries = torch.tensor([0, 8, 16], dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=8, max_k=8), (4,)
    )
    metadata = li_provider(metadata)
    assert metadata.plans[4].li_metadata is not None
    result = provider(metadata)
    tables = [item.plans[4].compressor_state_block_table for item in (captured or [result])]
    assert all((table is not None) == fused for table in tables)
    if fused:
        table, *_ = ascendc._state_inputs(
            boundaries, torch.empty(16, 8), max_length=8, ratio=4, head_dim=8,
            state_block_table=result.plans[4].compressor_state_block_table,
        )
        assert table is result.plans[4].compressor_state_block_table


def _document_pool(x, wkv, wgate, ape, cu, ratio):
    outputs = []
    width = wkv.shape[0] // (2 if ratio == 4 else 1)
    for start, end in zip(cu[:-1], cu[1:]):
        values = F.linear(x[start:end].float(), wkv.float())
        scores = F.linear(x[start:end].float(), wgate.float())
        for offset in range(0, (end - start) // ratio * ratio, ratio):
            current_values = values[offset:offset + ratio, -width:]
            current_scores = scores[offset:offset + ratio, -width:] + ape[:, -width:]
            if ratio == 4 and offset:
                current_values = torch.cat((values[offset - ratio:offset, :width], current_values))
                current_scores = torch.cat((scores[offset - ratio:offset, :width] + ape[:, :width], current_scores))
            outputs.append((current_values * current_scores.softmax(0)).sum(0))
    return torch.stack(outputs)


def _postprocess(module, pooled, positions):
    normalized = F.rms_norm(pooled, (module.head_dim,), module.norm.weight, module.norm.eps)
    return module.rope(normalized[None, :, None, :], positions=positions[None, :])[0, :, 0, :]


@pytest.fixture
def compressor_factory(monkeypatch):
    registry = importlib.import_module("torchtitan.config.override")
    with monkeypatch.context() as scoped, torch.random.fork_rng():
        scoped.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
        fake_module = types.ModuleType("compressor")
        fake_module.compressor = lambda *args, **kwargs: pytest.fail("unexpected CANN compressor call")
        scoped.setitem(sys.modules, "cann_ops_transformer.ops.compressor", fake_module)
        torch.manual_seed(93)

        def build(ratio, indexer=False):
            cfg = _debugmodel()
            apply_overrides(OverrideConfig(imports=["torchtitan_npu.override.deepseek_v4.compressor.asc"]), cfg)
            attention = cfg.layers[2 if ratio == 4 else 3].attention
            compressor_cfg = attention.indexer.compressor if indexer else attention.compressor
            compressor_cfg.wkv.in_features = 1024
            compressor_cfg.wgate.in_features = 1024
            module = compressor_cfg.build()
            assert isinstance(module.implementation, ascendc.AscCompressor)
            module.init_states(buffer_device=torch.device("cpu"))
            module.wkv.to(torch.float16)
            module.wgate.to(torch.float16)
            module.norm.to(torch.float16)
            return module

        try:
            yield build
        finally:
            torch._dynamo.reset()


def test_prebuilt_compressor_tables_follow_current_boundaries():
    widths = []
    for lengths in ((16, 16), (8, 24), (8, 8, 16)):
        boundaries = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
        metadata = build_compressed_varlen_metadata(
            VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=16, max_k=16), (1, 4, 128)
        )
        # The current boundaries, not a possibly stale host maximum, size the table.
        ascendc.build_compressor_state_tables(metadata)
        assert metadata.plans[1].compressor_state_block_table is None
        for ratio, block_size in ((4, 8), (128, 16)):
            table = metadata.plans[ratio].compressor_state_block_table
            assert table.shape == (len(lengths), (max(lengths) + block_size - 1) // block_size)
            assert table.dtype == torch.int32 and table.is_contiguous()
            assert not table.any()
            assert getattr(table, "_dynamo_unbacked_indices") == {0, 1}
        widths.append(metadata.plans[4].compressor_state_block_table.shape[1])
    assert widths == [2, 3, 2]


def test_prebuilt_compressor_table_is_a_pytree_input_and_is_reused(monkeypatch):
    from torch.utils._pytree import tree_flatten, tree_unflatten

    boundaries = torch.tensor([0, 8, 32], dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=24, max_k=24), (4,)
    )
    ascendc.build_compressor_state_tables(metadata)
    table = metadata.plans[4].compressor_state_block_table
    leaves, spec = tree_flatten(metadata)
    assert any(leaf is table for leaf in leaves)
    restored = tree_unflatten(leaves, spec)
    assert restored.plans[4].compressor_state_block_table is table
    assert restored.cu_seq_q_host is None  # The eager-only cache is not a graph input.
    zeros = torch.zeros

    def checked_zeros(shape, **kwargs):
        assert len(shape) == 3, "only the state cache should be allocated inside the graph"
        return zeros(shape, **kwargs)

    monkeypatch.setattr(torch, "zeros", checked_zeros)
    inputs = ascendc._state_inputs(
        boundaries,
        torch.empty((1, 32, 8)),
        max_length=32,
        ratio=4,
        head_dim=8,
        state_block_table=table,
    )
    assert inputs[0] is table
    torch.testing.assert_close(inputs[3], torch.tensor([8, 24], dtype=torch.int32))


def test_prebuilt_compressor_table_width_changes_on_graph_replay():
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced

    batches = []
    for lengths in ((16, 16), (8, 24), (8, 8, 16)):
        boundaries = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
        metadata = build_compressed_varlen_metadata(
            VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=max(lengths), max_k=max(lengths)), (4,)
        )
        ascendc.build_compressor_state_tables(metadata)
        setattr(boundaries, "_dynamo_unbacked_indices", {0})  # noqa: B010
        batches.append(metadata)

    def consume(metadata):
        table, cache, _, _ = ascendc._state_inputs(
            metadata.varlen.cu_seq_q,
            torch.empty((1, 32, 8)),
            max_length=32,
            ratio=4,
            head_dim=8,
            state_block_table=metadata.plans[4].compressor_state_block_table,
        )
        return table + cache.sum().to(torch.int32)

    traced = minimal_fx_tracer(consume)(batches[0])
    run = run_traced(traced, _validate_runtime=True)
    for metadata in batches:
        actual = run(metadata)
        torch.testing.assert_close(actual, metadata.plans[4].compressor_state_block_table)


@pytest.mark.parametrize("ratio,indexer", [(4, False), (4, True), (128, False)], ids=["c4-attention", "c4-indexer", "c128"])
@pytest.mark.parametrize("api", ["torch-op", "python-wrapper"])
@pytest.mark.parametrize("ape_dtype", [torch.float32, torch.bfloat16])
def test_compressor_cann_schema_document_outputs_and_gradients(
    compressor_factory, monkeypatch, ratio, indexer, api, ape_dtype
):
    calls = []

    def fake_cann(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *, state_block_table=None,
                  cu_seqlens=None, seqused=None, start_pos=None, coff=1, cache_mode=1):
        assert torch.is_grad_enabled()
        calls.append(cmp_ratio)
        assert x.dtype == wkv.dtype == wgate.dtype == torch.float16
        assert ape.dtype == state_cache.dtype == torch.float32
        assert x.shape[1] == wkv.shape[1] == wgate.shape[1] == 1024
        assert wkv is module.wkv.weight and wgate is module.wgate.weight
        torch.testing.assert_close(ape, module.ape.float())
        block_size = 8 if cmp_ratio == 4 else 16
        assert state_cache.shape == (1, block_size, 2 * coff * module.head_dim)
        assert state_block_table is metadata.plans[ratio].compressor_state_block_table
        assert state_block_table.shape == (len(lengths), max((max(lengths) + block_size - 1) // block_size, 1))
        assert not state_block_table.any() and start_pos is None and cache_mode == 1
        torch.testing.assert_close(cu_seqlens, boundaries)
        torch.testing.assert_close(seqused, torch.tensor(lengths, dtype=torch.int32))
        pooled = (2 * _document_pool(x, wkv, wgate, ape, cu, cmp_ratio) + perturbation).to(x.dtype)
        # Poison unused TND capacity so consuming it fails output comparisons.
        return torch.cat((pooled, pooled.new_full((len(lengths), module.head_dim), float("nan"))))

    if api == "torch-op":
        monkeypatch.setattr(
            torch.ops.cann_ops_transformer, "compressor", types.SimpleNamespace(default=fake_cann), raising=False
        )
    else:
        monkeypatch.setattr(torch.ops, "cann_ops_transformer", types.SimpleNamespace())
        monkeypatch.setattr(sys.modules["cann_ops_transformer.ops.compressor"], "compressor", fake_cann)

    module = compressor_factory(ratio, indexer)
    module.ape = torch.nn.Parameter(module.ape.detach().to(ape_dtype), requires_grad=module.ape.requires_grad)
    module.wgate.weight.requires_grad_(not indexer)
    reference = copy.deepcopy(module)
    lengths = [2 * ratio + 1, ratio - 1, ratio + 2]
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    boundaries = torch.tensor(cu, dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=max(lengths), max_k=max(lengths)), (ratio,)
    )
    ascendc.build_compressor_state_tables(metadata)
    generator = torch.Generator().manual_seed(21)
    x = (torch.randn(1, cu[-1], module.wkv.in_features, generator=generator) * 0.1).half().requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    perturbation = torch.linspace(-0.02, 0.03, module.head_dim)

    actual = module(x, metadata)
    # Match the operator boundary: accumulate APE gradients in FP32, then cast
    # once back to the BF16 parameter instead of rounding each sliced contribution.
    raw = _document_pool(
        reference_x[0], reference.wkv.weight, reference.wgate.weight, reference.ape.float(), cu, ratio
    )
    expected = _postprocess(reference, (2 * raw + perturbation).half(), metadata.plans[ratio].block_positions)
    upstream_grad = torch.randn(actual.shape, generator=generator).half()
    actual.backward(upstream_grad)
    expected.backward(upstream_grad)

    assert calls == [ratio]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0.006, atol=0.004)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        actual_grad = module.get_parameter(name).grad
        expected_grad = reference.get_parameter(name).grad
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.006, atol=0.004)


def test_compressor_empty_plan(compressor_factory):
    module = compressor_factory(4)
    boundaries = torch.tensor([0, 3], dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=3, max_k=3), (4,)
    )
    x = torch.ones(1, 3, module.wkv.in_features, dtype=torch.float16, requires_grad=True)
    actual = module(x, metadata)
    torch.testing.assert_close(actual, module._forward(x, metadata), rtol=0, atol=0)
    assert actual.shape == (0, module.head_dim)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("api", ["torch-op", "python-wrapper"])
def test_compressor_available_api_compiles(compressor_factory, monkeypatch, api):
    def fake_cann(x, wkv, wgate, state_cache, ape, cmp_ratio=4, **kwargs):
        width = wkv.shape[0] // 2
        values = F.linear(x.float(), wkv.float()).reshape(-1, cmp_ratio, 2, width)
        scores = F.linear(x.float(), wgate.float()).reshape_as(values) + ape.reshape(1, cmp_ratio, 2, width)
        return (values[:, :, 1] * scores[:, :, 1].softmax(dim=1)).sum(dim=1).to(x.dtype)

    if api == "torch-op":
        monkeypatch.setattr(
            torch.ops.cann_ops_transformer, "compressor", types.SimpleNamespace(default=fake_cann), raising=False
        )
    else:
        monkeypatch.setattr(torch.ops, "cann_ops_transformer", types.SimpleNamespace())
        monkeypatch.setattr(sys.modules["cann_ops_transformer.ops.compressor"], "compressor", fake_cann)
    module = compressor_factory(4).float()
    reference = copy.deepcopy(module)
    boundaries = torch.tensor([0, 8], dtype=torch.int32)
    metadata = build_compressed_varlen_metadata(
        VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=8, max_k=8), (4,)
    )
    ascendc.build_compressor_state_tables(metadata)
    setattr(metadata.plans[4].gather_indices, "_dynamo_unbacked_indices", {0})  # noqa: B010
    x = torch.randn(1, 8, module.wkv.in_features, generator=torch.Generator().manual_seed(17)).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()

    actual = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)(x, metadata)
    expected = reference(reference_x, metadata)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        torch.testing.assert_close(module.get_parameter(name).grad, reference.get_parameter(name).grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ratio,indexer", [(4, False), (4, True), (128, False)], ids=["c4-attention", "c4-indexer", "c128"])
@pytest.mark.parametrize("headtail", [False, True], ids=["contiguous", "headtail"])
@pytest.mark.parametrize("empty_rank", [False, True], ids=["packed", "empty-rank"])
@pytest.mark.parametrize("compile_model", [False, True], ids=["eager", "aot-eager"])
def test_compressor_cp_outputs_and_gradients(compressor_factory, monkeypatch, ratio, indexer, headtail, empty_rank, compile_model):
    from torch.distributed.tensor.experimental._context_parallel._load_balancer import _HeadTailLoadBalancer
    from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan

    module = compressor_factory(ratio, indexer).float()
    reference = copy.deepcopy(module)
    cp_size, total = 4, 8 * ratio
    # Misaligned documents, an empty document, and a document without full blocks.
    cu = [0, ratio - 1, 3 * ratio + 1, 3 * ratio + 1, total]
    if empty_rank:
        cu = [*range(total // cp_size + 1), total]
    boundaries = torch.tensor(cu, dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries, max_q=total, max_k=total)
    lb = _HeadTailLoadBalancer(total, cp_size, "cpu") if headtail else None
    perm = lb._generate_indices(restore=False).reshape(-1) if lb else torch.arange(total)
    metadata = [types.SimpleNamespace(varlen=v, plans=p, window=w) for v, p, w in (
        build_cp_plan(varlen, lb, rank=r, cp_size=cp_size, shard_len=total // cp_size,
                      window_size=4, ratios=[ratio]) for r in range(cp_size)
    )]
    if compile_model:
        for meta in metadata:
            indices = meta.plans[ratio].gather_indices
            if indices.numel() > 0:
                setattr(indices, "_dynamo_unbacked_indices", {0})  # noqa: B010
    generator = torch.Generator().manual_seed(42)
    x = (torch.randn(total, module.wkv.in_features, generator=generator) * 0.1).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    shards = x[perm].reshape(cp_size, total // cp_size, -1)
    calls = []

    def fake_cann(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *, state_block_table=None,
                  cu_seqlens=None, seqused=None, start_pos=None, coff=1, cache_mode=1):
        calls.append(x.shape[0])
        assert start_pos is None and cache_mode == 1
        # Tensor-only CPU kernel substitute; the independent oracle loops over
        # the original unsliced documents instead of consuming the CP plan.
        width = wkv.shape[0] // coff
        values = F.linear(x.float(), wkv.float()).reshape(-1, cmp_ratio, coff * width)
        scores = F.linear(x.float(), wgate.float()).reshape_as(values) + ape
        if coff == 2:
            previous = (torch.arange(values.shape[0]) - 1).clamp_min(0)
            first = torch.zeros(values.shape[0], dtype=torch.bool).scatter(0, cu_seqlens[:-1].long() // cmp_ratio, True)
            left_values = values[previous, :, :width].masked_fill(first[:, None, None], 0)
            left_scores = scores[previous, :, :width].masked_fill(first[:, None, None], float("-inf"))
            values = torch.cat((left_values, values[:, :, width:]), dim=1)
            scores = torch.cat((left_scores, scores[:, :, width:]), dim=1)
        return (values * scores.softmax(1)).sum(1).to(x.dtype)

    monkeypatch.setattr(module.implementation, "_compressor_fn", fake_cann)
    global_raw = _document_pool(reference_x, reference.wkv.weight, reference.wgate.weight, reference.ape, cu, ratio)
    positions = torch.cat([torch.arange(0, (b - a) // ratio * ratio, ratio) for a, b in zip(cu[:-1], cu[1:])])
    global_expected = _postprocess(reference, global_raw, positions)
    losses, expected_losses = [], []
    exchange_calls = []
    for rank, meta in enumerate(metadata):
        plan = meta.plans[ratio]

        def exchange(payload, in_splits, out_splits):
            exchange_calls.append(rank)
            pieces = []
            for peer, peer_meta in enumerate(metadata):
                ex = peer_meta.plans[ratio].exchange
                begin = sum(ex.send_splits[:rank])
                end = begin + ex.send_splits[rank]
                pieces.append(shards[peer][ex.send_indices[begin:end]])
            return torch.cat(pieces)

        monkeypatch.setattr(module.token_dispatcher, "_all_to_all", exchange)
        run = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True) if compile_model else module
        actual = run(shards[rank:rank + 1], meta)[plan.compressed_rows]
        # Independently map kept block positions to the unsliced document oracle.
        foreign = []
        for peer, peer_meta in enumerate(metadata):
            ex = peer_meta.plans[ratio].exchange
            begin = sum(ex.send_splits[:rank])
            end = begin + ex.send_splits[rank]
            foreign.append(perm[peer * (total // cp_size) + ex.send_indices[begin:end]])
        received = torch.cat(foreign)[plan.exchange.recv_offsets]
        augmented = torch.cat((perm[rank * (total // cp_size):(rank + 1) * (total // cp_size)], received))
        starts = augmented[plan.gather_indices].reshape(-1, ratio)[:, 0][plan.compressed_rows]
        expected_indices = []
        for start in starts.tolist():
            doc = next(i for i, (a, b) in enumerate(zip(cu[:-1], cu[1:])) if a <= start < b)
            expected_indices.append(sum((cu[i + 1] - cu[i]) // ratio for i in range(doc)) + (start - cu[doc]) // ratio)
        expected = global_expected[torch.tensor(expected_indices, dtype=torch.int64)]
        # CP segments and full documents use different CPU GEMM batch shapes.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        grad = torch.randn(actual.shape, generator=generator)
        losses.append((actual * grad).sum())
        expected_losses.append((expected * grad).sum())
    sum(losses).backward()
    sum(expected_losses).backward()
    assert len(calls) == sum(m.plans[ratio].gather_indices.numel() > 0 for m in metadata)
    assert exchange_calls == list(range(cp_size))
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        torch.testing.assert_close(module.get_parameter(name).grad, reference.get_parameter(name).grad, rtol=1e-4, atol=1e-5)
