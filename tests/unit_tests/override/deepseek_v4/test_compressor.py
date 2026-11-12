# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU oracle at the CANN boundary, with real configs and document metadata.

The fused compressor is CP-agnostic: it compresses the local stream it is handed, addressed by
that stream's own ``cu_seqlens``.  Under context parallelism the ratio-4 overlap's carry is
``CarryExchange``'s job, so the CP cases here compose the exchange with the same kernel module
and compare each rank's kept blocks with the whole row's pool.
"""

import copy
import importlib
import sys
import types
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import OverrideConfig, apply_overrides

from torchtitan_npu.models.common.carry_exchange import CarryExchange
from torchtitan_npu.models.deepseek_v4 import _debugmodel as _native_debugmodel
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.backend import with_npu_attention_backend
from torchtitan_npu.models.deepseek_v4.compressor import (
    Compressor,
    CompressorImplementation,
)
from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata
from torchtitan_npu.models.deepseek_v4.sharding import set_deepseek_v4_sharding_config
from torchtitan_npu.override.deepseek_v4.compressor import ascendc
from torchtitan_npu.override.deepseek_v4.metadata import AicpuGeometry
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import build_aicpu_metadata

_FUSED_GEOMETRY = AicpuGeometry(
    num_heads=16,
    head_dim=512,
    window_size=8,
    index_topk=16,
    index_num_heads=8,
    index_head_dim=128,
)


def _debugmodel():
    return with_npu_attention_backend(_native_debugmodel())


def _positions(cu: torch.Tensor) -> torch.Tensor:
    """A packed row's document-relative positions, from its boundaries."""
    return torch.cat([torch.arange(end - start) for start, end in pairwise(cu.tolist())])


def _entry(cu: torch.Tensor, ratio: int):
    """One geometry's model entry: the row and this ratio's fold of it."""
    return build_v4_metadata(_positions(cu), ratio, cu_seq_q=cu)


def _fused_entry(cu: torch.Tensor, ratio: int = 4):
    """One geometry's fused entry: the model entry plus its records."""
    return build_aicpu_metadata(_entry(cu, ratio), geometry=_FUSED_GEOMETRY)


def _tiny_config():
    """A two-layer stack with one compressed (ratio-4) layer, cheap to build."""
    config = _make_v4_config(
        dim=32,
        n_layers=2,
        vocab_size=64,
        n_heads=4,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=(1, 4),
        window_size=128,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=16,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=256,
        compress_rope_theta=10000.0,
        original_seq_len=256,
    )
    # The sharding pass runs at config resolution, before the overrides.
    config = with_npu_attention_backend(config)
    set_deepseek_v4_sharding_config(config, enable_sp=False, enable_ep=False)
    return config


@pytest.mark.parametrize("reverse", [False, True], ids=["compressor-first", "compressor-last"])
def test_compressor_override_composes_with_norm_and_rope(monkeypatch, reverse):
    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.common.rope import AscComplexRoPE

    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    cfg = _debugmodel()
    imports = [
        "torchtitan_npu.override.deepseek_v4.compressor.legacy_asc",
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


def _document_pool(x, wkv, wgate, ape, cu, ratio):
    outputs = []
    width = wkv.shape[0] // (2 if ratio == 4 else 1)
    for start, end in pairwise(cu):
        values = F.linear(x[start:end].float(), wkv.float())
        scores = F.linear(x[start:end].float(), wgate.float())
        for offset in range(0, (end - start) // ratio * ratio, ratio):
            current_values = values[offset : offset + ratio, -width:]
            current_scores = scores[offset : offset + ratio, -width:] + ape[:, -width:]
            if ratio == 4 and offset:
                current_values = torch.cat((values[offset - ratio : offset, :width], current_values))
                current_scores = torch.cat((scores[offset - ratio : offset, :width] + ape[:, :width], current_scores))
            outputs.append((current_values * current_scores.softmax(0)).sum(0))
    return torch.stack(outputs)


def _postprocess(module, pooled, positions):
    normalized = F.rms_norm(pooled, (module.head_dim,), module.norm.weight, module.norm.eps)
    return module.rope(normalized[:, None, :], positions=positions)[:, 0, :]


@pytest.fixture
def compressor_factory(monkeypatch):
    registry = importlib.import_module("torchtitan.config.override")
    with monkeypatch.context() as scoped, torch.random.fork_rng(devices=[]):
        scoped.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
        fake_module = types.ModuleType("compressor")
        fake_module.compressor = lambda *args, **kwargs: pytest.fail("unexpected CANN compressor call")
        scoped.setitem(sys.modules, "cann_ops_transformer.ops.compressor", fake_module)
        torch.manual_seed(93)

        def build(ratio, indexer=False):
            cfg = _debugmodel()
            apply_overrides(OverrideConfig(imports=["torchtitan_npu.override.deepseek_v4.compressor.legacy_asc"]), cfg)
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


@pytest.mark.parametrize(
    "ratio,indexer", [(4, False), (4, True), (128, False)], ids=["c4-attention", "c4-indexer", "c128"]
)
@pytest.mark.parametrize("api", ["torch-op", "python-wrapper"])
@pytest.mark.parametrize("ape_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_compressor_cann_schema_document_outputs_and_gradients(
    compressor_factory, monkeypatch, ratio, indexer, api, ape_dtype, batch_size
):
    calls = []
    lengths = [2 * ratio, ratio, 2 * ratio] * batch_size

    def fake_cann(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio=4,
        *,
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        coff=1,
        cache_mode=1,
    ):
        assert torch.is_grad_enabled()
        calls.append(cmp_ratio)
        assert x.dtype == wkv.dtype == wgate.dtype == torch.float16
        assert ape.dtype == state_cache.dtype == torch.float32
        assert x.shape[1] == wkv.shape[1] == wgate.shape[1] == 1024
        assert wkv is module.wkv.weight and wgate is module.wgate.weight
        torch.testing.assert_close(ape, module.ape.float())
        block_size = 8 if cmp_ratio == 4 else 16
        assert state_cache.shape == (1, block_size, 2 * coff * module.head_dim)
        # The port builds the all-zero tiling table from the stream's own shape symbols (the
        # model-level interface never sees it); a host maximum would go stale on the next batch.
        assert state_block_table.shape[0] == len(lengths)
        assert state_block_table.shape[1] == max((sum(lengths) + block_size - 1) // block_size, 1)
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
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    boundaries = torch.tensor(cu, dtype=torch.int32)
    positions = _positions(boundaries)
    metadata = _entry(boundaries, ratio)
    generator = torch.Generator().manual_seed(21)
    x = (
        (torch.randn(cu[-1], module.wkv.in_features, generator=generator) * 0.1)
        .half()
        .requires_grad_()
    )
    reference_x = x.detach().clone().requires_grad_()
    perturbation = torch.linspace(-0.02, 0.03, module.head_dim)

    actual = module(x, positions, cu_seqlens=metadata.swa_k.cu_seqlens)
    # Match the operator boundary: accumulate APE gradients in FP32, then cast
    # once back to the BF16 parameter instead of rounding each sliced contribution.
    raw = _document_pool(
        reference_x, reference.wkv.weight, reference.wgate.weight, reference.ape.float(), cu, ratio
    )
    expected = _postprocess(reference, (2 * raw + perturbation).half(), positions[::ratio])
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


@pytest.mark.parametrize("api", ["torch-op", "python-wrapper"])
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
def test_compressor_available_api_compiles(compressor_factory, monkeypatch, api, batch_size):
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
    boundaries = torch.arange(batch_size + 1, dtype=torch.int32) * 8
    positions = _positions(boundaries)
    metadata = _entry(boundaries, 4)
    x = torch.randn(batch_size * 8, module.wkv.in_features, generator=torch.Generator().manual_seed(17)).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()

    actual = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)(
        x, positions, cu_seqlens=metadata.swa_k.cu_seqlens
    )
    expected = reference(reference_x, positions, cu_seqlens=metadata.swa_k.cu_seqlens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    expected.sum().backward()
    # The crossing-document fixture also changes the CP-segment / full-document GEMM
    # batch shapes, so the gradient accumulation order differs; the measured worst
    # at c128 is 1.43e-05 (2/524288 elements), still fp32 rounding noise.
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=2e-5)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        torch.testing.assert_close(
            module.get_parameter(name).grad, reference.get_parameter(name).grad, rtol=1e-4, atol=1e-5
        )


class _FakeMesh:
    """The two mesh methods the carry reads."""

    def __init__(self, rank: int, size: int):
        self._rank, self._size = rank, size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


@pytest.mark.parametrize("ratio,indexer", [(4, False), (4, True)], ids=["c4-attention", "c4-indexer"])
def test_compressor_cp_carry_trim_matches_the_global_pool(compressor_factory, monkeypatch, ratio, indexer):
    """The exchange's keep-set is exactly this rank's block range of the whole row's pool.

    Two equal documents span the four slabs, so ranks 1 and 3 carry the previous rank's last
    block; rank 2's slab starts at the second document's edge and carries it as a segment of its
    own.  The fake kernel is the same per-document pool the unsharded tests use, so any
    difference is the CP framing itself.
    """
    from torchtitan_npu.models.common import carry_exchange as carry_module

    cp_size = 4
    total = 16 * ratio
    cu = [0, 8 * ratio, total]
    boundaries = torch.tensor(cu, dtype=torch.int32)
    positions = _positions(boundaries)
    slab = total // cp_size

    module = compressor_factory(ratio, indexer).float()
    reference = copy.deepcopy(module)
    generator = torch.Generator().manual_seed(42)
    x = (torch.randn(total, module.wkv.in_features, generator=generator) * 0.1).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    shards = x.reshape(cp_size, slab, -1)
    current = [0]

    def fake_all_to_all(send, *, output_split_sizes, input_split_sizes, group):
        del group
        # ``all_to_all_single`` splits dim 0, so the block travels as ``[ratio, D]`` rows.
        assert send.size(0) == sum(input_split_sizes)
        sources = [source for source, split in enumerate(output_split_sizes) if split]
        if not sources:
            return send[:0]
        assert sources == [current[0] - 1]
        source = sources[0]
        first = source * slab
        if send.dtype == torch.int32:
            return positions[..., first + slab - ratio : first + slab].reshape(-1).to(torch.int32)
        return shards[source][-ratio:]

    def fake_cann(
        x_flat,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio=4,
        *,
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        coff=1,
        cache_mode=1,
    ):
        del state_cache, state_block_table, seqused, start_pos, cache_mode
        assert coff == (2 if cmp_ratio == 4 else 1)
        return _document_pool(x_flat, wkv, wgate, ape, cu_seqlens.tolist(), cmp_ratio).to(x_flat.dtype)

    monkeypatch.setattr(carry_module, "all_to_all_single", fake_all_to_all)
    monkeypatch.setattr(module.implementation, "_compressor_fn", fake_cann)

    global_raw = _document_pool(reference_x, reference.wkv.weight, reference.wgate.weight, reference.ape, cu, ratio)
    global_positions = torch.cat(
        [torch.arange(0, (end - start) // ratio * ratio, ratio) for start, end in pairwise(cu)]
    )
    global_expected = _postprocess(reference, global_raw, global_positions)
    losses, expected_losses = [], []
    for rank in range(cp_size):
        current[0] = rank
        exchange = CarryExchange(CarryExchange.Config(ratio=ratio))
        exchange.group = _FakeMesh(rank, cp_size)
        carry = exchange(
            shards[rank : rank + 1],
            positions[None, rank * slab : (rank + 1) * slab],
            boundaries,
        )
        pooled = module(carry.x.squeeze(0), carry.positions.squeeze(0), cu_seqlens=carry.cu_seqlens)
        pooled = pooled.narrow(0, carry.trim, pooled.size(0) - carry.trim)
        actual = pooled
        expected = global_expected[rank * (slab // ratio) : (rank + 1) * (slab // ratio)]
        # The CP segments and full documents use different CPU GEMM batch shapes.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        grad = torch.randn(actual.shape, generator=generator)
        losses.append((actual * grad).sum())
        expected_losses.append((expected * grad).sum())
    sum(losses).backward()
    sum(expected_losses).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
    for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight"):
        torch.testing.assert_close(
            module.get_parameter(name).grad, reference.get_parameter(name).grad, rtol=1e-4, atol=1e-5
        )
