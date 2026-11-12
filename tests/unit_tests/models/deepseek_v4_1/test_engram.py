# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchtitan.components.optim import AdamW, DistMuon

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import cp_parallel_dims, tiny_v41_model
from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import SparseAdam
from torchtitan_npu.models.deepseek_v4_1 import (
    EngramArgs,
    model_registry,
)
from torchtitan_npu.models.deepseek_v4_1.engram.config import _make_engram_configs
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
)
from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_multimodal

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _build_engram(token_id_map_path=None):
    config = _make_engram_configs(
        hidden_size=8,
        hc_mult=4,
        vocab_size=32,
        engram=EngramArgs(
            layer_ids=(1,),
            vocab_size_per_ngram=(11, 17),
            n_embed_per_ngram=6,
            num_heads_per_ngram=2,
            pad_id=2,
            require_token_id_map=False,
            token_id_map_path=token_id_map_path,
        ),
    )[1]
    engram = config.build()
    engram.init_states(buffer_device=torch.device("cpu"))
    return engram


def _numpy_hash_reference(table, input_ids, positions=None, *, image_mask=None, token_history=None):
    """Scalar N-gram oracle, independent of the production shift construction."""
    mapping = table.token_id_map.numpy()
    tokens = mapping[input_ids.numpy()]
    history = mapping[token_history.numpy()] if token_history is not None else np.empty(0, dtype=np.int64)
    pad = mapping[table.pad_id]
    rows = []
    for index in range(len(tokens)):
        head_rows = []
        head = 0
        for order in table.ngram_orders:
            mixed = np.int64(0)
            for distance in range(order):
                source = index - distance
                token = tokens[source] if source >= 0 else history[source] if source >= -len(history) else pad
                if positions is not None and positions[index] < distance:
                    token = pad
                if image_mask is not None and image_mask[max(0, source) : index + 1].any():
                    token = pad
                mixed = np.bitwise_xor(mixed, token * table.hash_multipliers[distance].item())
            for _ in range(table.num_heads):
                head_rows.append(mixed % table.head_vocab_sizes[head].item() + table.offsets[head].item())
                head += 1
        rows.append(head_rows)
    return torch.tensor(rows, dtype=torch.int64)


def _official_demo_forward_reference(engram, hidden, input_ids, positions):
    """Independent, branch-by-branch port of the public Engram demo."""
    table = engram.table
    hashes = _numpy_hash_reference(table, input_ids, positions)
    memory = F.embedding(hashes, table.weight).flatten(1)
    D, M = engram.gate.hidden_size, engram.gate.num_branches
    # the fused projection holds each branch's key first, then the shared value
    key_rows, value_rows = engram.gate.wkv.split([M * D, D], dim=0)
    values = F.linear(memory, value_rows)

    gated = []
    for branch in range(M):
        key = F.linear(memory, key_rows[branch * D : (branch + 1) * D])
        key = F.rms_norm(
            key,
            (engram.gate.hidden_size,),
            engram.gate.k_weight[branch],
            engram.gate.norm_eps,
        )
        query = F.rms_norm(
            hidden[:, branch],
            (engram.gate.hidden_size,),
            engram.gate.q_weight[branch],
            engram.gate.norm_eps,
        )
        logits = (key * query).sum(-1) / math.sqrt(engram.gate.hidden_size)
        logits = logits.abs().clamp_min(1e-6).sqrt() * logits.sign()
        gated.append(logits.sigmoid().unsqueeze(-1) * values)
    return hidden + torch.stack(gated, dim=1)


def test_hash_resets_ngram_context_at_packed_document_boundary():
    table = _build_engram().table
    packed_ids = torch.tensor([4, 5, 9, 10], dtype=torch.int64)
    packed_positions = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    packed_hashes = table.hash(packed_ids, packed_positions)

    second_document = torch.tensor([9, 10], dtype=torch.int64)
    document_positions = torch.tensor([0, 1], dtype=torch.int64)
    document_hashes = table.hash(second_document, document_positions)
    torch.testing.assert_close(packed_hashes[2:], document_hashes, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ids,positions,history,image_mask",
    [
        ([4], [0], None, None),
        ([4, 4, 5, 4, 4], [0, 1, 0, 1, 2], None, None),
        ([4, 5], [1, 2], [9], None),
        ([4, 5], [5, 6], [7, 8, 9, 10, 11], None),
        ([4, 5], [0, 1], [7, 8, 9], None),
        ([4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6], None, [False, True, True, False, False, False, False]),
    ],
    ids=["short", "repeated-packed", "short-history", "long-history", "reset-history", "image-span"],
)
def test_hash_with_compressed_tokens_matches_scalar_ngram_oracle(tmp_path, ids, positions, history, image_mask):
    mapping_path = tmp_path / "token_map.npy"
    np.save(mapping_path, np.array([(i * 5 + 3) % 11 for i in range(32)], dtype=np.int64))
    table = _build_engram(str(mapping_path)).table
    ids, positions = torch.tensor(ids), torch.tensor(positions)
    history = torch.tensor(history) if history is not None else None
    image_mask = torch.tensor(image_mask) if image_mask is not None else None

    actual = table.hash(ids, positions, token_history=history, image_mask=image_mask)
    expected = _numpy_hash_reference(table, ids, positions, token_history=history, image_mask=image_mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fetched = table(ids, positions, token_history=history, image_mask=image_mask)
    torch.testing.assert_close(fetched, F.embedding(expected, table.weight).flatten(1), rtol=0, atol=0)


def test_token_history_requires_positions_and_text_only():
    table = _build_engram().table
    ids, history = torch.tensor([4, 5]), torch.tensor([1])
    with pytest.raises(ValueError, match="document-relative positions"):
        table.hash(ids, token_history=history)
    with pytest.raises(ValueError, match="text-only"):
        table.hash(ids, torch.arange(2), token_history=history, image_mask=torch.zeros(2, dtype=torch.bool))


def test_per_row_history_and_one_window_state_the_same_rows():
    """Two history contracts, one meaning: the tokens before the local rows.

    ``[H]`` states one window shared by every local row; ``[T, H]`` states each token's
    own predecessors in the document-parallel layout.
    Where both can express the same rows they must hash identically.
    """
    table = _build_engram().table
    ids = torch.tensor([9, 10], dtype=torch.int64)
    positions = torch.tensor([0, 1], dtype=torch.int64)

    # Two n-gram orders, so a row looks two tokens back: the window carries both distances.
    window = table.hash(ids, positions, token_history=torch.tensor([9, 9], dtype=torch.int64))
    per_row = table.hash(ids, positions, token_history=torch.tensor([[9, 9], [9, 9]], dtype=torch.int64))
    torch.testing.assert_close(window, per_row, rtol=0, atol=0)

    # The first row starts its document, so its entries are masked either way; giving it another
    # document's tail changes nothing, which is the mask the per-row contract relies on.
    other_tail = table.hash(ids, positions, token_history=torch.tensor([[5, 5], [9, 9]], dtype=torch.int64))
    torch.testing.assert_close(window, other_tail, rtol=0, atol=0)


def test_doc_chunk_engram_history_matches_full_document_hashes(monkeypatch):
    model, _ = tiny_v41_model(monkeypatch)
    model.engram_history_size = 2
    tokens = torch.arange(24)
    positions = torch.cat((torch.arange(8), torch.arange(16)))
    local_rows = torch.tensor([4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23])

    history = model._engram_row_history(tokens, positions, cp_parallel_dims(2, 1))
    table = _build_engram().table
    expected = table.hash(tokens, positions)[local_rows]
    actual = table.hash(tokens[local_rows], positions[local_rows], token_history=history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_forward_backward_matches_official_demo_dataflow():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        engram = _build_engram()
        reference = copy.deepcopy(engram)
        hidden = torch.randn(9, 4, 8, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        input_ids = torch.randint(0, 32, (9,))
        positions = torch.arange(9)
        actual = engram(hidden, input_ids, positions)
        expected = _official_demo_forward_reference(reference, reference_hidden, input_ids, positions)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        upstream_grad = torch.randn_like(actual)
        actual.backward(upstream_grad)
        expected.backward(upstream_grad)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            engram.table.pending_sparse_grad().to_dense(), reference.table.weight.grad, rtol=2e-4, atol=2e-5
        )
        for name, parameter in engram.gate.named_parameters():
            torch.testing.assert_close(
                parameter.grad, dict(reference.gate.named_parameters())[name].grad, rtol=2e-4, atol=2e-5
            )


def test_bfloat16_gate_keeps_normalization_and_residual_in_fp32():
    # Independent per-branch RMSNorm oracle for cann-recipes-infer 76b9cb8,
    # models/deepseek_v4_1/models/modules/engram.py (unquantized projection).
    torch.manual_seed(31)
    engram = _build_engram().to(dtype=torch.bfloat16)
    reference = copy.deepcopy(engram)
    hidden = torch.randn(17, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    input_ids = torch.randint(0, 32, (17,))
    with torch.no_grad():
        engram.gate.q_weight.uniform_(0.25, 1.75)
        engram.gate.k_weight.uniform_(0.25, 1.75)
        reference.gate.load_state_dict(engram.gate.state_dict())

    actual = engram(hidden, input_ids)
    memory = F.embedding(_numpy_hash_reference(reference.table, input_ids), reference.table.weight).flatten(1)
    kv = F.linear(memory.to(hidden.dtype), reference.gate.wkv).float()
    keys, value = kv.split([32, 8], dim=-1)
    gated = []
    for branch, key in enumerate(keys.split(8, dim=-1)):
        query = F.rms_norm(
            reference_hidden[:, branch].float(), (8,), reference.gate.q_weight[branch].float(), engram.gate.norm_eps
        )
        key = F.rms_norm(key, (8,), reference.gate.k_weight[branch].float(), engram.gate.norm_eps)
        dot = (query * key).sum(-1) / math.sqrt(8)
        magnitude = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.where(dot >= 0, magnitude, -magnitude).sigmoid()
        gated.append(gate.unsqueeze(-1) * value)
    expected = (reference_hidden.float() + torch.stack(gated, dim=1)).to(hidden.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream_grad = torch.randn_like(actual)
    actual.backward(upstream_grad)
    expected.backward(upstream_grad)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0.02, atol=0.015625)
    for name, parameter in engram.gate.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(reference.gate.named_parameters())[name].grad, rtol=0.02, atol=0.015625
        )
    torch.testing.assert_close(
        engram.table.pending_sparse_grad().to_dense(), reference.table.weight.grad, rtol=0.02, atol=0.015625
    )


def test_zero_dot_gate_uses_positive_clamped_sqrt():
    engram = _build_engram()
    with torch.no_grad():
        engram.table.weight.fill_(1)
        engram.gate.wkv.zero_()
        engram.gate.wkv[-8:].fill_(1 / engram.gate.wkv.shape[1])
        engram.gate.q_weight.zero_()
    hidden = torch.zeros(2, 4, 8, requires_grad=True)

    output = engram(hidden, torch.tensor([4, 5]))

    # Zero dot maps to sigmoid(sqrt(1e-6)), rather than sigmoid(0).
    expected_gate = 1 / (1 + math.exp(-0.001))
    torch.testing.assert_close(output, torch.full_like(output, expected_gate), rtol=0, atol=1e-7)
    output.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    torch.testing.assert_close(engram.gate.q_weight.grad, torch.zeros_like(engram.gate.q_weight), rtol=0, atol=0)


def test_debug_flavor_enables_engram_and_selects_sparse_adam():
    spec = model_registry("deepseek_v4_1_debugmodel")
    enabled_layers = [idx for idx, layer in enumerate(spec.layers) if layer.engram is not None]
    assert enabled_layers == [1, 14]

    trainer_config = deepseek_v4_1_debugmodel_multimodal()
    param_groups = trainer_config.optim.optimizer.optimizers
    assert param_groups[0].pattern == r".*\.engram\.table\.weight$"
    assert isinstance(param_groups[0], SparseAdam.Config)
    assert param_groups[0].lr == 5e-5
    assert isinstance(param_groups[1], AdamW.Config)


def test_released_v41_geometry_builds_published_logical_tables():
    """Official row totals exercise prime allocation across both complete layers."""
    full = model_registry("deepseek_v4_1_flash_40layers_24experts_multimodal")
    layers = [(i, layer.engram) for i, layer in enumerate(full.layers) if layer.engram is not None]
    # Source: deepseek-ai/DeepSeek-V4.1-Flash, revision 2bc89ac, text_config.
    assert [i for i, _ in layers] == [1, 14]
    for (_, config), logical_rows in zip(layers, (384006168, 384016682), strict=True):
        table = config.table
        assert sum(table.head_vocab_sizes) == logical_rows
        assert table.num_embeddings == (logical_rows + 2047) // 2048 * 2048
        assert (table.ngram_orders, table.num_heads, table.embedding_dim) == ((2, 3, 4), 8, 256)
        assert table.pad_id == 2
        assert table.require_token_id_map and table.compressed_vocab_size == 99092
        assert config.gate.memory_dim == 6144
        assert config.gate.norm_eps == 1e-20


def test_image_tokens_reset_hash_history_and_receive_no_residual_gradient():
    engram = _build_engram()
    ids = torch.tensor([1, 4, 5, 6, 7, 8])
    positions = torch.arange(6)
    mask = torch.tensor([False, True, True, False, False, False])
    hashes = engram.table.hash(ids, positions, image_mask=mask)
    # Text suffix after an image must hash exactly like a new document.
    expected = engram.table.hash(ids[3:], torch.arange(3))
    torch.testing.assert_close(hashes[3:], expected, rtol=0, atol=0)
    hidden = torch.randn(6, 4, 8, requires_grad=True)
    output = engram(hidden, ids, positions, image_mask=mask)
    torch.testing.assert_close(output[mask], hidden[mask], rtol=0, atol=0)
    output[mask].sum().backward()
    torch.testing.assert_close(hidden.grad[mask], torch.ones_like(hidden.grad[mask]))
    assert torch.count_nonzero(engram.gate.wkv.grad) == 0
    assert torch.count_nonzero(engram.table.pending_sparse_grad().values()) == 0


@pytest.mark.parametrize("optimizer", ["AdamW", "Muon"])
def test_explicit_disable_preserves_selected_optimizer(optimizer):
    trainer = replace(deepseek_v4_1_debugmodel_multimodal(), engram_enabled=False)
    trainer.optim.optimizer.name = optimizer
    trainer.training.num_tokens_per_microbatch_per_dp_rank = 512
    trainer.model.update_from_config(config=trainer)
    assert all(layer.engram is None for layer in trainer.model.layers)
    assert type(trainer.optim.optimizer) is OptimizerConfig
    assert trainer.optim.optimizer.name == optimizer
    trainer.optim.optimizer.materialize()
    groups = trainer.optim.optimizer.optimizers
    expected_types = [DistMuon.Config, AdamW.Config] if optimizer == "Muon" else [AdamW.Config]
    assert len(groups) == len(expected_types)
    assert all(isinstance(group, expected) for group, expected in zip(groups, expected_types, strict=True))
    adapter = DeepSeekV41StateDictAdapter(trainer.model, None)
    assert adapter.to_hf({}) == {}


@pytest.mark.parametrize("fused_attention", [False, True], ids=["reference", "fused"])
def test_model_config_accepts_parameterized_engram_override(fused_attention):
    from torchtitan.config.override import apply_overrides

    from torchtitan_npu.override.deepseek_v4_1.engram.mxfp8 import MXFP8HostOffloadEngramTable
    from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_text

    config = deepseek_v4_1_debugmodel_text()
    config.hf_assets_path = "tests/assets/deepseek_v3"
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.override.imports = [
        (
            "torchtitan_npu.override.deepseek_v4_1.engram.host_offload_mxfp8",
            {"num_max_tokens_per_rank": 3072, "shard_over_efsdp": True},
        ),
    ]
    if fused_attention:
        config.override.imports += [
            "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            ("torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc", {}),
        ]
    # Follow Trainer's order: model validation runs before applying overrides.
    config.model.update_from_config(config=config)
    apply_overrides(config.override, config)

    tables = [layer.engram.table for layer in config.model.layers if layer.engram is not None]
    assert tables
    assert all(isinstance(table, MXFP8HostOffloadEngramTable.Config) for table in tables)
    assert all(table.num_max_tokens_per_rank == 3072 and table.shard_over_efsdp for table in tables)
