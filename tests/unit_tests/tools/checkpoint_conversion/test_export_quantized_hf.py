# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO / "torchtitan_npu" / "tools" / "checkpoint_conversion" / "export_quantized_hf.py"

_K = 64  # weight inner dim; 2 MX blocks of 32 per row
_N = 4  # per-expert rows for w1 / w3
_M = 6  # per-expert rows for w2
_N_EXPERTS = 2


@dataclass(frozen=True)
class _FakeQuantizeConfig:
    """Value-comparable stand-in for ``MXQuantizeConfig``; the merge compares configs by value."""

    elem_dtype: torch.dtype


_CFG = _FakeQuantizeConfig(torch.int32)


class _FakeMXTensor:
    """The ``MXTensor`` attribute surface the export script reads or writes."""

    def __init__(self, qdata, scale, orig_dtype, quant_axis, quant_config, act_quant_config=None, pack_axis=None):
        if qdata.dtype is not quant_config.elem_dtype:
            raise ValueError(
                f"qdata dtype {qdata.dtype} does not match quant_config.elem_dtype {quant_config.elem_dtype}"
            )
        if scale.ndim != qdata.ndim + 1:
            raise ValueError(f"scale.ndim must be qdata.ndim + 1 = {qdata.ndim + 1}, got {scale.ndim}")
        self.qdata = qdata
        self.scale = scale
        self.orig_dtype = orig_dtype
        self.quant_axis = quant_axis % qdata.ndim
        self.quant_config = quant_config
        self.act_quant_config = act_quant_config
        self.pack_axis = pack_axis % qdata.ndim if pack_axis is not None else None


@pytest.fixture(autouse=True)
def _fake_torchao_npu(monkeypatch):
    """Fake the script's call-time ``torchao_npu`` import (main-repo CI has no torchao)."""
    pkg = types.ModuleType("torchao_npu")
    sub = types.ModuleType("torchao_npu.quantized_tensors")
    sub.MXTensor = _FakeMXTensor
    pkg.quantized_tensors = sub
    monkeypatch.setitem(sys.modules, "torchao_npu", pkg)
    monkeypatch.setitem(sys.modules, "torchao_npu.quantized_tensors", sub)


@pytest.fixture(scope="module")
def export():
    spec = importlib.util.spec_from_file_location("export_quantized_hf_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values(layer, group, expert, rows):
    """Values unique to (layer, group, expert), ``group`` 0/1/2 = w1/w3/w2; reordering any level changes them."""
    base = layer * 100_000 + group * 10_000 + expert * 1_000
    return base + torch.arange(rows * _K).reshape(rows, _K)


def _mk_expert(layer, group, expert, rows, *, pack_axis=None, cfg=_CFG):
    """A per-expert 2D quantized weight with quant axis on its last dim."""
    scale = torch.full((_K // 32, 2), group * 10 + expert, dtype=torch.int8).repeat(rows, 1, 1)
    qdata = _values(layer, group, expert, rows).to(cfg.elem_dtype)
    return _FakeMXTensor(qdata, scale, torch.bfloat16, -1, cfg, None, pack_axis)


def _expert_fqns(layer, w, experts=range(_N_EXPERTS)):
    return [f"layers.{layer}.ffn.experts.{e}.w{w}.weight" for e in experts]


# =========================================================================
# _merge_routed_experts
# =========================================================================


def test_merge_w2_down_proj(export):
    """w2 experts merge into the 3D down_proj with per-expert values intact."""
    sd = {fqn: _mk_expert(0, 2, e, _M) for e, fqn in enumerate(_expert_fqns(0, "2"))}
    merged = export._merge_routed_experts(sd, sorted(sd))

    assert set(merged) == {"model.layers.0.mlp.experts.down_proj"}
    t = merged["model.layers.0.mlp.experts.down_proj"]
    assert t.qdata.shape == (_N_EXPERTS, _M, _K)
    assert t.scale.shape == (_N_EXPERTS, _M, _K // 32, 2)
    assert t.quant_axis == t.qdata.ndim - 1
    assert t.pack_axis is None
    for e in range(_N_EXPERTS):
        assert torch.equal(t.qdata[e].to(torch.int64), _values(0, 2, e, _M))


def test_merge_w1_w3_gate_up_concat_order(export):
    """gate_up_proj pairs each expert's w1 rows first, then that same expert's w3 rows."""
    sd = {}
    for e in range(_N_EXPERTS):
        sd[f"layers.0.ffn.experts.{e}.w1.weight"] = _mk_expert(0, 0, e, _N)
        sd[f"layers.0.ffn.experts.{e}.w3.weight"] = _mk_expert(0, 1, e, _N)
    merged = export._merge_routed_experts(sd, sorted(sd))

    assert set(merged) == {"model.layers.0.mlp.experts.gate_up_proj"}
    t = merged["model.layers.0.mlp.experts.gate_up_proj"]
    assert t.qdata.shape == (_N_EXPERTS, 2 * _N, _K)
    for e in range(_N_EXPERTS):
        expected = torch.cat([_values(0, 0, e, _N), _values(0, 1, e, _N)], dim=0)
        assert torch.equal(t.qdata[e].to(torch.int64), expected)
        # scale value encodes (group, expert): this expert's w1 (0) and w3 (1) rows only
        assert t.scale[e].unique().tolist() == sorted((0 * 10 + e, 1 * 10 + e))


def test_merge_pack_axis_rebased_to_last_axis(export):
    """The merged pack axis points at the merged tensor's last dim, not the stale per-expert index."""
    sd = {fqn: _mk_expert(0, 2, e, _M, pack_axis=-1) for e, fqn in enumerate(_expert_fqns(0, "2"))}
    t = export._merge_routed_experts(sd, sorted(sd))["model.layers.0.mlp.experts.down_proj"]
    assert t.qdata.ndim == 3
    assert t.pack_axis == t.qdata.ndim - 1


def test_merge_layer_fqn_mapping(export):
    """Each (layer, w) group lands in its own FQN with only that layer's values."""
    sd = {}
    for layer in (0, 7):
        for group, w, rows in ((0, "1", _N), (1, "3", _N), (2, "2", _M)):
            for e in range(_N_EXPERTS):
                sd[f"layers.{layer}.ffn.experts.{e}.w{w}.weight"] = _mk_expert(layer, group, e, rows)
    merged = export._merge_routed_experts(sd, sorted(sd))

    assert set(merged) == {f"model.layers.{layer}.mlp.experts.gate_up_proj" for layer in (0, 7)} | {
        f"model.layers.{layer}.mlp.experts.down_proj" for layer in (0, 7)
    }
    for layer in (0, 7):
        t = merged[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        for e in range(_N_EXPERTS):
            expected = torch.cat([_values(layer, 0, e, _N), _values(layer, 1, e, _N)], dim=0)
            assert torch.equal(t.qdata[e].to(torch.int64), expected)


def test_merge_rejects_incomplete_experts(export):
    sd = {fqn: _mk_expert(0, 2, e, _M) for e, fqn in zip((0, 2), _expert_fqns(0, "2", experts=(0, 2)), strict=True)}
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        export._merge_routed_experts(sd, sorted(sd))


@pytest.mark.parametrize("w3_experts", [(), (0,)])
def test_merge_rejects_w1_without_w3(export, w3_experts):
    sd = {fqn: _mk_expert(0, 0, e, _N) for e, fqn in enumerate(_expert_fqns(0, "1"))}
    for e in w3_experts:
        sd[f"layers.0.ffn.experts.{e}.w3.weight"] = _mk_expert(0, 1, e, _N)
    with pytest.raises(ValueError, match="w3"):
        export._merge_routed_experts(sd, sorted(sd))


def test_merge_rejects_attr_mismatch(export):
    """Experts whose quantization configs differ by value cannot be merged."""
    sd = {
        "layers.0.ffn.experts.0.w2.weight": _mk_expert(0, 2, 0, _M, cfg=_FakeQuantizeConfig(torch.int16)),
        "layers.0.ffn.experts.1.w2.weight": _mk_expert(0, 2, 1, _M),
    }
    with pytest.raises(ValueError, match="disagree"):
        export._merge_routed_experts(sd, sorted(sd))


# =========================================================================
# _to_in_memory_layout
# =========================================================================

_DENSE_CASES = {
    "layers.5.attn.wq_a.weight": "model.layers.5.self_attn.q_a_proj.weight",
    "layers.5.attn.wq_b.weight": "model.layers.5.self_attn.q_b_proj.weight",
    "layers.5.attn.wkv.weight": "model.layers.5.self_attn.kv_proj.weight",
    "layers.5.attn.wo_a.weight": "model.layers.5.self_attn.o_a_proj.weight",
    "layers.5.attn.wo_b.weight": "model.layers.5.self_attn.o_b_proj.weight",
    "layers.5.attn.indexer.wq_b.weight": "model.layers.5.self_attn.compressor.indexer.q_b_proj.weight",
    "layers.5.ffn.shared_experts.w1.weight": "model.layers.5.mlp.shared_experts.gate_proj.weight",
    "layers.5.ffn.shared_experts.w2.weight": "model.layers.5.mlp.shared_experts.down_proj.weight",
    "layers.5.ffn.shared_experts.w3.weight": "model.layers.5.mlp.shared_experts.up_proj.weight",
}


def test_to_in_memory_layout_renames_dense(export):
    """All nine dense mappings rename; non-quantized tensors pass through untouched."""
    sd = {fqn: _mk_expert(5, 0, 0, _N) for fqn in _DENSE_CASES}
    sd["layers.5.attn.wo_b.bias"] = torch.zeros(_N, dtype=torch.bfloat16)
    sd["layers.5.rope.freqs"] = torch.zeros(8, dtype=torch.int64)

    out, new_quantized = export._to_in_memory_layout(sd, list(_DENSE_CASES))

    for src, dst in _DENSE_CASES.items():
        assert out[dst] is sd[src]
    assert out["layers.5.attn.wo_b.bias"] is sd["layers.5.attn.wo_b.bias"]
    assert out["layers.5.rope.freqs"] is sd["layers.5.rope.freqs"]
    assert set(out) == set(_DENSE_CASES.values()) | {"layers.5.attn.wo_b.bias", "layers.5.rope.freqs"}
    assert new_quantized == [_DENSE_CASES[f] for f in sorted(_DENSE_CASES)]


def test_to_in_memory_layout_merges_experts_and_preserves_plain(export):
    """Quantized tensors become merged 3D experts + renamed dense; plain tensors are preserved."""
    sd = {fqn: _mk_expert(5, 0, 0, _N) for fqn in _DENSE_CASES}
    for e in range(_N_EXPERTS):
        for group, w, rows in ((0, "1", _N), (1, "3", _N), (2, "2", _M)):
            sd[f"layers.5.ffn.experts.{e}.w{w}.weight"] = _mk_expert(5, group, e, rows)
    sd["lm_head.weight"] = torch.randn(32, 8)

    out, new_quantized = export._to_in_memory_layout(sd, [k for k in sd if k != "lm_head.weight"])

    assert not any(export._EXPERT_FQN_RE.match(k) for k in out)
    assert "model.layers.5.mlp.experts.gate_up_proj" in out
    assert "model.layers.5.mlp.experts.down_proj" in out
    assert out["lm_head.weight"] is sd["lm_head.weight"]
    assert set(new_quantized) == set(_DENSE_CASES.values()) | {
        "model.layers.5.mlp.experts.gate_up_proj",
        "model.layers.5.mlp.experts.down_proj",
    }


def test_to_in_memory_layout_rejects_unmapped_quantized(export):
    sd = {"layers.0.attn.wz.weight": _mk_expert(0, 0, 0, _N)}
    with pytest.raises(ValueError, match="no in-memory name mapping"):
        export._to_in_memory_layout(sd, list(sd))


# =========================================================================
# _copy_hf_assets
# =========================================================================


def test_copy_hf_assets_warns_on_skipped_entries(export, tmp_path, caplog):
    """Copied files land in the output; skipped dirs/symlinks each produce a warning."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text("{}")
    (assets / "tokenizer_config.json").write_text("{}")
    (assets / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (assets / "model.safetensors.index.json").write_text("{}")
    (assets / "tokenizer").mkdir()
    (assets / "README.md").symlink_to(assets / "config.json")
    out = tmp_path / "out"
    out.mkdir()

    with caplog.at_level(logging.WARNING, logger="export_quantized_hf_under_test"):
        export._copy_hf_assets(assets, out)

    assert sorted(p.name for p in out.iterdir()) == ["config.json", "tokenizer_config.json"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    skipped = [name for name in ("tokenizer", "README.md") if any(name in msg for msg in warnings)]
    assert skipped == ["tokenizer", "README.md"]
