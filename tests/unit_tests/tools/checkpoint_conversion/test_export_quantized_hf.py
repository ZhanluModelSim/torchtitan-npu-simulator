# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import logging
import re
import struct
import subprocess
import sys
import types
from collections import OrderedDict
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

    def __init__(self, qdata, scale, dtype, quant_axis, quant_config, act_quant_config=None, pack_axis=None):
        if qdata.dtype is not quant_config.elem_dtype:
            raise ValueError(
                f"qdata dtype {qdata.dtype} does not match quant_config.elem_dtype {quant_config.elem_dtype}"
            )
        if scale.ndim != qdata.ndim + 1:
            raise ValueError(f"scale.ndim must be qdata.ndim + 1 = {qdata.ndim + 1}, got {scale.ndim}")
        self.qdata = qdata
        self.scale = scale
        self.dtype = dtype
        self.quant_axis = quant_axis % qdata.ndim
        self.quant_config = quant_config
        self.act_quant_config = act_quant_config
        self.pack_axis = pack_axis % qdata.ndim if pack_axis is not None else None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSwap:
    """Stand-in for ``ParamSwapConfig``: identity is all the FqnToConfig builder needs."""

    def __init__(self, name):
        self.name = name


class _FakeFqnToConfig:
    """Field surface the FqnToConfig builder uses (mirrors torchao quant_api)."""

    def __init__(self, fqn_to_config=None):
        self.fqn_to_config = OrderedDict() if fqn_to_config is None else fqn_to_config


def _fake_config_to_dict(config):
    """``_type/_data`` record shape of torchao's ``config_to_dict``, tagged per swap."""
    mapping = {k: {"_type": "ParamSwapConfig", "_data": {"tag": v.name}} for k, v in config.fqn_to_config.items()}
    return {
        "_type": "FqnToConfig",
        "_version": 1,
        # mirrors ``FqnToConfig.__post_init__`` aliasing both fields to one dict
        "_data": {"fqn_to_config": mapping, "module_fqn_to_config": dict(mapping)},
    }


@pytest.fixture(autouse=True)
def _fake_torchao_npu(monkeypatch):
    """Fake the script's call-time ``torchao_npu``/``torchao`` imports (absent in main-repo CI).

    Stubs match the record shape the export relies on; real fidelity is covered
    by the NPU e2e round-trip.
    """
    pkg = types.ModuleType("torchao_npu")
    sub = types.ModuleType("torchao_npu.quantized_tensors")
    sub.MXTensor = _FakeMXTensor
    pkg.quantized_tensors = sub
    monkeypatch.setitem(sys.modules, "torchao_npu", pkg)
    monkeypatch.setitem(sys.modules, "torchao_npu.quantized_tensors", sub)

    # ``_plan_weight_config`` reads ``dataclasses.fields(MXQuantizeConfig)`` like
    # ``to_mx_tensor``, so the stub must expose those field names.
    @dataclass
    class _FakeMXQuantizeConfig:
        block_size: int = 32
        elem_dtype: torch.dtype = torch.float8_e4m3fn
        round_mode: str = "rint"
        scale_alg: int | None = None
        dst_type_max: float = 0.0

    quantization = types.ModuleType("torchao_npu.quantization")
    quant_configs = types.ModuleType("torchao_npu.quantization.quant_configs")
    quant_configs.MXQuantizeConfig = _FakeMXQuantizeConfig
    quantization.quant_configs = quant_configs
    pkg.quantization = quantization
    monkeypatch.setitem(sys.modules, "torchao_npu.quantization", quantization)
    monkeypatch.setitem(sys.modules, "torchao_npu.quantization.quant_configs", quant_configs)

    ao = types.ModuleType("torchao")
    core = types.ModuleType("torchao.core")
    core_cfg = types.ModuleType("torchao.core.config")
    core_cfg.config_to_dict = _fake_config_to_dict
    core.config = core_cfg
    quant = types.ModuleType("torchao.quantization")
    quant.FqnToConfig = _FakeFqnToConfig
    ao.core = core
    ao.quantization = quant
    for name, mod in (
        ("torchao", ao),
        ("torchao.core", core),
        ("torchao.core.config", core_cfg),
        ("torchao.quantization", quant),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture(scope="module")
def export():
    return _load_module("export_quantized_hf_under_test", _SCRIPT)


def _values(layer, group, expert, rows):
    """Values unique to (layer, group, expert), ``group`` 0/1/2 = w1/w3/w2; reordering any level changes them."""
    base = layer * 100_000 + group * 10_000 + expert * 1_000
    return base + torch.arange(rows * _K).reshape(rows, _K)


def _mk_expert(layer, group, expert, rows, *, pack_axis=None, cfg=_CFG, act=None):
    """A per-expert 2D quantized weight with quant axis on its last dim."""
    scale = torch.full((_K // 32, 2), group * 10 + expert, dtype=torch.int8).repeat(rows, 1, 1)
    qdata = _values(layer, group, expert, rows).to(cfg.elem_dtype)
    return _FakeMXTensor(qdata, scale, torch.bfloat16, -1, cfg, act, pack_axis)


def _expert_fqns(layer, w, experts=range(_N_EXPERTS)):
    return [f"layers.{layer}.ffn.experts.{e}.w{w}.weight" for e in experts]


# =========================================================================
# build_recipe_plan / quantize_state_dict
# =========================================================================


def test_build_recipe_plan_routes_recipes(export, monkeypatch):
    """Each recipe maps to the right ``(swap, filter)`` order; the mxfp4 flag reaches the routed entry."""
    filters = types.ModuleType("torchao_npu.quantization.filters")
    filters.any_filter = lambda *fs: ("any", *fs)
    filters.match_fqn_regex = lambda pat: ("regex", pat)
    filters.match_fqn_suffix = lambda *sfx: ("suffix", *sfx)
    # Stub the converter leaf so its import never touches the real module (which
    # pulls the heavy ``torchtitan_npu`` package + a ``torchao.core.config`` the
    # autouse torchao stub intentionally omits).
    tc = types.ModuleType("torchtitan_npu.extensions.adaptor.torchao.converter")
    tc.mxfp8_param_swap = lambda: "MX"
    tc.block_fp8_param_swap = lambda **kw: ("BLOCK", tuple(sorted(kw)))
    sys.modules["torchao_npu.quantization"].filters = filters
    for name, mod in (
        ("torchao_npu.quantization.filters", filters),
        ("torchtitan_npu.extensions.adaptor.torchao.converter", tc),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    p = export.build_recipe_plan("all_mxfp8", enable_mxfp4_qat=False)
    assert [cfg for cfg, _ in p] == ["MX"]
    assert p[0][1][0] == "any"

    p = export.build_recipe_plan("mix", enable_mxfp4_qat=True, dst_type_max=5.0)
    assert [cfg for cfg, _ in p] == ["MX", ("BLOCK", ("dst_type_max", "enable_mxfp4_qat"))]

    p = export.build_recipe_plan("all_block_fp8", enable_mxfp4_qat=False)
    assert [cfg for cfg, _ in p] == [("BLOCK", ()), ("BLOCK", ("dst_type_max", "enable_mxfp4_qat"))]

    with pytest.raises(ValueError, match="recipe must be one of"):
        export.build_recipe_plan("nope", enable_mxfp4_qat=False)
    with pytest.raises(ValueError, match="all_hif8"):
        export.build_recipe_plan("all_hif8", enable_mxfp4_qat=False)


def test_quantize_state_dict_threads_plan_index(export, monkeypatch):
    """Matched tensors become quantized and carry the index of the plan entry that claimed them.

    This ``fqn -> plan index`` dict is what ``_build_fqn_to_config_record`` groups the
    ``FqnToConfig`` plan by, so attribution must survive first-match and passthrough.
    """
    monkeypatch.setattr(export, "_quantize_to_mx_tensor", lambda w, swap, device: f"Q<{swap}>")
    plan = [
        ("MX", lambda p, fqn: fqn.endswith(".q_a_proj.weight")),
        ("BR", lambda p, fqn: "experts" in fqn),
    ]
    sd = {
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(2, 2),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 2),
        "lm_head.weight": torch.randn(2, 2),  # matched by neither filter -> passthrough
        "router.tid2eid": torch.zeros(3, dtype=torch.int64),  # non-float -> passthrough
    }

    out, quantized = export.quantize_state_dict(sd, plan, device="cpu")

    assert quantized == {
        "model.layers.0.self_attn.q_a_proj.weight": 0,
        "model.layers.0.mlp.experts.gate_up_proj": 1,
    }
    assert out["model.layers.0.self_attn.q_a_proj.weight"] == "Q<MX>"
    assert out["model.layers.0.mlp.experts.gate_up_proj"] == "Q<BR>"
    assert out["lm_head.weight"] is sd["lm_head.weight"]
    assert out["router.tid2eid"] is sd["router.tid2eid"]


def test_quantize_state_dict_first_matching_plan_entry_wins(export, monkeypatch):
    monkeypatch.setattr(export, "_quantize_to_mx_tensor", lambda w, swap, device: f"Q<{swap}>")
    plan = [("broad", lambda p, fqn: True), ("never", lambda p, fqn: False)]

    out, quantized = export.quantize_state_dict({"x.w": torch.randn(2, 2)}, plan, device="cpu")

    assert quantized == {"x.w": 0}
    assert out["x.w"] == "Q<broad>"


# =========================================================================
# _merge_routed_experts
# =========================================================================


def test_merge_w2_down_proj(export):
    """w2 experts merge into the 3D down_proj with per-expert values intact."""
    sd = {fqn: _mk_expert(0, 2, e, _M) for e, fqn in enumerate(_expert_fqns(0, "2"))}
    merged, merged_plan = export._merge_routed_experts(sd, dict.fromkeys(sd, 1))

    assert set(merged) == {"model.layers.0.mlp.experts.down_proj"}
    assert merged_plan == {"model.layers.0.mlp.experts.down_proj": 1}
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
    merged, _ = export._merge_routed_experts(sd, dict.fromkeys(sd, 1))

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
    merged, _ = export._merge_routed_experts(sd, dict.fromkeys(sd, 0))
    t = merged["model.layers.0.mlp.experts.down_proj"]
    assert t.qdata.ndim == 3
    assert t.pack_axis == t.qdata.ndim - 1


def test_merge_layer_fqn_mapping(export):
    """Each (layer, w) group lands in its own FQN with only that layer's values."""
    sd = {}
    for layer in (0, 7):
        for group, w, rows in ((0, "1", _N), (1, "3", _N), (2, "2", _M)):
            for e in range(_N_EXPERTS):
                sd[f"layers.{layer}.ffn.experts.{e}.w{w}.weight"] = _mk_expert(layer, group, e, rows)
    merged, merged_plan = export._merge_routed_experts(sd, dict.fromkeys(sd, 0))

    assert (
        set(merged)
        == set(merged_plan)
        == {f"model.layers.{layer}.mlp.experts.gate_up_proj" for layer in (0, 7)}
        | {f"model.layers.{layer}.mlp.experts.down_proj" for layer in (0, 7)}
    )
    for layer in (0, 7):
        t = merged[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        for e in range(_N_EXPERTS):
            expected = torch.cat([_values(layer, 0, e, _N), _values(layer, 1, e, _N)], dim=0)
            assert torch.equal(t.qdata[e].to(torch.int64), expected)


def test_merge_rejects_incomplete_experts(export):
    sd = {fqn: _mk_expert(0, 2, e, _M) for e, fqn in zip((0, 2), _expert_fqns(0, "2", experts=(0, 2)), strict=True)}
    with pytest.raises(ValueError, match=r"missing \[1\]"):
        export._merge_routed_experts(sd, dict.fromkeys(sd, 0))


@pytest.mark.parametrize("w3_experts", [(), (0,)])
def test_merge_rejects_w1_without_w3(export, w3_experts):
    sd = {fqn: _mk_expert(0, 0, e, _N) for e, fqn in enumerate(_expert_fqns(0, "1"))}
    for e in w3_experts:
        sd[f"layers.0.ffn.experts.{e}.w3.weight"] = _mk_expert(0, 1, e, _N)
    with pytest.raises(ValueError, match="w3"):
        export._merge_routed_experts(sd, dict.fromkeys(sd, 0))


def test_merge_rejects_attr_mismatch(export):
    """Experts whose quantization configs differ by value cannot be merged."""
    sd = {
        "layers.0.ffn.experts.0.w2.weight": _mk_expert(0, 2, 0, _M, cfg=_FakeQuantizeConfig(torch.int16)),
        "layers.0.ffn.experts.1.w2.weight": _mk_expert(0, 2, 1, _M),
    }
    with pytest.raises(ValueError, match="disagree"):
        export._merge_routed_experts(sd, dict.fromkeys(sd, 0))


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

    out, new_quantized = export._to_in_memory_layout(sd, dict.fromkeys(_DENSE_CASES, 0))

    for src, dst in _DENSE_CASES.items():
        assert out[dst] is sd[src]
    assert out["layers.5.attn.wo_b.bias"] is sd["layers.5.attn.wo_b.bias"]
    assert out["layers.5.rope.freqs"] is sd["layers.5.rope.freqs"]
    assert set(out) == set(_DENSE_CASES.values()) | {"layers.5.attn.wo_b.bias", "layers.5.rope.freqs"}
    assert new_quantized == dict.fromkeys(_DENSE_CASES.values(), 0)


def test_to_in_memory_layout_merges_experts_and_preserves_plain(export):
    """Quantized tensors become merged 3D experts + renamed dense; plain tensors are preserved."""
    sd = {fqn: _mk_expert(5, 0, 0, _N) for fqn in _DENSE_CASES}
    for e in range(_N_EXPERTS):
        for group, w, rows in ((0, "1", _N), (1, "3", _N), (2, "2", _M)):
            sd[f"layers.5.ffn.experts.{e}.w{w}.weight"] = _mk_expert(5, group, e, rows)
    sd["lm_head.weight"] = torch.randn(32, 8)

    out, new_quantized = export._to_in_memory_layout(sd, {k: 0 for k in sd if k != "lm_head.weight"})

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
        export._to_in_memory_layout(sd, dict.fromkeys(sd, 0))


# =========================================================================
# _copy_hf_assets
# =========================================================================


def test_copy_hf_assets_warns_on_skipped_entries(export, tmp_path, caplog):
    """Regular and symlinked files are copied; dirs and dangling symlinks each produce a warning."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "config.json").write_text("{}")
    (assets / "tokenizer_config.json").write_text("{}")
    (assets / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (assets / "model.safetensors.index.json").write_text("{}")
    (assets / "configuration.json").write_text("{}")  # load-irrelevant: skipped silently
    (assets / "tokenizer").mkdir()
    (assets / "README.md").symlink_to(assets / "config.json")
    (assets / "BROKEN.md").symlink_to(assets / "missing.txt")
    out = tmp_path / "out"
    out.mkdir()

    with caplog.at_level(logging.WARNING, logger="export_quantized_hf_under_test"):
        export._copy_hf_assets(assets, out)

    assert sorted(p.name for p in out.iterdir()) == ["README.md", "config.json", "tokenizer_config.json"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    skipped = [name for name in ("tokenizer", "BROKEN.md") if any(name in msg for msg in warnings)]
    assert skipped == ["tokenizer", "BROKEN.md"]


# =========================================================================
# _build_fqn_to_config_record / _inject_torchao_quant_config
# =========================================================================


def test_generalized_fqn_pattern(export):
    """Numeric components generalize so one key covers every layer's sibling."""
    pattern = export._generalized_fqn_pattern("model.layers.7.self_attn.q_a_proj.weight")

    assert pattern == r"re:model\.layers\.\d+\.self_attn\.q_a_proj\.weight"
    regex = pattern.removeprefix("re:")
    assert re.fullmatch(regex, "model.layers.0.self_attn.q_a_proj.weight")
    assert not re.fullmatch(regex, "model.layers.0.self_attn.q_b_proj.weight")


def test_build_fqn_to_config_record_groups_by_plan(export):
    """One deduped ``re:`` key per quantized FQN shape, valued with that group's swap config."""
    dense, routed = _FakeSwap("dense"), _FakeSwap("routed")
    quantized = {
        "model.layers.1.self_attn.q_a_proj.weight": 0,
        "model.layers.0.self_attn.q_a_proj.weight": 0,  # same shape, deduped
        "model.layers.0.mlp.experts.gate_up_proj": 1,
    }

    record = export._build_fqn_to_config_record(quantized, [(dense, None), (routed, None)])

    assert record["_type"] == "FqnToConfig"
    fqn_to_config = record["_data"]["fqn_to_config"]
    assert list(fqn_to_config) == [
        r"re:model\.layers\.\d+\.self_attn\.q_a_proj\.weight",
        r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
    ]  # plan-group order
    assert [v["_data"]["tag"] for v in fqn_to_config.values()] == ["dense", "routed"]
    # the ``__post_init__`` alias duplicates the mapping; the export strips it
    assert "module_fqn_to_config" not in record["_data"]
    json.dumps(record)  # the record must survive config.json serialization


def test_build_fqn_to_config_record_without_quantized(export):
    assert export._build_fqn_to_config_record({}, [(_FakeSwap("d"), None)]) is None


def test_build_fqn_to_config_record_survives_real_torchao_encoder(export, tmp_path):
    probe = tmp_path / "real_torchao_probe.py"
    probe.write_text(
        "import importlib.util, json, sys\n"
        "try:\n"
        "    import torchao.core.config\n"
        "    import torchao_npu.configs.param_swap as ps\n"
        "    import torchao_npu.quantization.quant_configs as qc\n"
        "except Exception as exc:\n"
        "    print('SKIP', exc)\n"
        "    sys.exit(0)\n"
        f"spec = importlib.util.spec_from_file_location('export_mod', {str(export.__file__)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "swap = ps.ParamSwapConfig(weight_config=qc.MXQuantizeConfig())\n"
        "record = mod._build_fqn_to_config_record(\n"
        "    {'model.layers.0.self_attn.q_a_proj.weight': 0}, [(swap, None)]\n"
        ")\n"
        "entry = record['_data']['fqn_to_config']['re:' + r'model\\.layers\\.\\d+\\.self_attn\\.q_a_proj\\.weight']\n"
        "assert entry['_type'] == 'ParamSwapConfig'\n"
        "assert 'params_filter_fn' not in entry['_data']\n"
        "json.dumps(record)\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=600)
    if proc.stdout.startswith("SKIP"):
        pytest.skip("real torchao stack unavailable")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "OK"


def test_inject_torchao_quant_config_writes_fqn_plan(export, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"architectures": ["XForCausalLM"]}))
    record = {"_type": "FqnToConfig", "_data": {"fqn_to_config": {r"re:a\.0\.b": {"_type": "ParamSwapConfig"}}}}

    export._inject_torchao_quant_config(out, record)

    config = json.loads((out / "config.json").read_text())
    qc = config["quantization_config"]
    # AO-official shape: one "default" entry holding the serialized FqnToConfig plan.
    assert qc["quant_type"] == {"default": record}
    assert qc["quant_method"] == "torchao"
    assert config["architectures"] == ["XForCausalLM"]
    # The static constant never carries a placeholder quant_type.
    assert "quant_type" not in export._TORCHAO_QUANTIZATION_CONFIG


def test_discard_failed_export_removes_only_this_runs_artifacts(export, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "user_notes.txt").write_text("keep")

    export._discard_failed_export(tmp_path, ["model.safetensors"])

    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "user_notes.txt").is_file()


def test_inject_torchao_quant_config_without_record_skips(export, tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}")

    with caplog.at_level(logging.WARNING, logger="export_quantized_hf_under_test"):
        export._inject_torchao_quant_config(out, None)

    assert "quantization_config" not in json.loads((out / "config.json").read_text())
    assert any("no quantized tensors" in r.getMessage() for r in caplog.records)


# =========================================================================
# _model_derived_config_fields / _apply_model_derived_config
# =========================================================================

_NS = types.SimpleNamespace


def _dsv4_like_config():
    """Nested config with the DSv4 surface: 4 layers, 2 hash layers, indexer only on cr=4."""

    def attn(ratio):
        rope = (
            _NS(theta=10000.0)
            if ratio <= 1
            else _NS(
                theta=40000.0,
                scaling="yarn",
                rope_factor=4.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_seq_len=65536,
            )
        )
        return _NS(
            n_heads=8,
            head_dim=16,
            rope_head_dim=4,
            q_lora_rank=32,
            wkv=_NS(out_features=16),
            wo_a=_NS(out_features=64, n_heads=4),
            inner_attention=_NS(window_size=128, index_topk=512),
            rope=rope,
            norm_eps=1e-6,
            compress_ratio=ratio,
            indexer=_NS(num_index_heads=8, index_head_dim=16) if ratio == 4 else None,
        )

    def moe(layer_id):
        return _NS(
            num_experts=4,
            router=_NS(
                top_k=2, layer_id=layer_id, n_hash_layers=2, score_func="sigmoid", route_scale=2.5, route_norm=True
            ),
            routed_experts=_NS(w13=_NS(out_features=128), activation_fn=_NS(swiglu_limit=10.0)),
            shared_experts=_NS(),
        )

    layers = [
        _NS(
            attention=attn(ratio),
            moe=moe(i),
            hc_attn_pre=_NS(hc_mult=4, sinkhorn_iters=20, eps=1e-6),
        )
        for i, ratio in enumerate([1, 1, 4, 128])
    ]
    return _NS(vocab_size=1000, dim=64, layers=layers, mtp_layers=[], enable_weight_tying=False)


def test_model_derived_config_fields_maps_dsv4_truth(export):
    fields = export._model_derived_config_fields(_dsv4_like_config())
    assert fields["num_hidden_layers"] == 4
    assert fields["num_nextn_predict_layers"] == 0
    assert fields["tie_word_embeddings"] is False
    assert fields["num_key_value_heads"] == 1  # wkv 16 rows / head_dim 16
    assert fields["o_lora_rank"] == 64 and fields["o_groups"] == 4
    assert fields["index_n_heads"] == 8 and fields["index_head_dim"] == 16
    assert fields["n_routed_experts"] == 4
    assert fields["num_experts_per_tok"] == 2
    assert fields["num_hash_layers"] == 2
    assert fields["moe_intermediate_size"] == 128
    assert fields["n_shared_experts"] == 1
    assert fields["scoring_func"] == "sigmoid"
    assert fields["routed_scaling_factor"] == 2.5
    assert fields["norm_topk_prob"] is True
    assert fields["hc_mult"] == 4 and fields["hc_sinkhorn_iters"] == 20 and fields["hc_eps"] == 1e-6
    # HF spells "uncompressed" 0 where the model config spells it 1
    assert fields["compress_ratios"] == [0, 0, 4, 128]
    assert fields["layer_types"] == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    # hash layers are the router-hash prefix; the rest stay plain moe
    assert fields["mlp_layer_types"] == ["hash_moe", "hash_moe", "moe", "moe"]
    assert fields["rope_theta"] == 10000.0
    assert fields["compress_rope_theta"] == 40000.0
    assert fields["max_position_embeddings"] == 262144


def test_model_derived_config_max_position_follows_release_window(export):
    cfg = _dsv4_like_config()
    for layer in cfg.layers:
        rope = layer.attention.rope
        if getattr(rope, "scaling", None) == "yarn":
            rope.rope_factor = 16.0
    fields = export._model_derived_config_fields(cfg)
    assert fields["max_position_embeddings"] == 1048576


def test_model_derived_config_fields_unknown_ratio_skips_layer_types(export):
    """A compress_ratio the HF table does not name yields no layer_types guess."""
    cfg = _NS(vocab_size=1, dim=1, mtp_layers=[], layers=[_NS(attention=_NS(compress_ratio=2))])
    fields = export._model_derived_config_fields(cfg)
    assert fields["compress_ratios"] == [2]
    assert "layer_types" not in fields


def test_model_derived_config_fields_skips_unbacked_structures(export):
    """Absent or partial moe/indexer/hc sub-configs emit only the fields they actually back."""
    cfg = _NS(vocab_size=32, dim=8, layers=[_NS(attention=_NS(n_heads=2, head_dim=4))])
    assert export._model_derived_config_fields(cfg) == {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 4,
    }

    moe = _NS(num_experts=4, shared_experts=None)
    cfg = _NS(vocab_size=32, dim=8, layers=[_NS(attention=_NS(n_heads=2, head_dim=4), moe=moe)])
    fields = export._model_derived_config_fields(cfg)
    assert fields["n_routed_experts"] == 4
    assert fields["n_shared_experts"] == 0
    assert not {"num_experts_per_tok", "moe_intermediate_size", "scoring_func"} & set(fields)


def test_apply_model_derived_config_rewrites_and_preserves(export, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["XForCausalLM"],
                "n_routed_experts": 256,
                "num_hidden_layers": 999,
                "max_position_embeddings": 1048576,
                "rope_scaling": {"type": "yarn"},
                "quantization_config": {"quant_method": "torchao"},
            }
        )
    )

    export._apply_model_derived_config(out, _dsv4_like_config())

    config = json.loads((out / "config.json").read_text())
    assert config["n_routed_experts"] == 4  # asset number replaced by the model's truth
    assert config["num_hidden_layers"] == 4
    assert config["compress_ratios"] == [0, 0, 4, 128]
    assert config["compress_rope_theta"] == 40000.0
    assert config["max_position_embeddings"] == 262144
    # fields the model config does not declare keep the asset value
    assert config["architectures"] == ["XForCausalLM"]
    assert config["quantization_config"] == {"quant_method": "torchao"}
    # a rope_scaling without yarn leaves is untouched by the leaf-level sync
    assert config["rope_scaling"] == {"type": "yarn"}
    # alt-spelling fields the asset omits (official release layout) stay absent
    assert not {"layer_types", "mlp_layer_types", "rope_parameters"} & set(config)


def test_apply_model_derived_config_syncs_carried_alt_spellings(export, tmp_path):
    """An asset carrying layer_types/mlp_layer_types/rope_parameters gets them refreshed."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps(
            {
                # stale from a 3-hash release asset over a 5-layer model
                "compress_ratios": [0, 0, 4, 128, 4],
                "layer_types": ["sliding_attention"] * 3 + ["compressed_sparse_attention"] * 2,
                "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * 2,
                "rope_parameters": {
                    "main": {"rope_type": "default", "rope_theta": 10000.0, "attention_factor": 1.0},
                    "compress": {
                        "rope_type": "yarn",
                        "rope_theta": 160000.0,
                        "factor": 16.0,
                        "beta_fast": 32.0,
                        "beta_slow": 1.0,
                        "original_max_position_embeddings": 65536,
                        "partial_rotary_factor": 0.125,
                        "attention_factor": 1.0,
                    },
                },
            }
        )
    )

    export._apply_model_derived_config(out, _dsv4_like_config())

    config = json.loads((out / "config.json").read_text())
    # same length as compress_ratios, distribution follows the model's compress ratios
    assert config["layer_types"] == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config["mlp_layer_types"] == ["hash_moe", "hash_moe", "moe", "moe"]
    compress = config["rope_parameters"]["compress"]
    assert compress["rope_theta"] == 40000.0
    assert compress["factor"] == 4.0
    assert compress["original_max_position_embeddings"] == 65536
    # fixture: rope_head_dim 4 / head_dim 16
    assert compress["partial_rotary_factor"] == 0.25
    # leaves the model config does not model keep the asset value
    assert compress["attention_factor"] == 1.0
    assert compress["beta_fast"] == 32.0
    assert config["rope_parameters"]["main"] == {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "attention_factor": 1.0,
    }
    # the stale 5-entry compress_ratios is replaced by the model's 4-layer truth
    assert config["compress_ratios"] == [0, 0, 4, 128]


def test_apply_model_derived_config_syncs_legacy_rope_scaling(export, tmp_path):
    """Official release assets spell the yarn scalars in a flat legacy ``rope_scaling``."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps(
            {
                # release values; the fixture model is the debugmodel-shaped truth
                "compress_rope_theta": 160000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16.0,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "original_max_position_embeddings": 65536,
                },
            }
        )
    )

    export._apply_model_derived_config(out, _dsv4_like_config())

    config = json.loads((out / "config.json").read_text())
    assert config["compress_rope_theta"] == 40000.0
    scaling = config["rope_scaling"]
    assert scaling["factor"] == 4.0
    assert scaling["beta_fast"] == 32.0
    assert scaling["original_max_position_embeddings"] == 65536
    # the sync rewrites modeled leaves only; spelling and unmodeled keys stay
    assert scaling["type"] == "yarn"
    assert "rope_parameters" not in config


def test_apply_model_derived_config_without_config_json_warns(export, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="export_quantized_hf_under_test"):
        export._apply_model_derived_config(tmp_path, _dsv4_like_config())
    assert any("skipping model-derived" in r.getMessage() for r in caplog.records)


# =========================================================================
# _validate_plan_against_shards
# =========================================================================

_PATTERN_DENSE = r"re:model\.layers\.\d+\.self_attn\.q_a_proj\.weight"
_PATTERN_EXPERTS = r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj"
_FQNS = (
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.1.self_attn.q_a_proj.weight",
    "model.layers.0.mlp.experts.gate_up_proj",
)


def _mx_cfg(dtype, *, scale_alg=1, block_size=32):
    """Plan-side MX record: the ``config_to_dict`` shape, ``_version`` and all."""
    return {
        "_type": "MXQuantizeConfig",
        "_version": 1,
        "_data": {
            "block_size": block_size,
            "elem_dtype": {"_type": "torch.dtype", "_data": dtype},
            "round_mode": "rint",
            "scale_alg": scale_alg,
            "dst_type_max": 0.0,
        },
    }


def _block_mx_cfg(inner):
    return {
        "_type": "BlockMXQuantizeConfig",
        "_version": 1,
        "_data": {
            "block_size": 32,
            "elem_dtype": {"_type": "torch.dtype", "_data": "float8_e4m3fn"},
            "round_mode": "rint",
            "scale_alg": 0,
            "dst_type_max": 0.0,
            "mxfp4_fake_quantize_config": inner,
            "fsdp_prequantize": False,
        },
    }


def _swap_record(weight, act):
    return {"_type": "ParamSwapConfig", "_data": {"weight_config": weight, "activation_config": act}}


def _header_mx_record(weight, act):
    """Shard-side ``MXTensor`` record: type + data only, as the flatten writer stamps it (no ``_version``)."""

    def strip(cfg):
        return None if cfg is None else {"_type": cfg["_type"], "_data": cfg["_data"]}

    return {"_type": "MXTensor", "_data": {"quant_config": strip(weight), "act_quant_config": strip(act)}}


def _write_shard(path: Path, records):
    """Header-only safetensors stand-in: the validator never reads tensor bytes."""
    meta = {fqn: json.dumps(rec) for fqn, rec in records.items()}
    meta["format"] = "pt"
    header = json.dumps({"__metadata__": meta}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def _consistent_export():
    """Plan + shard records mirroring a real debug export: fp8 dense, block-MX (mxfp4) experts."""
    fp8, fp4, act = _mx_cfg("float8_e4m3fn"), _mx_cfg("float4_e2m1fn_x2", scale_alg=2), _mx_cfg("float8_e4m3fn")
    patterns = OrderedDict(
        [
            (_PATTERN_DENSE, _swap_record(fp8, act)),
            (_PATTERN_EXPERTS, _swap_record(_block_mx_cfg(fp4), act)),
        ]
    )
    record = {"_type": "FqnToConfig", "_version": 1, "_data": {"fqn_to_config": patterns}}
    headers = {
        _FQNS[0]: _header_mx_record(_mx_cfg("float8_e4m3fn"), _mx_cfg("float8_e4m3fn")),
        _FQNS[1]: _header_mx_record(_mx_cfg("float8_e4m3fn"), _mx_cfg("float8_e4m3fn")),
        # The block path records the *inner* mxfp4 config as the tensor's quant_config.
        _FQNS[2]: _header_mx_record(_mx_cfg("float4_e2m1fn_x2", scale_alg=2), _mx_cfg("float8_e4m3fn")),
    }
    return record, headers


def _blockmx_flat_export():
    """mxfp4-QAT off: on-disk quant_config is the MX-field rebuild ``to_mx_tensor`` writes (fp8 ``scale_alg=0``)."""
    fp8, act = _mx_cfg("float8_e4m3fn"), _mx_cfg("float8_e4m3fn")
    patterns = OrderedDict(
        [
            (_PATTERN_DENSE, _swap_record(fp8, act)),
            (_PATTERN_EXPERTS, _swap_record(_block_mx_cfg(None), act)),
        ]
    )
    record = {"_type": "FqnToConfig", "_version": 1, "_data": {"fqn_to_config": patterns}}
    headers = {
        _FQNS[0]: _header_mx_record(_mx_cfg("float8_e4m3fn"), _mx_cfg("float8_e4m3fn")),
        _FQNS[1]: _header_mx_record(_mx_cfg("float8_e4m3fn"), _mx_cfg("float8_e4m3fn")),
        # mxfp4 off: derived fp8 MX config (block's scale_alg=0), not the block record.
        _FQNS[2]: _header_mx_record(_mx_cfg("float8_e4m3fn", scale_alg=0), _mx_cfg("float8_e4m3fn")),
    }
    return record, headers


def _validate(export, tmp_path, record, headers, quantized=None, shards=None):
    shard_names = shards or ["model.safetensors"]
    _write_shard(tmp_path / shard_names[0], headers)
    export._validate_plan_against_shards(
        tmp_path, shard_names, record, dict.fromkeys(quantized if quantized is not None else headers, 0)
    )


def test_validate_plan_against_shards_passes(export, tmp_path):
    """Agreeing plan and shards pass; the ``_version`` asymmetry between the two writers is tolerated."""
    record, headers = _consistent_export()
    _validate(export, tmp_path, record, headers)


def test_validate_plan_against_shards_passes_across_shards(export, tmp_path):
    """Tensor records split over shards are unioned before matching."""
    record, headers = _consistent_export()
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    halves = (dict(list(headers.items())[:2]), dict(list(headers.items())[2:]))
    for name, part in zip(shards, halves, strict=True):
        _write_shard(tmp_path / name, part)
    export._validate_plan_against_shards(tmp_path, shards, record, dict.fromkeys(headers, 0))


def test_validate_plan_against_shards_passes_blockmx_without_mxfp4(export, tmp_path):
    """mxfp4-QAT off: the plan's block-MX entry is compared against the MX config ``to_mx_tensor`` writes."""
    record, headers = _blockmx_flat_export()
    _validate(export, tmp_path, record, headers)


def test_validate_detects_blockmx_flat_drift(export, tmp_path):
    """Same path: a block-config MX field drifting from the header must still be caught."""
    record, headers = _blockmx_flat_export()
    block = record["_data"]["fqn_to_config"][_PATTERN_EXPERTS]["_data"]["weight_config"]["_data"]
    del block["dst_type_max"]
    with pytest.raises(ValueError, match=r"gate_up_proj.*dst_type_max"):
        _validate(export, tmp_path, record, headers)


def test_validate_blockmx_flat_ignores_block_only_fields(export, tmp_path):
    """The mxfp4-off rebuild compares only MX fields: block-only ``fsdp_prequantize`` must not affect the check.

    ``all_block_fp8`` sets ``fsdp_prequantize=True`` on the routed block config, so this
    is the realistic shape -- guarding against a future filter that stops dropping the
    block-only fields and silently breaks the mxfp4-off comparison.
    """
    record, headers = _blockmx_flat_export()
    block = record["_data"]["fqn_to_config"][_PATTERN_EXPERTS]["_data"]["weight_config"]["_data"]
    block["fsdp_prequantize"] = True
    _validate(export, tmp_path, record, headers)


def test_plan_weight_config_rejects_malformed_records(export):
    with pytest.raises(ValueError, match="no serialized weight_config record"):
        export._plan_weight_config({"_data": {"weight_config": "not-a-record"}})
    rec = {"_data": {"weight_config": {"_type": "Float8FakeQuantizeConfig", "_data": {}}}}
    with pytest.raises(ValueError, match="unexpected plan weight_config type"):
        export._plan_weight_config(rec)


def test_validate_detects_weight_config_drift(export, tmp_path):
    record, headers = _consistent_export()
    record["_data"]["fqn_to_config"][_PATTERN_DENSE]["_data"]["weight_config"]["_data"]["block_size"] = 64
    with pytest.raises(ValueError, match=r"q_a_proj.*block_size.*64"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_blockmx_inner_drift(export, tmp_path):
    """The comparison that must use the nested mxfp4 config: outer block-MX dtype is fp8 by design."""
    record, headers = _consistent_export()
    inner = record["_data"]["fqn_to_config"][_PATTERN_EXPERTS]["_data"]["weight_config"]["_data"]
    inner["mxfp4_fake_quantize_config"]["_data"]["elem_dtype"] = {"_type": "torch.dtype", "_data": "float8_e4m3fn"}
    with pytest.raises(ValueError, match="gate_up_proj"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_activation_drift(export, tmp_path):
    record, headers = _consistent_export()
    act = record["_data"]["fqn_to_config"][_PATTERN_DENSE]["_data"]["activation_config"]["_data"]
    act["elem_dtype"] = {"_type": "torch.dtype", "_data": "float4_e2m1fn_x2"}
    with pytest.raises(ValueError, match="activation_config"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_uncovered_tensor(export, tmp_path):
    record, headers = _consistent_export()
    headers["model.layers.0.mlp.experts.down_proj"] = headers[_FQNS[2]]
    with pytest.raises(ValueError, match=r"down_proj.*matches no plan pattern"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_stale_pattern(export, tmp_path):
    record, headers = _consistent_export()
    record["_data"]["fqn_to_config"][r"re:model\.layers\.\d+\.mlp\.experts\.down_proj"] = _swap_record(
        _mx_cfg("float8_e4m3fn"), None
    )
    with pytest.raises(ValueError, match=r"down_proj.*matches no quantized tensor"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_shadowed_pattern(export, tmp_path):
    """Upstream applies only the first matching pattern; a fully-shadowed later pattern misdescribes the plan."""
    record, headers = _consistent_export()
    broad = r"re:model\.layers\.\d+\.self_attn\.q_a_proj\..+"  # same tensors as _PATTERN_DENSE, inserted later
    record["_data"]["fqn_to_config"][broad] = _swap_record(_mx_cfg("float4_e2m1fn_x2"), None)
    with pytest.raises(ValueError, match="never wins"):
        _validate(export, tmp_path, record, headers)


def test_validate_detects_tensor_missing_from_shards(export, tmp_path):
    """A tensor the plan covers but the shard writer dropped."""
    record, headers = _consistent_export()
    headers.pop(_FQNS[1])
    with pytest.raises(ValueError, match="missing from shards"):
        _validate(export, tmp_path, record, headers, quantized=_FQNS)
