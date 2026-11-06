# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list

E2M1_LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])

Q_A_FQN = "model.layers.0.self_attn.q_a_proj.weight"
SHARED_W1_FQN = "model.layers.0.mlp.shared_experts.gate_proj.weight"
GATE_UP_FQN = "model.layers.0.mlp.experts.gate_up_proj"
DOWN_FQN = "model.layers.0.mlp.experts.down_proj"


@pytest.fixture
def readers(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "torchtitan_npu/extensions/mx_storage_reader"
    package = ModuleType("_ao_test_components")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = {}
    for name in ("mx_dequant_backend", "hf_storage", "ao_hf_storage"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


@pytest.fixture
def cpu_npu_boundary(monkeypatch):
    calls = []

    def anti_quant(qdata, scale, *, axis, dst_type, src_type):
        calls.append((qdata.shape, scale.shape, axis, dst_type, src_type))
        if qdata.dtype == torch.uint8:
            lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
            values = lut[torch.stack((qdata & 15, qdata >> 4), dim=-1).long()].flatten(-2)
        else:
            values = qdata.float()
        scales = torch.exp2(scale.flatten(-2).float() - 127).repeat_interleave(32, -1)
        return values * scales[..., : values.shape[-1]]

    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    device_module = SimpleNamespace(
        Stream=lambda **kw: stream,
        Event=lambda: event,
        current_stream=lambda device: stream,
        stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            float4_e2m1fn_x2=object(),
            npu_anti_mx_quant=anti_quant,
        ),
    )
    return calls


def _mx_entry(
    elem_dtype: str, data_names: bool = True, quant_axis: int | None = 1, pack_axis: int | None = None
) -> str:
    entry = {
        "_type": "MXTensor",
        "_data": {
            "pack_axis": pack_axis,
            "orig_dtype": {"_type": "torch.dtype", "_data": "bfloat16"},
            "quant_axis": quant_axis,
            "quant_config": {
                "_type": "MXQuantizeConfig",
                "_data": {
                    "block_size": 32,
                    "elem_dtype": {"_type": "torch.dtype", "_data": elem_dtype},
                    "round_mode": "rint",
                    "scale_alg": 1,
                    "dst_type_max": 0.0,
                },
            },
            "act_quant_config": None,
        },
    }
    if data_names:
        entry["_tensor_data_names"] = ["qdata", "scale"]
    return json.dumps(entry)


def _f4(byte_values: torch.Tensor) -> torch.Tensor:
    return byte_values.contiguous().view(torch.float4_e2m1fn_x2)


def _f4_bytes(*shape: int) -> torch.Tensor:
    return _f4(torch.zeros(*shape, dtype=torch.uint8))


def _write_shard(path: Path, tensors: dict, entries: dict[str, str]) -> None:
    save_file(tensors, str(path), metadata={**entries, "tensor_names": json.dumps(sorted(entries))})


def _unpack_f4(byte_values: torch.Tensor) -> torch.Tensor:
    return E2M1_LUT[torch.stack((byte_values & 15, byte_values >> 4), dim=-1).long()].flatten(-2)


def _expand_scale(scale: torch.Tensor) -> torch.Tensor:
    flat = scale.reshape(*scale.shape[:-2], -1)
    return torch.exp2(flat.float() - 127).repeat_interleave(32, dim=-1)


def _build_ao_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Write a miniature AO-protocol checkpoint; returns expected dequantized values per FQN."""
    fp8_qdata = torch.ones(4, 96).to(torch.float8_e4m3fn)
    fp8_scale = torch.tensor([[126, 127], [128, 127]], dtype=torch.uint8).repeat(4, 1, 1)
    shared_bytes = torch.full((2, 32), 0xE2, dtype=torch.uint8)
    shared_scale = torch.full((2, 1, 2), 127, dtype=torch.uint8)
    gu_bytes = torch.tensor([[0xE2, 0xE2, 0x2E, 0x2E], [0xE2, 0x2E, 0xE2, 0x2E]], dtype=torch.uint8)
    gu_bytes = gu_bytes[:, :, None].expand(2, 4, 64).contiguous()
    gu_scale = (127 - torch.arange(2, dtype=torch.uint8)[:, None, None, None]).expand(2, 4, 2, 2).contiguous()
    down_bytes = torch.full((2, 2, 32), 0x2E, dtype=torch.uint8)
    down_scale = (127 + torch.arange(2, dtype=torch.uint8)[:, None, None, None]).expand(2, 2, 1, 2).contiguous()
    embed = torch.tensor([1.5, 2.5, 3.5], dtype=torch.bfloat16)
    tid2eid = torch.tensor([[7, 8]], dtype=torch.int64)
    hc_scale = torch.zeros(3, dtype=torch.bfloat16)

    # Axes mirror what the exporter writes: stacked 3D experts re-base both axes onto dim 2.
    entries = {
        Q_A_FQN: _mx_entry("float8_e4m3fn"),
        SHARED_W1_FQN: _mx_entry("float4_e2m1fn_x2", pack_axis=1),
        GATE_UP_FQN: _mx_entry("float4_e2m1fn_x2", quant_axis=2, pack_axis=2),
        DOWN_FQN: _mx_entry("float4_e2m1fn_x2", quant_axis=2, pack_axis=2),
        # Identity entries carry no physical data and must be skipped.
        "model.layers.0.self_attn.o_a_proj.weight": json.dumps({"_type": "Identity"}),
    }
    _write_shard(
        path / "model_00001.safetensors",
        {
            "model.layers.0.self_attn.q_a_proj._weight_qdata": fp8_qdata,
            "model.layers.0.self_attn.q_a_proj._weight_scale": fp8_scale,
            "embed.weight": embed,
            "layers.0.ffn.gate.tid2eid": tid2eid,
            # Plain tensor whose name ends in `_scale`: must pass through untouched.
            "hc_head_scale": hc_scale,
        },
        entries,
    )
    _write_shard(
        path / "model_00002.safetensors",
        {
            "model.layers.0.mlp.shared_experts.gate_proj._weight_qdata": _f4(shared_bytes),
            "model.layers.0.mlp.shared_experts.gate_proj._weight_scale": shared_scale,
            "model.layers.0.mlp.experts._gate_up_proj_qdata": _f4(gu_bytes),
            "model.layers.0.mlp.experts._gate_up_proj_scale": gu_scale,
            "model.layers.0.mlp.experts._down_proj_qdata": _f4(down_bytes),
            "model.layers.0.mlp.experts._down_proj_scale": down_scale,
        },
        entries,
    )

    wq_a = fp8_qdata.float() * _expand_scale(fp8_scale)[..., :96]
    shared = _unpack_f4(shared_bytes) * _expand_scale(shared_scale)[..., :64]
    gate_up = _unpack_f4(gu_bytes) * _expand_scale(gu_scale)[..., :128]
    down = _unpack_f4(down_bytes) * _expand_scale(down_scale)[..., :64]
    return {
        "layers.0.attn.wq_a.weight": wq_a,
        "layers.0.ffn.shared_experts.w1.weight": shared,
        "layers.0.ffn.experts.0.w1.weight": gate_up[0, :2],
        "layers.0.ffn.experts.0.w3.weight": gate_up[0, 2:],
        "layers.0.ffn.experts.1.w1.weight": gate_up[1, :2],
        "layers.0.ffn.experts.1.w3.weight": gate_up[1, 2:],
        "layers.0.ffn.experts.0.w2.weight": down[0],
        "layers.0.ffn.experts.1.w2.weight": down[1],
        "embed.weight": embed,
        "layers.0.ffn.gate.tid2eid": tid2eid,
        "hc_head_scale": hc_scale,
    }


MX_LOGICAL_KEYS = {
    "layers.0.attn.wq_a.weight",
    "layers.0.ffn.shared_experts.w1.weight",
    *(f"layers.0.ffn.experts.{e}.w{w}.weight" for e in range(2) for w in (1, 2, 3)),
}


def test_read_metadata_presents_logical_fp32_tensors(readers, tmp_path):
    _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    assert reader.detect()

    state_dict_metadata = reader.read_metadata().state_dict_metadata

    assert set(reader._regions) == MX_LOGICAL_KEYS
    for key in MX_LOGICAL_KEYS:
        assert state_dict_metadata[key].properties.dtype == torch.float32
    assert state_dict_metadata["layers.0.attn.wq_a.weight"].size == (4, 96)
    assert state_dict_metadata["layers.0.ffn.shared_experts.w1.weight"].size == (2, 64)
    assert state_dict_metadata["layers.0.ffn.experts.1.w1.weight"].size == (2, 128)
    assert state_dict_metadata["layers.0.ffn.experts.0.w2.weight"].size == (2, 64)
    # Plain tensors keep their physical dtype; the Identity entry is skipped.
    assert state_dict_metadata["embed.weight"].properties.dtype == torch.bfloat16
    assert state_dict_metadata["hc_head_scale"].properties.dtype == torch.bfloat16
    assert state_dict_metadata["layers.0.ffn.gate.tid2eid"].properties.dtype == torch.int64
    assert "layers.0.attn.wo_a.weight" not in state_dict_metadata
    assert not [
        key
        for key in state_dict_metadata
        if key.startswith("model.layers") or "_qdata" in key or key.endswith("_weight_scale")
    ]


def test_dcp_load_dequantizes_all_tensors(readers, cpu_npu_boundary, tmp_path):
    expected = _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    targets = {
        "layers.0.attn.wq_a.weight": torch.empty(4, 96),
        "layers.0.ffn.shared_experts.w1.weight": torch.empty(2, 64),
        **{
            f"layers.0.ffn.experts.{e}.w{w}.weight": torch.empty(2, s)
            for e in range(2)
            for w, s in ((1, 128), (2, 64), (3, 128))
        },
        "embed.weight": torch.empty(3, dtype=torch.bfloat16),
        "layers.0.ffn.gate.tid2eid": torch.empty(1, 2, dtype=torch.int64),
        "hc_head_scale": torch.zeros(3, dtype=torch.bfloat16),
    }

    dcp.load(targets, storage_reader=reader)

    for key in MX_LOGICAL_KEYS:
        torch.testing.assert_close(targets[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(targets["embed.weight"], expected["embed.weight"], rtol=0, atol=0)
    torch.testing.assert_close(targets["hc_head_scale"], expected["hc_head_scale"], rtol=0, atol=0)
    assert torch.equal(targets["layers.0.ffn.gate.tid2eid"], expected["layers.0.ffn.gate.tid2eid"])
    assert len(cpu_npu_boundary) == len(MX_LOGICAL_KEYS)
    assert all(call[2:4] == (-1, torch.float32) for call in cpu_npu_boundary)
    fp4_src = sys.modules["torch_npu"].float4_e2m1fn_x2
    assert sum(call[4] is fp4_src for call in cpu_npu_boundary) == len(MX_LOGICAL_KEYS) - 1


class _ListPlanner:
    """Minimal duck-typed LoadPlanner: resolve from a dict, record commits."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state
        self.committed: list[str] = []

    def resolve_tensor(self, request):
        return self.state[request.storage_index.fqn]

    def commit_tensor(self, request, tensor) -> None:
        self.committed.append(request.storage_index.fqn)


def _whole_tensor_plan(metadata, *fqns: str) -> LoadPlan:
    items = []
    for fqn in fqns:
        tensor_md = metadata.state_dict_metadata[fqn]
        chunk = ChunkStorageMetadata(torch.Size((0,) * len(tensor_md.size)), tensor_md.size)
        items.extend(create_read_items_for_chunk_list(fqn, tensor_md, [chunk]))
    return LoadPlan(items)


def test_plain_targets_may_sit_on_other_devices(readers, cpu_npu_boundary, tmp_path):
    expected = _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    # Plain tensors off the MX device (e.g. a CPU int64 index table) must not drag
    # the whole plan into the one-device-per-rank check; only MX targets must agree.
    targets = {
        "layers.0.attn.wq_a.weight": torch.empty(4, 96),
        "embed.weight": torch.empty(3, dtype=torch.bfloat16, device="meta"),
        "layers.0.ffn.gate.tid2eid": torch.empty(1, 2, dtype=torch.int64, device="meta"),
    }
    planner = _ListPlanner(targets)

    reader.read_data(_whole_tensor_plan(metadata, *targets), planner).wait()

    assert set(planner.committed) == set(targets)
    wq_a = targets["layers.0.attn.wq_a.weight"]
    torch.testing.assert_close(wq_a, expected["layers.0.attn.wq_a.weight"], rtol=0, atol=0)


def test_mx_device_conflict_lists_fqns(readers, tmp_path):
    _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    targets = {
        "layers.0.attn.wq_a.weight": torch.empty(4, 96),
        "layers.0.ffn.shared_experts.w1.weight": torch.empty(2, 64, device="meta"),
    }

    with pytest.raises(ValueError, match="one NPU device per rank") as excinfo:
        reader.read_data(_whole_tensor_plan(metadata, *targets), _ListPlanner(targets))

    message = str(excinfo.value)
    assert "layers.0.attn.wq_a.weight" in message
    assert "layers.0.ffn.shared_experts.w1.weight" in message


def test_read_metadata_caches_the_scan_across_resets(readers, cpu_npu_boundary, monkeypatch, tmp_path):
    _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))

    scans: list[str] = []
    real_read = readers.ao_hf_storage._read_raw_header
    monkeypatch.setattr(
        readers.ao_hf_storage,
        "_read_raw_header",
        lambda path: (scans.append(path), real_read(path))[1],
    )

    first = reader.read_metadata()
    assert len(scans) == 2  # one header read per shard

    # The adapter's eager call and dcp.load's internal one must not re-parse.
    assert reader.read_metadata() is first
    assert len(scans) == 2

    reader.reset()  # same path, fresh load_id: the scan stays valid
    revalidated = reader.read_metadata()
    assert revalidated is first
    assert revalidated.storage_meta.load_id == reader.load_id
    assert len(scans) == 2

    # dcp.load over a pre-warmed reader still dequantizes correctly.
    targets = {"layers.0.attn.wq_a.weight": torch.empty(4, 96)}
    dcp.load(targets, storage_reader=reader)
    assert len(scans) == 2

    other = tmp_path / "other"
    other.mkdir()
    _build_ao_checkpoint(other)
    reader.reset(other)  # different checkpoint: the cache must drop
    assert reader.read_metadata() is not first
    assert len(scans) == 4


class WindowPlanner(DefaultLoadPlanner):
    def __init__(self, windows: dict[str, tuple[tuple[int, int], tuple[int, int]]]):
        super().__init__()
        self.windows = windows

    def lookup_tensor(self, index):
        return self.state_dict[index.fqn]

    def create_local_plan(self):
        items = []
        for fqn, (offsets, sizes) in self.windows.items():
            region = ChunkStorageMetadata(torch.Size(offsets), torch.Size(sizes))
            items.extend(create_read_items_for_chunk_list(fqn, self.metadata.state_dict_metadata[fqn], [region]))
        return LoadPlan(items)


def test_dcp_load_windowed_slices_match_full_dequant(readers, cpu_npu_boundary, tmp_path):
    expected = _build_ao_checkpoint(tmp_path)
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    # Targets are window-sized: create_read_items_for_chunk_list emits
    # window-relative dest offsets, as the DefaultLoadPlanner narrows by them.
    windows = {
        # Odd fp8 column window inside the first scale pair.
        "layers.0.attn.wq_a.weight": ((1, 7), (2, 41)),
        # F4 window starting mid-pair on a row band that needs expert+row offsets.
        "layers.0.ffn.experts.1.w1.weight": ((1, 70), (1, 34)),
    }
    targets = {fqn: torch.empty(*sizes) for fqn, (_, sizes) in windows.items()}

    dcp.load(targets, storage_reader=reader, planner=WindowPlanner(windows))

    for fqn, ((row0, col0), (rows, cols)) in windows.items():
        row_end, col_end = row0 + rows, col0 + cols
        torch.testing.assert_close(
            targets[fqn],
            expected[fqn][row0:row_end, col0:col_end],
            rtol=0,
            atol=0,
        )
    assert len(cpu_npu_boundary) == 2


def test_ao_scale_in_a_separate_shard_loads(readers, cpu_npu_boundary, tmp_path):
    # Greedy shard packing can put an MXTensor's qdata and scale in different shards.
    qdata = torch.ones(2, 64).to(torch.float8_e4m3fn)
    scale = torch.full((2, 1, 2), 127, dtype=torch.uint8)
    prefix = "model.layers.0.self_attn.q_a_proj._weight_"
    entries = {Q_A_FQN: _mx_entry("float8_e4m3fn")}
    _write_shard(tmp_path / "model_00001.safetensors", {prefix + "qdata": qdata}, entries)
    _write_shard(tmp_path / "model_00002.safetensors", {prefix + "scale": scale}, entries)

    targets = {"layers.0.attn.wq_a.weight": torch.empty(2, 64)}
    dcp.load(targets, storage_reader=readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path)))

    torch.testing.assert_close(targets["layers.0.attn.wq_a.weight"], qdata.float(), rtol=0, atol=0)


def test_spilled_scale_shard_opens_once_per_load(readers, cpu_npu_boundary, monkeypatch, tmp_path):
    # Two tensors whose scales spill into the same shard must share one handle per load.
    entries = {
        Q_A_FQN: _mx_entry("float8_e4m3fn"),
        "model.layers.0.self_attn.q_b_proj.weight": _mx_entry("float8_e4m3fn"),
    }
    scale = torch.full((2, 1, 2), 127, dtype=torch.uint8)
    _write_shard(
        tmp_path / "model_00001.safetensors",
        {
            "model.layers.0.self_attn.q_a_proj._weight_qdata": torch.ones(2, 64).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.q_b_proj._weight_qdata": torch.full((2, 64), 2.0).to(torch.float8_e4m3fn),
        },
        entries,
    )
    _write_shard(
        tmp_path / "model_00002.safetensors",
        {
            "model.layers.0.self_attn.q_a_proj._weight_scale": scale,
            "model.layers.0.self_attn.q_b_proj._weight_scale": scale.clone(),
        },
        entries,
    )

    opens: list[str] = []
    real_open = readers.ao_hf_storage.safe_open

    def counting_open(*args, **kwargs):
        opens.append(str(args[0] if args else kwargs["filename"]))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(readers.ao_hf_storage, "safe_open", counting_open)

    targets = {
        "layers.0.attn.wq_a.weight": torch.empty(2, 64),
        "layers.0.attn.wq_b.weight": torch.empty(2, 64),
    }
    dcp.load(targets, storage_reader=readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path)))

    assert opens.count(str(tmp_path / "model_00002.safetensors")) == 1
    torch.testing.assert_close(targets["layers.0.attn.wq_a.weight"], torch.ones(2, 64), rtol=0, atol=0)
    torch.testing.assert_close(targets["layers.0.attn.wq_b.weight"], torch.full((2, 64), 2.0), rtol=0, atol=0)


def test_shard_without_ao_header_warns(readers, caplog, tmp_path):
    _build_ao_checkpoint(tmp_path)
    # An AO stamp with zero entries is valid; only a wholly unstamped shard is skipped.
    save_file({"plain.weight": torch.ones(2, 2)}, str(tmp_path / "model_00003.safetensors"))

    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    with caplog.at_level(logging.WARNING):
        metadata = reader.read_metadata()

    assert "model_00003.safetensors" in caplog.text
    assert "layers.0.attn.wq_a.weight" in metadata.state_dict_metadata
    assert "plain.weight" not in metadata.state_dict_metadata


def test_remote_checkpoint_paths_rejected(readers, tmp_path):
    reader = readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path))
    for url in ("s3://bucket/ckpt", "gs://bucket/ckpt", "file:///tmp/ckpt"):
        with pytest.raises(ValueError, match="local filesystem"):
            readers.ao_hf_storage.MxAOHuggingFaceStorageReader(url)
        with pytest.raises(ValueError, match="local filesystem"):
            reader.reset(url)


def test_detect_rejects_other_checkpoint_formats(readers, tmp_path):
    mx3 = tmp_path / "mx3"
    mx3.mkdir()
    save_file(
        {
            "linear.weight": torch.ones(1, 32).to(torch.float8_e4m3fn),
            "linear.scale": torch.full((1, 1), 127, dtype=torch.uint8),
        },
        str(mx3 / "model.safetensors"),
    )
    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    save_file({"embed.weight": torch.ones(3, dtype=torch.bfloat16)}, str(bf16 / "model.safetensors"))
    plain_ao = tmp_path / "plain-ao"
    plain_ao.mkdir()
    _write_shard(
        plain_ao / "model.safetensors",
        {"embed.weight": torch.ones(3, dtype=torch.bfloat16)},
        {"embed.weight": json.dumps({"_type": "Identity"})},
    )

    for path in (mx3, bf16, plain_ao):
        assert not readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(path)).detect(), path


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown-name",
        "missing-scale",
        "wrong-pairs",
        "dtype-mismatch",
        "scale-dtype",
        "gate-up-odd",
        "gate-up-2d",
        "dense-3d",
        "no-data-names",
        "quant-axis",
        "pack-axis",
    ],
)
def test_read_metadata_rejects_malformed_entries(readers, tmp_path, invalid):
    fp8_prefix = "model.layers.0.self_attn.q_a_proj._weight_"
    expert_prefix = "model.layers.0.mlp.experts._gate_up_proj_"
    fp8_pair = {
        fp8_prefix + "qdata": torch.ones(2, 64).to(torch.float8_e4m3fn),
        fp8_prefix + "scale": torch.full((2, 1, 2), 127, dtype=torch.uint8),
    }
    fp8 = _mx_entry("float8_e4m3fn")
    f4 = _mx_entry("float4_e2m1fn_x2")
    f4_3d = _mx_entry("float4_e2m1fn_x2", quant_axis=2, pack_axis=2)
    unknown_prefix = "model.layers.0.self_attn.unknown_proj._weight_"
    cases = {
        "unknown-name": (
            "model.layers.0.self_attn.unknown_proj.weight",
            {unknown_prefix + k.removeprefix(fp8_prefix): v for k, v in fp8_pair.items()},
            fp8,
            "Unsupported AO quantized tensor name",
        ),
        "missing-scale": (Q_A_FQN, {fp8_prefix + "qdata": fp8_pair[fp8_prefix + "qdata"]}, fp8, "absent on disk"),
        "wrong-pairs": (
            Q_A_FQN,
            {**fp8_pair, fp8_prefix + "scale": torch.full((2, 2, 2), 127, dtype=torch.uint8)},
            fp8,
            "scale pairs",
        ),
        "dtype-mismatch": (Q_A_FQN, fp8_pair, f4, "disagrees"),
        "scale-dtype": (Q_A_FQN, {**fp8_pair, fp8_prefix + "scale": torch.ones(2, 1, 2)}, fp8, "0 scales"),
        "gate-up-odd": (
            GATE_UP_FQN,
            {
                expert_prefix + "qdata": _f4_bytes(1, 3, 32),
                expert_prefix + "scale": torch.full((1, 3, 1, 2), 127, dtype=torch.uint8),
            },
            f4_3d,
            "not even",
        ),
        "gate-up-2d": (
            GATE_UP_FQN,
            {
                expert_prefix + "qdata": _f4_bytes(3, 32),
                expert_prefix + "scale": torch.full((3, 1, 2), 127, dtype=torch.uint8),
            },
            f4,
            "must be 3D",
        ),
        "dense-3d": (
            Q_A_FQN,
            {
                fp8_prefix + "qdata": _f4_bytes(2, 2, 32),
                fp8_prefix + "scale": torch.full((2, 2, 1, 2), 127, dtype=torch.uint8),
            },
            f4_3d,
            "must be 2D",
        ),
        "no-data-names": (Q_A_FQN, fp8_pair, _mx_entry("float8_e4m3fn", data_names=False), "lacks _tensor_data_names"),
        # quant_axis=0 is MXTensor's matmul-right-operand layout (-2 quantized); scale
        # geometry alone cannot tell it apart from a last-dim tensor, so axes are checked.
        "quant-axis": (Q_A_FQN, fp8_pair, _mx_entry("float8_e4m3fn", quant_axis=0), "quant_axis"),
        "pack-axis": (Q_A_FQN, fp8_pair, _mx_entry("float4_e2m1fn_x2", pack_axis=0), "pack_axis"),
    }

    fqn, tensors, entry, message = cases[invalid]
    _write_shard(tmp_path / "model.safetensors", tensors, {fqn: entry})

    with pytest.raises(ValueError, match=message):
        readers.ao_hf_storage.MxAOHuggingFaceStorageReader(str(tmp_path)).read_metadata()


def test_only_v4_selects_ao_reader(readers, monkeypatch, tmp_path):
    fallback = object()

    class V3Adapter:
        def get_hf_storage_reader(self, path, from_quantized=False):
            return fallback

    upstream = ModuleType("torchtitan.models.deepseek_v3.state_dict_adapter")
    upstream.DeepSeekV3StateDictAdapter = V3Adapter
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    monkeypatch.setitem(
        sys.modules,
        "_ao_test_components.model",
        SimpleNamespace(
            DeepSeekV4Model=SimpleNamespace(Config=object),
        ),
    )

    class V4Adapter(V3Adapter):
        pass

    upstream_v4 = ModuleType("torchtitan.models.deepseek_v4.state_dict_adapter")
    upstream_v4.DeepSeekV4StateDictAdapter = V4Adapter
    monkeypatch.setitem(sys.modules, upstream_v4.__name__, upstream_v4)

    class PEFTMixin:
        pass

    monkeypatch.setitem(
        sys.modules,
        "torchtitan_npu.models.deepseek_v4.lora",
        SimpleNamespace(DeepSeekPEFTAdapter=PEFTMixin, is_lora_parameter=lambda _name: False),
    )
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.mx_storage_reader.hf_storage", readers.hf_storage)
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.mx_storage_reader.ao_hf_storage", readers.ao_hf_storage)
    root = Path(__file__).resolve().parents[4] / "torchtitan_npu/models"
    spec = importlib.util.spec_from_file_location(
        "_ao_test_components.deepseek_v4", root / "deepseek_v4/state_dict_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    v4 = module.DeepSeekV4StateDictAdapter

    ao_dir = tmp_path / "ao"
    ao_dir.mkdir()
    _build_ao_checkpoint(ao_dir)
    mx3_dir = tmp_path / "mx3"
    mx3_dir.mkdir()
    save_file(
        {
            "linear.weight": torch.ones(1, 32).to(torch.float8_e4m3fn),
            "linear.scale": torch.full((1, 1), 127, dtype=torch.uint8),
        },
        str(mx3_dir / "model.safetensors"),
    )
    bf16_dir = tmp_path / "bf16"
    bf16_dir.mkdir()
    save_file({"embed.weight": torch.ones(3, dtype=torch.bfloat16)}, str(bf16_dir / "model.safetensors"))

    reader = object.__new__(v4).get_hf_storage_reader(str(ao_dir), from_quantized=True)
    assert isinstance(reader, readers.ao_hf_storage.MxAOHuggingFaceStorageReader)
    assert reader._regions
    reader = object.__new__(v4).get_hf_storage_reader(str(mx3_dir), from_quantized=True)
    assert isinstance(reader, readers.hf_storage.MXHuggingFaceStorageReader)
    assert object.__new__(v4).get_hf_storage_reader(str(bf16_dir), from_quantized=True) is fallback
    assert object.__new__(v4).get_hf_storage_reader(str(ao_dir), from_quantized=False) is fallback
