# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint._hf_utils import CUSTOM_METADATA_KEY, SAVED_OFFSETS_KEY
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list


@pytest.fixture
def reader_module(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "torchtitan_npu/extensions/mx_storage_reader"
    package = ModuleType("_mx_test_components")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ("mx_dequant_backend", "hf_storage", "ao_hf_storage"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.hf_storage"]


@pytest.fixture
def ao_module(reader_module):
    return sys.modules["_mx_test_components.ao_hf_storage"]


@pytest.fixture
def cpu_npu_boundary(monkeypatch):
    calls = []

    def anti_quant(qdata, scale, *, axis, dst_type, src_type):
        calls.append((qdata.shape, scale.shape, axis, dst_type, src_type))
        if qdata.dtype == torch.uint8:
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
            values = lut[torch.stack((qdata & 15, qdata >> 4), dim=-1).long()].flatten(-2)
        else:
            values = qdata.float()
        scales = torch.exp2(scale.flatten(-2).float() - 127).repeat_interleave(32, -1)
        return values * scales[..., :values.shape[-1]]

    event = SimpleNamespace(record=lambda stream: None, synchronize=lambda: None)
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    device_module = SimpleNamespace(
        Stream=lambda **kw: stream, Event=lambda: event,
        current_stream=lambda device: stream, stream=lambda stream: nullcontext(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        float4_e2m1fn_x2=object(), npu_anti_mx_quant=anti_quant,
    ))
    return calls


class SlicePlanner(DefaultLoadPlanner):
    def lookup_tensor(self, index):
        return self.state_dict[index.fqn]

    def create_local_plan(self):
        region = ChunkStorageMetadata(torch.Size((0, 7)), torch.Size((2, 89)))
        return LoadPlan(create_read_items_for_chunk_list(
            "linear.weight", self.metadata.state_dict_metadata["linear.weight"], [region],
        ))


@pytest.mark.parametrize("fp4", [False, True], ids=["mxfp8", "mxfp4"])
@pytest.mark.parametrize("sliced", [False, True], ids=["full", "odd-offset"])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu], ids=["uint8-scale", "e8m0-scale"])
def test_dcp_load_sharded_mx_weights(reader_module, cpu_npu_boundary, tmp_path, fp4, sliced, scale_dtype):
    # 0xE2 stores +1 and -4, low nibble first. Three blocks exercise scale padding.
    qdata = torch.full((2, 48), 0xE2, dtype=torch.uint8) if fp4 else torch.ones((2, 96)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[126, 127, 128], [129, 128, 127]], dtype=torch.uint8).view(scale_dtype)
    split = 16 if fp4 else 32
    for i, (start, end) in enumerate(((0, split), (split, 2 * split), (2 * split, qdata.shape[-1]))):
        save_file({"linear.weight": qdata[:, start:end].contiguous()}, tmp_path / f"weight-{i}.safetensors",
                  metadata={CUSTOM_METADATA_KEY: json.dumps({"linear.weight": {SAVED_OFFSETS_KEY: [0, start]}})})
    for i, (start, end) in enumerate(((0, 1), (1, 3))):
        save_file({"linear.scale": scale[:, start:end].contiguous()}, tmp_path / f"scale-{i}.safetensors",
                  metadata={CUSTOM_METADATA_KEY: json.dumps({"linear.scale": {SAVED_OFFSETS_KEY: [0, start]}})})
    save_file({"norm.weight": torch.tensor([1., 2.])}, tmp_path / "norm.safetensors")
    reader = reader_module.MXHuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    assert metadata.state_dict_metadata["linear.weight"].size == (2, 96)
    assert "linear.scale" not in metadata.state_dict_metadata
    assert set(reader._mx_tensors) == {"linear.weight"}

    target = {"linear.weight": torch.empty((2, 89 if sliced else 96), dtype=torch.bfloat16)}
    if not sliced:
        target["norm.weight"] = torch.empty(2)
    dcp.load(target, storage_reader=reader, planner=SlicePlanner() if sliced else None)

    values = torch.tensor([1., -4.]).repeat(2, 48) if fp4 else torch.ones(2, 96)
    expected = values * torch.tensor([[.5, 1., 2.], [4., 2., 1.]]).repeat_interleave(32, -1)
    expected_loaded = expected[:, 7:] if sliced else expected
    torch.testing.assert_close(target["linear.weight"], expected_loaded.bfloat16(), rtol=0, atol=0)
    if not sliced:
        torch.testing.assert_close(target["norm.weight"], torch.tensor([1., 2.]))
    assert all(call[2:4] == (-1, torch.float32) for call in cpu_npu_boundary)
    expected_src = sys.modules["torch_npu"].float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn
    assert all(call[4] == expected_src and call[1][-1] == 2 for call in cpu_npu_boundary)


@pytest.mark.parametrize("invalid", ["dtype", "shape"])
def test_mx_scale_validation(reader_module, tmp_path, invalid):
    tensors = {"linear.weight": torch.ones((2, 32), dtype=torch.uint8)}
    if invalid != "missing":
        scale_dtype = torch.float32 if invalid == "dtype" else torch.uint8
        tensors["linear.scale"] = torch.ones((2, 1 if invalid == "shape" else 2), dtype=scale_dtype)
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="scale"):
        reader_module.MXHuggingFaceStorageReader(str(tmp_path)).read_metadata()


def test_non_mx_quantized_weight_without_scale_falls_back(reader_module, tmp_path):
    save_file({"linear.weight": torch.ones((2, 32), dtype=torch.uint8)}, tmp_path / "model.safetensors")
    reader = reader_module.MXHuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    assert not reader._mx_tensors
    assert metadata.state_dict_metadata["linear.weight"].properties.dtype == torch.uint8


def test_dcp_load_mixed_mx_formats(reader_module, cpu_npu_boundary, tmp_path):
    save_file({
        "fp4.weight": torch.full((1, 16), 0xE2, dtype=torch.uint8),
        "fp4.scale": torch.full((1, 1), 127, dtype=torch.uint8),
        "fp8.weight": torch.full((1, 32), 2.).to(torch.float8_e4m3fn),
        "fp8.scale": torch.full((1, 1), 126, dtype=torch.uint8),
        "buffer": torch.tensor([7], dtype=torch.uint8),
    }, tmp_path / "model.safetensors")
    target = {"fp4.weight": torch.empty(1, 32), "fp8.weight": torch.empty(1, 32),
              "buffer": torch.empty(1, dtype=torch.uint8)}

    dcp.load(target, storage_reader=reader_module.MXHuggingFaceStorageReader(str(tmp_path)))

    torch.testing.assert_close(target["fp4.weight"], torch.tensor([1., -4.]).repeat(1, 16), rtol=0, atol=0)
    torch.testing.assert_close(target["fp8.weight"], torch.ones(1, 32), rtol=0, atol=0)
    assert target["buffer"].item() == 7


_EXPORT_SCRIPT = (
    Path(__file__).resolve().parents[4] / "torchtitan_npu/tools/checkpoint_conversion/export_quantized_hf.py"
)


@pytest.fixture(scope="module")
def export_module():
    """Execute the export script for its tables, leaving global import state untouched.

    Loading the script runs ``_bootstrap_sys_path``, which appends the repo's
    ``experiments/torchao-npu`` tree to ``sys.path``; leaked, that flips other
    tests' ``importorskip("torchao_npu")`` from skip to fail depending on order.
    ``MonkeyPatch.context`` is the module-scope stand-in for the function fixture.
    """
    root = _EXPORT_SCRIPT.parent
    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setattr(sys, "path", list(sys.path))
        monkey = ModuleType("_export_tables_under_test")
        monkey.__path__ = [str(root)]
        spec = importlib.util.spec_from_file_location(f"{monkey.__name__}.export_quantized_hf", _EXPORT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        mpatch.setitem(sys.modules, monkey.__name__, monkey)
        mpatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        yield module


def _sample(pattern: str) -> str:
    """A concrete FQN a regex pattern like ``^model\\.layers\\.(\\d+)\\.x$`` matches."""
    return pattern.strip("^$").replace(r"\d+", "7").replace("\\", "").replace("(", "").replace(")", "")


def test_export_reader_rename_tables_are_inverses(ao_module, export_module):
    """The hand-written export/reader rename tables must stay exact inverses; drift strands weights silently."""
    assert len(export_module._LOGICAL_TO_HF) == len(ao_module._HF_TO_LOGICAL)
    for pattern, replacement in export_module._LOGICAL_TO_HF:
        logical = _sample(pattern)
        hf = re.sub(pattern, replacement, logical, count=1)
        assert hf != logical, f"{pattern!r} does not rename its sample"
        back = [p.sub(t, hf) for p, t in ao_module._HF_TO_LOGICAL if p.match(hf)]
        assert back == [logical], f"{hf!r} maps back through {len(back)} reader rules, expected exactly {logical!r}"
    for pattern, replacement in ao_module._HF_TO_LOGICAL:
        hf = _sample(pattern.pattern)
        logical = pattern.sub(replacement, hf)
        assert logical != hf, f"{pattern.pattern!r} does not rename its sample"
        back = [re.sub(p, t, logical, count=1) for p, t in export_module._LOGICAL_TO_HF if re.match(p, logical)]
        assert back == [hf]
    # the dense-quant recipe filter names exactly the logical side of the table
    suffixes = {_sample(p).removeprefix("layers.7") for p, _ in export_module._LOGICAL_TO_HF}
    assert suffixes == set(export_module._DENSE_WEIGHT_SUFFIXES)


_AO_FQN = "model.layers.0.self_attn.q_a_proj.weight"


def _write_ao_shard(tmp_path, *, data_names=("qdata", "scale"), block_size=32, quant_axis=1, pack_axis=None):
    """One dense fp8 MX tensor (2x64, all 1.0, scale 2^0) plus a plain tensor, in AO protocol."""
    prefix = f"{_AO_FQN.rsplit('.', 1)[0]}._weight_"
    qdata = torch.ones((2, 64), dtype=torch.float8_e4m3fn)
    scale = torch.full((2, 1, 2), 127, dtype=torch.uint8)  # (rows, ceil(2 blocks / 2), 2)
    record = {
        "_type": "MXTensor",
        "_tensor_data_names": list(data_names),
        "_data": {
            "quant_axis": quant_axis,
            "pack_axis": pack_axis,
            "quant_config": {
                "_type": "MXQuantizeConfig",
                "_data": {"elem_dtype": "float8_e4m3fn", "block_size": block_size},
            },
            "act_quant_config": None,
        },
    }
    save_file(
        {
            f"{prefix}{data_names[0]}": qdata,
            f"{prefix}{data_names[1]}": scale,
            "norm.weight": torch.ones(2),
        },
        tmp_path / "model.safetensors",
        metadata={
            "format": "pt",
            "tensor_names": json.dumps([_AO_FQN]),
            _AO_FQN: json.dumps(record),
        },
    )


def test_ao_block_size_must_match_kernel(ao_module, tmp_path):
    """Unsupported ``block_size`` fails closed."""
    _write_ao_shard(tmp_path, block_size=64)
    reader = ao_module.MxAOHuggingFaceStorageReader(str(tmp_path))
    with pytest.raises(ValueError, match=r"block_size.*32"):
        reader.read_metadata()


@pytest.mark.parametrize("bad_axis", ["quant", "pack"])
def test_ao_axes_must_point_at_last_dim(ao_module, tmp_path, bad_axis):
    """Non-last-dim quant/pack layouts would dequantize silently wrong; fail closed by name."""
    _write_ao_shard(tmp_path, **{f"{bad_axis}_axis": 0})
    reader = ao_module.MxAOHuggingFaceStorageReader(str(tmp_path))
    with pytest.raises(ValueError, match=f"{bad_axis}_axis"):
        reader.read_metadata()


def test_ao_components_resolved_by_dtype(ao_module, cpu_npu_boundary, tmp_path):
    """Components named by ``tensor_data_names`` load by dtype role, not literal names."""
    _write_ao_shard(tmp_path, data_names=("payload", "scales"))
    reader = ao_module.MxAOHuggingFaceStorageReader(str(tmp_path))
    metadata = reader.read_metadata()
    assert metadata.state_dict_metadata["layers.0.attn.wq_a.weight"].size == (2, 64)
    region = reader._regions["layers.0.attn.wq_a.weight"]
    assert region.qdata.key.endswith("_weight_payload") and region.scale.key.endswith("_weight_scales")

    target = {"layers.0.attn.wq_a.weight": torch.empty((2, 64), dtype=torch.bfloat16), "norm.weight": torch.empty(2)}
    dcp.load(target, storage_reader=reader)
    torch.testing.assert_close(target["layers.0.attn.wq_a.weight"], torch.ones(2, 64).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(target["norm.weight"], torch.ones(2))


@pytest.mark.parametrize("weight_format", ["mxfp4", "mxfp8", "bf16", "block-fp8"])
def test_only_v4_selects_mx_reader(reader_module, monkeypatch, tmp_path, weight_format):
    # Isolate model imports; execute the real V4/V3.2 adapter classes and MX reader.
    fallback = object()

    class V3Adapter:
        def get_hf_storage_reader(self, path, from_quantized=False, *, thread_count=None):
            return fallback

    original = V3Adapter.get_hf_storage_reader
    upstream = ModuleType("torchtitan.models.deepseek_v3.state_dict_adapter")
    upstream.DeepSeekV3StateDictAdapter = V3Adapter
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)

    class V4Adapter(V3Adapter):
        pass

    upstream_v4 = ModuleType("torchtitan.models.deepseek_v4.state_dict_adapter")
    upstream_v4.DeepSeekV4StateDictAdapter = V4Adapter
    monkeypatch.setitem(sys.modules, upstream_v4.__name__, upstream_v4)
    monkeypatch.setitem(
        sys.modules,
        "_mx_test_components.model",
        SimpleNamespace(
            DeepSeekV4Model=SimpleNamespace(Config=object),
        ),
    )

    # Reader selection only needs the PEFT mixin in the MRO; conversion is outside this test.
    class PEFTMixin:
        pass

    monkeypatch.setitem(
        sys.modules,
        "torchtitan_npu.models.deepseek_v4.lora",
        SimpleNamespace(
            DeepSeekPEFTAdapter=PEFTMixin,
            is_lora_parameter=lambda _name: False,
        ),
    )
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.mx_storage_reader.hf_storage", reader_module)
    # The V4 adapter probes the AO reader before the fixed-format MX reader.
    monkeypatch.setitem(
        sys.modules,
        "torchtitan_npu.extensions.mx_storage_reader.ao_hf_storage",
        sys.modules["_mx_test_components.ao_hf_storage"],
    )
    root = Path(__file__).resolve().parents[4] / "torchtitan_npu/models"
    classes = []
    for model, class_name in (
        ("deepseek_v4", "DeepSeekV4StateDictAdapter"),
        ("deepseek_v3_2", "DeepSeekV32StateDictAdapter"),
    ):
        spec = importlib.util.spec_from_file_location(
            f"_mx_test_components.{model}", root / model / "state_dict_adapter.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        classes.append(getattr(module, class_name))
    v4, v32 = classes
    dtype = {
        "mxfp4": torch.uint8,
        "mxfp8": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
        "block-fp8": torch.float8_e4m3fn,
    }[weight_format]
    tensors = {"linear.weight": torch.ones(1, 32).to(dtype)}
    if weight_format.startswith("mx"):
        tensors["linear.scale"] = torch.full((1, 2 if weight_format == "mxfp4" else 1), 127, dtype=torch.uint8)
    elif weight_format == "block-fp8":
        tensors["linear.weight_scale_inv"] = torch.ones(1, 1)
    save_file(tensors, tmp_path / "model.safetensors")

    reader = object.__new__(v4).get_hf_storage_reader(str(tmp_path), from_quantized=weight_format != "bf16")

    if weight_format.startswith("mx"):
        assert isinstance(reader, reader_module.MXHuggingFaceStorageReader)
    else:
        assert reader is fallback
    assert V3Adapter.get_hf_storage_reader is original
    assert v32.get_hf_storage_reader is original
