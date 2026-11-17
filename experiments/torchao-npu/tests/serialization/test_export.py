# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the HF-layout safetensors writer (to_bytes and save_hf_safetensors)."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict
from torchao.prototype.safetensors.safetensors_utils import TensorSubclassAttributeJSONEncoder, is_metadata_torchao
from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantized_tensors import BlockMXTensor, MXTensor
from torchao_npu.serialization.export import save_hf_safetensors, to_bytes


def make_mx_tensor() -> MXTensor:
    """An MXTensor built straight from qdata/scale, without the NPU quant kernel."""
    qdata = torch.randn(4, 32).to(torch.float8_e4m3fn)
    scale = torch.zeros(4, 1, 2, dtype=torch.uint8)
    return MXTensor(qdata, scale, torch.bfloat16, -1, MXQuantizeConfig())


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1024, 1024),
        ("1024", 1024),
        ("1K", 1024),
        ("1kb", 1024),
        ("512MB", 512 * 1024**2),
        ("5GB", 5 * 1024**3),
        ("1T", 1024**4),
        (" 2 G ", 2 * 1024**3),
        ("1.5K", 1536),
    ],
)
def test_to_bytes_parses(size, expected):
    assert to_bytes(size) == expected


@pytest.mark.parametrize("size", [True, False, 3.5, None, [], {}])
def test_to_bytes_rejects_non_int_or_str(size):
    with pytest.raises(TypeError):
        to_bytes(size)


@pytest.mark.parametrize("size", [0, -1, "0", "0B", "0.001B", "", "abc", "5X", "GB5", "5GB1"])
def test_to_bytes_rejects_unparsable_or_nonpositive(size):
    with pytest.raises(ValueError):
        to_bytes(size)


def test_empty_state_dict_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_hf_safetensors({}, tmp_path)


def test_single_shard_writes_model_safetensors_without_index(tmp_path):
    sd = {
        "model.embed_tokens.weight": torch.randn(8, 8).to(torch.bfloat16),
        "hc_head_fn": torch.arange(4, dtype=torch.int64),
    }

    assert save_hf_safetensors(sd, tmp_path) == ["model.safetensors"]
    assert not (tmp_path / "model.safetensors.index.json").exists()

    loaded = load_file(str(tmp_path / "model.safetensors"))
    assert torch.equal(loaded["model.embed_tokens.weight"], sd["model.embed_tokens.weight"])
    assert torch.equal(loaded["hc_head_fn"], sd["hc_head_fn"])


def test_parameters_are_saved_as_plain_detached_tensors(tmp_path):
    weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=True)

    save_hf_safetensors({"model.weight": weight}, tmp_path)

    loaded = load_file(str(tmp_path / "model.safetensors"))
    assert isinstance(loaded["model.weight"], torch.Tensor)
    assert not isinstance(loaded["model.weight"], MXTensor)
    assert not loaded["model.weight"].requires_grad
    assert torch.equal(loaded["model.weight"], weight)


def test_mx_tensor_flattens_to_qdata_scale_and_round_trips(tmp_path):
    mx = make_mx_tensor()
    fqn = "model.layers.0.mlp.experts.0.gate_proj.weight"
    prefix = fqn.rsplit(".", 1)[0]  # flattening replaces the last segment

    save_hf_safetensors({fqn: mx}, tmp_path)

    loaded = load_file(str(tmp_path / "model.safetensors"))
    assert sorted(loaded) == [f"{prefix}._weight_qdata", f"{prefix}._weight_scale"]
    assert torch.equal(loaded[f"{prefix}._weight_qdata"], mx.qdata)
    assert torch.equal(loaded[f"{prefix}._weight_scale"], mx.scale)


def test_mx_tensor_requires_dotted_fqn(tmp_path):
    with pytest.raises(ValueError, match="dotted"):
        save_hf_safetensors({"embed_weight": make_mx_tensor()}, tmp_path)


def test_unregistered_tensor_subclass_rejected(tmp_path):
    # BlockMXTensor is a training-side representation the convert step lowers to
    # MXTensor; if it ever reaches the writer, saving it as raw bytes would
    # silently produce a checkpoint the load side cannot rebuild.
    qdata = torch.randn(32, 32).to(torch.float8_e4m3fn)
    block_mx = BlockMXTensor(
        qdata,
        torch.zeros(32, 1, 2, dtype=torch.uint8),
        torch.zeros(1, 32, 2, dtype=torch.uint8),
        torch.bfloat16,
        BlockMXQuantizeConfig(),
    )

    with pytest.raises(TypeError, match="unsupported tensor subclass BlockMXTensor"):
        save_hf_safetensors({"model.layers.0.mlp.gate_proj.weight": block_mx}, tmp_path)


def test_non_tensor_value_rejected(tmp_path):
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        save_hf_safetensors({"model.weight": 42}, tmp_path)


def test_non_cpu_tensor_rejected(tmp_path):
    with pytest.raises(ValueError, match="CPU"):
        save_hf_safetensors({"model.weight": torch.zeros(2, device="meta")}, tmp_path)


def test_multi_shard_names_index_and_self_describing_headers(tmp_path):
    mx = make_mx_tensor()  # qdata 128 B + scale 8 B
    plain = torch.randn(8, 8).to(torch.bfloat16)  # 128 B
    sd = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": mx,
        "model.embed_tokens.weight": plain,
    }

    names = save_hf_safetensors(sd, tmp_path, max_shard_size=200)

    assert names == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in names:
        with safe_open(str(tmp_path / name), framework="pt", device="cpu") as f:
            assert is_metadata_torchao(f.metadata())

    idx = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == {
        "model.layers.0.mlp.experts.0.gate_proj._weight_qdata",
        "model.layers.0.mlp.experts.0.gate_proj._weight_scale",
        "model.embed_tokens.weight",
    }
    assert set(idx["weight_map"].values()) <= set(names)
    total = sum(t.nelement() * t.element_size() for name in names for t in load_file(str(tmp_path / name)).values())
    assert idx["metadata"]["total_size"] == total


def test_oversized_tensor_gets_a_shard_of_its_own(tmp_path):
    big = torch.randn(11, 10).to(torch.bfloat16)  # 220 B
    small = torch.randn(1).to(torch.bfloat16)  # 2 B

    names = save_hf_safetensors({"a.weight": big, "b.weight": small}, tmp_path, max_shard_size=200)

    assert len(names) == 2
    shards = {name: load_file(str(tmp_path / name)) for name in names}
    owners = {tensor: name for name, tensors in shards.items() for tensor in tensors}
    assert owners["a.weight"] != owners["b.weight"]
    assert set(shards[owners["a.weight"]]) == {"a.weight"}


def test_tensors_fitting_exactly_share_one_shard(tmp_path):
    a = torch.randn(100).to(torch.bfloat16)  # 200 B
    b = torch.randn(100).to(torch.bfloat16)  # 200 B

    assert save_hf_safetensors({"a.weight": a, "b.weight": b}, tmp_path, max_shard_size=400) == ["model.safetensors"]


def test_saved_mx_tensor_rebuilds_via_torchao_unflatten(tmp_path):
    # Load-side contract at unit level: the header metadata the writer saves is
    # sufficient for torchao to rebuild the MXTensor (the e2e pins this through
    # from_pretrained, which needs NPU + CANN).
    mx = make_mx_tensor()
    fqn = "model.layers.0.mlp.experts.0.gate_proj.weight"

    save_hf_safetensors({fqn: mx}, tmp_path)
    path = str(tmp_path / "model.safetensors")
    loaded = load_file(path)
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()

    rebuilt, leftover = unflatten_tensor_state_dict(loaded, metadata)

    assert not leftover
    got = rebuilt[fqn]
    assert isinstance(got, MXTensor)
    assert torch.equal(got.qdata, mx.qdata)
    assert torch.equal(got.scale, mx.scale)
    assert (got.orig_dtype, got.quant_axis, got.pack_axis) == (mx.orig_dtype, mx.quant_axis, mx.pack_axis)

    def enc(config) -> str:
        return json.dumps(config, cls=TensorSubclassAttributeJSONEncoder, sort_keys=True)

    assert enc(got.quant_config) == enc(mx.quant_config)
    assert enc(got.act_quant_config) == enc(mx.act_quant_config)


def test_non_contiguous_tensor_is_materialized_contiguous(tmp_path):
    # safetensors' save_file rejects non-contiguous inputs; the writer's
    # .contiguous() path must materialize them without changing values.
    t = torch.randn(8, 16, dtype=torch.bfloat16).t()
    assert not t.is_contiguous()

    save_hf_safetensors({"model.weight": t}, tmp_path)

    loaded = load_file(str(tmp_path / "model.safetensors"))
    assert loaded["model.weight"].is_contiguous()
    assert torch.equal(loaded["model.weight"], t)
