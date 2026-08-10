# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.ops.tilelang import moe_reduce, runtime


def _reference_forward(x, token_topk_to_pos):
    gathered = x[token_topk_to_pos.to(torch.int64)]
    return gathered.sum(dim=1)


def _reference_backward(dx, token_topk_to_pos, grad_output):
    dx[token_topk_to_pos.to(torch.int64)] = grad_output.unsqueeze(1)


@pytest.fixture
def fake_tilelang_sources(monkeypatch):
    build_calls = {"forward": 0, "backward": 0}

    def get_forward_kernel(**kwargs):
        assert kwargs["with_weights"] is False
        num_topk = kwargs["num_topk"]
        build_calls["forward"] += 1

        def kernel(*args):
            x, _weights, mapping, _sf, _x_sf, output = args
            mapping = mapping.view(output.shape[0], num_topk)
            output.copy_(_reference_forward(x, mapping))

        return kernel

    def get_backward_kernel(**kwargs):
        assert kwargs["with_weights"] is False
        num_topk = kwargs["num_topk"]
        build_calls["backward"] += 1

        def kernel(*args):
            _x, _weights, mapping, _sf, _x_sf, grad_output, grad_x, _dweights, _dx_sf, _dsf = args
            mapping = mapping.view(grad_output.shape[0], num_topk)
            _reference_backward(grad_x, mapping, grad_output)

        return kernel

    modules = (
        SimpleNamespace(get_reduce_fused_kernel=get_forward_kernel),
        SimpleNamespace(get_reduce_fused_backward_kernel=get_backward_kernel),
    )
    monkeypatch.setattr(runtime, "lazy_load_tilelang", lambda: modules)
    runtime.get_cached_forward_kernel.cache_clear()
    runtime.get_cached_backward_kernel.cache_clear()
    yield build_calls
    runtime.get_cached_forward_kernel.cache_clear()
    runtime.get_cached_backward_kernel.cache_clear()


def test_tilelang_moe_reduce_forward_backward_and_kernel_cache(fake_tilelang_sources):
    x = torch.arange(24, dtype=torch.float32).view(6, 4).requires_grad_()
    mapping = torch.tensor([[2, 0], [5, 1], [3, 4]], dtype=torch.int32)

    output = moe_reduce.tilelang_moe_reduce_fused(x, mapping)
    expected = _reference_forward(x, mapping)
    assert torch.equal(output, expected)

    output.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))

    moe_reduce.tilelang_moe_reduce_fused(x.detach(), mapping)
    assert fake_tilelang_sources == {"forward": 1, "backward": 1}


def test_tilelang_moe_reduce_supports_fullgraph_compile(fake_tilelang_sources):
    x = torch.arange(24, dtype=torch.float32).view(6, 4).requires_grad_()
    mapping = torch.tensor([[2, 0], [5, 1], [3, 4]], dtype=torch.int32)
    compiled = torch.compile(
        moe_reduce.tilelang_moe_reduce_fused,
        backend="eager",
        fullgraph=True,
    )

    output = compiled(x, mapping)
    assert torch.equal(output, _reference_forward(x, mapping))

    output.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_tilelang_moe_reduce_custom_op_contract(fake_tilelang_sources):
    x = torch.arange(24, dtype=torch.float32).view(6, 4).requires_grad_()
    mapping = torch.tensor([[2, 0], [5, 1], [3, 4]], dtype=torch.int32)

    results = torch.library.opcheck(
        moe_reduce.moe_reduce_fused_tilelang_op,
        (x, mapping),
    )

    assert all(result == "SUCCESS" for result in results.values())


def test_tilelang_moe_reduce_normalizes_mapping_at_custom_op_boundary(fake_tilelang_sources):
    x = torch.ones(4, 3)
    mapping = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    output = moe_reduce.tilelang_moe_reduce_fused(x, mapping)

    assert torch.equal(output, torch.full((2, 3), 2.0))


def test_tilelang_moe_reduce_context_does_not_save_forward_activation(fake_tilelang_sources):
    saved_tensors = []
    x = torch.ones(6, 4, requires_grad=True)
    mapping = torch.tensor([[2, 0], [5, 1], [3, 4]], dtype=torch.int32)

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        moe_reduce.tilelang_moe_reduce_fused(x, mapping).sum().backward()

    assert any(tensor is mapping for tensor in saved_tensors)
    assert all(tensor is not x for tensor in saved_tensors)


def test_tilelang_cache_isolated_by_job_and_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHTITAN_NPU_TILELANG_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job_1")
    monkeypatch.setenv("RANK", "7")

    cache_dir = runtime.configure_tilelang_cache()

    source_id = runtime.kernel_source_fingerprint()
    assert cache_dir == tmp_path / "job_1" / f"source_{source_id}" / "rank_7"
    assert cache_dir.is_dir()
    assert runtime.os.environ["TILELANG_CACHE_DIR"] == str(cache_dir)
