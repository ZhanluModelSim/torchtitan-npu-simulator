# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest
import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.config import SimulationConfig, prepare_config
from torchtitan_simulator.metadata import to_meta_metadata
from torchtitan_simulator.runtime import meta_runtime


def test_metadata_lift_preserves_views_and_repeated_identity_but_keeps_host_splits():
    @dataclass
    class Plan:
        indices: torch.Tensor
        alias: torch.Tensor
        repeated: torch.Tensor
        send_splits_tensor: torch.Tensor

    base = torch.arange(8)[1:].detach()
    view = base[1:4]
    splits = torch.tensor([2, 1])

    lifted = to_meta_metadata(Plan(base, view, view, splits))

    assert lifted.indices.device.type == lifted.alias.device.type == "meta"
    assert lifted.repeated is lifted.alias
    assert lifted.alias.untyped_storage()._cdata == lifted.indices.untyped_storage()._cdata
    assert lifted.alias.storage_offset() == 1
    assert lifted.send_splits_tensor is splits


def test_selected_li_metadata_uses_vendor_shape_without_device_initialization():
    prepared = prepare_config(SimulationConfig())
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    original = ascendc.lightning_indexer_metadata
    cu_q = torch.tensor([0, 128], dtype=torch.int32)
    cu_k = torch.tensor([0, 32], dtype=torch.int32)

    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        metadata = ascendc.lightning_indexer_metadata(
            8,
            1,
            128,
            512,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            cmp_residual_k=torch.tensor([0], dtype=torch.int32),
            layout_q="TND",
            layout_k="TND",
            mask_mode=3,
            cmp_ratio=4,
        )

    # LI's public vendor Meta implementation and C++ definition both specify
    # 1024 int32 entries; this test checks geometry, not opaque device values.
    assert metadata.device.type == "meta"
    assert metadata.shape == (1024,) and metadata.dtype == torch.int32
    assert torch.equal(cu_q, torch.tensor([0, 128], dtype=torch.int32))
    assert ascendc.lightning_indexer_metadata is original


@pytest.mark.parametrize(
    "name,size",
    [
        ("lightning_indexer_metadata", 1024),
        ("sparse_flash_mla_metadata", 1024),
        ("sparse_flash_mla_grad_metadata", 1024),
        ("sparse_lightning_indexer_kl_loss_grad_metadata", 64),
    ],
)
def test_vendor_metadata_dispatch_preserves_consumer_dependency_and_restores_scope(name, size):
    prepared = prepare_config(SimulationConfig())
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    original = getattr(ascendc, name)
    original_empty = torch.empty
    original_kernel = torch._C._dispatch_dump(f"cann_ops_transformer::{name}")
    for _ in range(2):
        capture = OpDispatchCapture()
        with meta_runtime() as runtime:
            runtime.configure_model(prepared)
            with capture:
                arguments = [8, 1, 128, 512] if name == "lightning_indexer_metadata" else [8, 1, 128]
                result = getattr(ascendc, name)(
                    *arguments, cu_seqlens_q=torch.empty(2, device="meta", dtype=torch.int32)
                )
                result.view(-1).clone()
        nodes = capture.build_nodes()
        native = next(
            node for node in nodes.values() if node.annotations["raw_op_type"] == f"cann_ops_transformer.{name}.default"
        )
        assert result.shape == (size,) and result.dtype == torch.int32 and result.device.type == "meta"
        assert native.successors
        assert all(native.op_id in nodes[key].predecessors for key in native.successors)
        assert getattr(ascendc, name) is original
        assert torch.empty is original_empty
        assert torch._C._dispatch_dump(f"cann_ops_transformer::{name}") == original_kernel


def test_vendor_metadata_meta_registration_is_restored_after_exception():
    prepared = prepare_config(SimulationConfig())
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    original = ascendc.lightning_indexer_metadata
    kernel = torch._C._dispatch_dump("cann_ops_transformer::lightning_indexer_metadata")
    with pytest.raises(RuntimeError, match="abort capture"), meta_runtime() as runtime:
        runtime.configure_model(prepared)
        raise RuntimeError("abort capture")
    assert ascendc.lightning_indexer_metadata is original
    assert torch._C._dispatch_dump("cann_ops_transformer::lightning_indexer_metadata") == kernel


def test_healthy_vendor_metadata_meta_kernel_is_not_overridden():
    prepared = prepare_config(SimulationConfig())
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    library = torch.library.Library("cann_ops_transformer", "IMPL", "Meta")
    library.impl(
        "lightning_indexer_metadata", lambda *args, **kwargs: torch.empty(1024, device="meta", dtype=torch.int32)
    )
    try:
        kernel = torch._C._dispatch_dump("cann_ops_transformer::lightning_indexer_metadata")
        with meta_runtime() as runtime:
            runtime.configure_model(prepared)
            assert torch._C._dispatch_dump("cann_ops_transformer::lightning_indexer_metadata") == kernel
            result = ascendc.lightning_indexer_metadata(
                8, 1, 128, 512, cu_seqlens_q=torch.empty(2, device="meta", dtype=torch.int32)
            )
            assert result.shape == (1024,)
        assert torch._C._dispatch_dump("cann_ops_transformer::lightning_indexer_metadata") == kernel
    finally:
        library._destroy()
