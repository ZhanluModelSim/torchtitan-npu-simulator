# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU contracts for model-independent NovaSwap group payloads."""

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.extensions.novaswap import swap_api
from torchtitan_npu.extensions.novaswap import swap_engine as product_engine
from torchtitan_npu.extensions.novaswap.swap_ir import SwapAction
from torchtitan_npu.extensions.novaswap.swap_primitive import SwapGroupHandle
from torchtitan_npu.extensions.novaswap.swap_registry import SwapRegistry


@pytest.fixture(autouse=True)
def _reset_swap_registry():
    SwapRegistry.clear_tensors()
    yield
    SwapRegistry.clear_tensors()


def test_register_group_returns_aligned_authoritative_layout():
    first = torch.ones(3, dtype=torch.uint8)
    second = torch.ones(2, dtype=torch.float32)

    registration = swap_api.register_group((first, second), "momentum")

    assert registration.name == "momentum"
    assert [(member.byte_offset, member.num_bytes) for member in registration.members] == [(0, 3), (4, 8)]
    assert registration.total_bytes == 12
    assert SwapRegistry.get_entry_by_name("momentum").group_registration is registration


@pytest.mark.parametrize(
    "tensors, message",
    [
        ((), "at least one"),
        ((torch.ones(4)[::2],), "contiguous"),
        ((torch.ones(4),) * 2, "share storage"),
    ],
    ids=("empty-group", "noncontiguous-tensor", "shared-storage"),
)
def test_register_group_rejects_invalid_storage(tensors, message):
    with pytest.raises(ValueError, match=message):
        swap_api.register_group(tensors, "invalid")


def test_single_member_group_retains_group_ownership():
    registration = swap_api.register_group((torch.ones(2),), "single")

    entry = SwapRegistry.get_entry_by_name("single")
    assert entry is not None
    assert entry.group_registration is registration
    assert len(entry.storage_tensors) == 1


def test_group_d2h_uses_one_packed_handle(monkeypatch):
    calls = []

    class Worker:
        def enqueue(self, *args):
            calls.append(("release", args))

    def fake_group(tensors, offsets, total_bytes, stream, *, owner):
        calls.append(("group", tensors, offsets, total_bytes, stream, owner))
        return SwapGroupHandle(tensors_npu=tensors, offsets=offsets, transfer="D2H")

    monkeypatch.setattr(product_engine, "d2h_group", fake_group)
    monkeypatch.setattr(product_engine.SwapEngine, "_ready", True)
    monkeypatch.setattr(product_engine.SwapEngine, "_handles", {})
    monkeypatch.setattr(product_engine.SwapEngine, "_offload_stream", object())
    monkeypatch.setattr(product_engine.SwapEngine, "_release_worker", Worker())
    registration = swap_api.register_group((torch.ones(2), torch.ones(3)), "bucket")

    product_engine.SwapEngine.execute(SwapAction("bucket", "", "D2H"))

    assert calls[0][0] == "group"
    assert calls[0][2] == tuple(member.byte_offset for member in registration.members)
    assert calls[0][3] == registration.total_bytes
    assert len(product_engine.SwapEngine._handles["bucket"]) == 1


def test_wait_device_if_needed_is_idempotent(monkeypatch):
    waits = []
    event = object()
    handle = SwapGroupHandle(transfer="H2D", swap_event=event)
    fake_npu = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(wait_event=waits.append)))
    monkeypatch.setattr(product_engine, "torch_npu", fake_npu)
    monkeypatch.setattr(product_engine.SwapEngine, "_ready", True)
    monkeypatch.setattr(product_engine.SwapEngine, "_handles", {"bucket": (handle,)})
    monkeypatch.setattr(product_engine.SwapEngine, "_release_worker", object())

    swap_api.wait_device_if_needed("bucket")
    swap_api.wait_device_if_needed("bucket")

    assert waits == [event]
    assert product_engine.SwapEngine.handle_phase("bucket") == "H2D_WAITED"


def test_release_npu_releases_every_group_member():
    first = torch.ones(2)
    second = torch.ones(3)

    product_engine._release_npu(SwapGroupHandle(tensors_npu=(first, second)))

    assert first.untyped_storage().size() == 0
    assert second.untyped_storage().size() == 0
