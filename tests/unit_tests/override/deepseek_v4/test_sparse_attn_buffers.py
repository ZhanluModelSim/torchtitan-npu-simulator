# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Buffer protocol tests for the DSV4 sparse-attention loss accumulator.

Pins the three subtle contracts of ``_init_self_buffers`` (review #808-18):
the plain attribute re-assignment must preserve the buffer *registration*
and its ``persistent=False`` flag, re-initialization must land on the
requested device (or keep the current one after ``to_empty``), and the
value must be re-zeroed. The tests call the real implementation through a
minimal carrier module so the heavy attention ``__init__`` is not needed.
"""

import logging

import pytest
import torch
import torch_npu

from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
)

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU not available"
)

logging.basicConfig(level=logging.INFO, force=True)


def _npu_device() -> torch.device:
    if not torch_npu.npu.is_available():
        pytest.skip("NPU not available")
    torch_npu.npu.set_device(0)
    return torch.device("npu", torch_npu.npu.current_device())


def _make_carrier() -> torch.nn.Module:
    """Minimal stand-in carrying exactly the reviewed buffer registration."""
    carrier = torch.nn.Module()
    carrier.register_buffer(
        "_indexer_loss_acc",
        torch.zeros((), dtype=torch.float32),
        persistent=False,
    )
    return carrier


def _reinit(carrier: torch.nn.Module, **kwargs) -> None:
    """Invoke the real (unbound) implementation against the carrier."""
    AscCompressedSparseInnerAttention._init_self_buffers(carrier, **kwargs)


def test_registration_and_persistence():
    carrier = _make_carrier()
    assert "_indexer_loss_acc" in carrier._buffers
    assert "_indexer_loss_acc" not in carrier.state_dict()  # persistent=False


def test_reinit_explicit_device():
    device = _npu_device()
    carrier = _make_carrier()
    _reinit(carrier, buffer_device=device)
    assert carrier._indexer_loss_acc.device == device
    assert carrier._indexer_loss_acc.item() == 0.0


def test_reinit_preserves_registration_and_persistence():
    device = _npu_device()
    carrier = _make_carrier()
    _reinit(carrier, buffer_device=device)
    # The plain attribute assignment must not demote the buffer.
    assert "_indexer_loss_acc" in carrier._buffers
    assert "_indexer_loss_acc" not in carrier.state_dict()


def test_reinit_after_to_empty_keeps_device_and_zeroes():
    device = _npu_device()
    carrier = _make_carrier()
    carrier.to_empty(device=device)  # protocol: moved, but garbage data
    carrier._indexer_loss_acc.fill_(3.0)
    _reinit(carrier)  # buffer_device=None: infer the emptied buffer's device
    assert carrier._indexer_loss_acc.device == device
    assert carrier._indexer_loss_acc.item() == 0.0
    assert "_indexer_loss_acc" not in carrier.state_dict()


def test_reinit_dirty_value_is_rezeroed():
    device = _npu_device()
    carrier = _make_carrier()
    _reinit(carrier, buffer_device=device)
    carrier._indexer_loss_acc.fill_(7.0)
    _reinit(carrier, buffer_device=device)
    assert carrier._indexer_loss_acc.item() == 0.0


if __name__ == "__main__":
    test_registration_and_persistence()
    test_reinit_explicit_device()
    test_reinit_preserves_registration_and_persistence()
    test_reinit_after_to_empty_keeps_device_and_zeroes()
    test_reinit_dirty_value_is_rezeroed()
    logging.info("SPARSE-ATTN BUFFER PROTOCOL TESTS PASSED")
