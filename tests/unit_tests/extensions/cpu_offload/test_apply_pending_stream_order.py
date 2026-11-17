# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Regression test for the clip-coefficient stream ordering (review #808-5).

The clip coefficient may still be in flight on a producer stream when the
optimizer consumes it. ``apply_pending`` must issue its wait on the stream
that executes the ``mul_`` (the current stream); a wait issued on any other
stream is invisible to it. This test makes the producer late on a side
stream and asserts the scaling observes the final coefficient value.
"""

import logging
import os

import torch
import torch_npu

import torchtitan_npu.extensions.cpu_offload.runtime as clip_state

import pytest

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason='NPU not available'
)


logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch_npu.npu.current_device())

    wrong, right = 7.0, 0.5
    coefficient = torch.full((), wrong, dtype=torch.float32, device=device)
    gradient = torch.full((), 2.0, dtype=torch.float32, device=device)
    torch_npu.npu.synchronize()

    # Producer on a side stream: a long op backlog delays the wrong->right
    # overwrite, and only then records the ready event.
    side = torch_npu.npu.Stream(device=device)
    with torch_npu.npu.stream(side):
        backlog = torch.randn(4096, 4096, device=device)
        for _ in range(30):
            backlog = backlog @ backlog
            backlog.mul_(1e-6)
        coefficient.fill_(right)
        ready = torch_npu.npu.Event()
        ready.record(side)

    channel = clip_state.GradientClipChannel()
    channel.publish([], coefficient, ready)

    # Consume on the main stream: the wait must land here for the mul_ below.
    scaled = gradient.clone()
    channel.apply_pending(scaled)
    got = float(scaled.item())
    channel.clear_pending()

    expected = 2.0 * right
    assert abs(got - expected) < 1e-6, (
        f"apply_pending observed the pre-update coefficient: got {got}, expected {expected}"
    )
    logging.info(f"apply_pending stream-order OK: got {got} == expected {expected}")

    # Cross-stream consumer: same contract must hold on a non-default stream.
    channel.publish([], coefficient, ready)
    other = torch_npu.npu.Stream(device=device)
    with torch_npu.npu.stream(other):
        result = torch.full((), 3.0, dtype=torch.float32, device=device)
        channel.apply_pending(result)
    got2 = float(result.item())
    channel.clear_pending()
    assert abs(got2 - 3.0 * right) < 1e-6, f"cross-stream apply_pending observed the pre-update coefficient: got {got2}"
    logging.info(f"apply_pending cross-stream OK: got {got2}")

    torch_npu.npu.synchronize()
    logging.info("CLIP COEFFICIENT STREAM-ORDER REGRESSION TEST PASSED")


if __name__ == "__main__":
    main()
