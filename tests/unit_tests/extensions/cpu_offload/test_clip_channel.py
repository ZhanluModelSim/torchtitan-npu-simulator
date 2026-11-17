# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Channel semantics test: the GradientClipChannel state machine.

Covers publish/take ownership rules (single take, metadata pinning, double
publish rejected), the clear/close lifecycle, and the stage_gradient
integration for both the cached and the fallback path with a pending
coefficient.
"""

import logging
import os

import torch
import torch_npu

import torchtitan_npu.extensions.cpu_offload.runtime as clip_state
from torchtitan_npu.extensions.cpu_offload.runtime import stage_gradient
from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging

import pytest

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason='NPU not available'
)


logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch_npu.npu.current_device())

    # ---- state machine on CPU tensors ----------------------------------
    channel = clip_state.GradientClipChannel()
    assert not channel.has_consumer()
    channel.register_consumer()
    channel.register_consumer()
    channel.unregister_consumer()
    assert channel.has_consumer(), "refcount semantics broken"

    parameter = torch.nn.Parameter(torch.randn(8))
    cpu_gradient = torch.randn(8)
    working = torch.randn(8, device=device)
    coefficient = torch.full((), 0.5, device=device)
    ready = torch_npu.npu.Event()
    ready.record(torch_npu.npu.current_stream())

    channel.publish([(parameter, cpu_gradient, working)], coefficient, ready)

    try:
        channel.publish([], coefficient, ready)
    except RuntimeError:
        pass
    else:
        raise AssertionError("double publish accepted")

    taken = channel.take_gradient(parameter, cpu_gradient)
    assert taken is working, "take returned a different tensor"
    try:
        channel.take_gradient(parameter, cpu_gradient)
    except RuntimeError:
        pass
    else:
        raise AssertionError("second take accepted")

    other = torch.nn.Parameter(torch.randn(8))
    try:
        channel.take_gradient(other, torch.randn(8))
    except RuntimeError:
        pass
    else:
        raise AssertionError("take for an unpublished parameter accepted")

    channel.clear_pending()
    assert channel.take_gradient(parameter, cpu_gradient) is None
    logging.info("channel state machine OK")

    # ---- stage_gradient integration (NPU) --------------------------------
    staging = CpuStaging(device, owner="test-channel")
    channel = clip_state.GradientClipChannel()
    channel.register_consumer()

    # Cached path: publish for p1; the working gradient is consumed and
    # scaled in place, so capture the expectation first.
    p1 = torch.nn.Parameter(torch.randn(16))
    g1 = torch.randn(16)
    w1 = torch.randn(16, device=device)
    expected1 = w1 * 0.25

    coefficient = torch.full((), 0.25, device=device)
    ready = torch_npu.npu.Event()
    ready.record(torch_npu.npu.current_stream())
    channel.publish([(p1, g1, w1)], coefficient, ready)

    dst1 = torch.empty(16, device=device)
    staged1, handles1 = stage_gradient(p1, g1, dst1, staging=staging, channel=channel, stream=staging.stream)
    assert handles1 == (), "cached path must not return transfer handles"
    torch_npu.npu.synchronize()
    assert torch.allclose(staged1, expected1, atol=1e-6), "cached gradient not scaled"
    logging.info("cached path OK")

    # Fallback path: no pending clip state, plain CPU->NPU staging.
    channel.clear_pending()
    p2 = torch.nn.Parameter(torch.randn(16))
    g2_cpu = torch.randn(16)
    dst2 = torch.empty(16, device=device)
    staged2, handles2 = stage_gradient(p2, g2_cpu, dst2, staging=staging, channel=channel, stream=staging.stream)
    for handle in handles2:
        handle.wait_on(torch_npu.npu.current_stream())
    torch_npu.npu.synchronize()
    assert torch.allclose(staged2.cpu(), g2_cpu, atol=1e-6), "fallback staging altered"
    logging.info("fallback path OK")

    channel.close()
    staging.close()
    logging.info("GRADIENT CLIP CHANNEL TEST PASSED")


if __name__ == "__main__":
    main()
