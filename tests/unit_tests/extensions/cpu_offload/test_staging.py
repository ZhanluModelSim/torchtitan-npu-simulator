# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Staging primitive tests: ownership protocol and event ordering (#808-8).

Covers the documented CpuStaging contract: data correctness and stream
ordering for pinned/pageable sources/destinations in both directions, the
pageable-D2H host-commit join, producer ordering, bounce-ring reuse beyond
the slot count, track=False caller-owned handles, validation rejections,
and the terminal close (idempotent, joins the executor thread, rejects
further submissions).
"""

import logging
import os
import threading

import torch
import torch_npu

from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging

import pytest

pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason='NPU not available'
)


logging.basicConfig(level=logging.INFO, force=True)


def staging_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("cpu-staging"))


def main() -> None:
    torch_npu.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu", torch_npu.npu.current_device())
    base_threads = staging_threads()

    staging = CpuStaging(device, owner="test-staging")

    # ---- 1. pinned H2D: data + consumer-stream ordering via wait_on ----
    pinned_src = torch.randn(1024).pin_memory()
    dst = torch.empty(1024, device=device)
    handle = staging.submit_h2d(pinned_src, dst)
    side = torch_npu.npu.Stream(device=device)
    with torch_npu.npu.stream(side):
        handle.wait_on(side)
        result = dst * 2.0  # enqueued on `side`, must observe the H2D result
    torch_npu.npu.synchronize()
    assert torch.allclose(result.cpu(), pinned_src * 2.0, atol=1e-5), "pinned H2D ordering broken"
    logging.info("pinned H2D OK")

    # ---- 2. pageable H2D: bounce-ring path ----
    pageable_src = torch.randn(2048)
    dst2 = torch.empty(2048, device=device)
    handle2 = staging.submit_h2d(pageable_src, dst2)
    handle2.wait()
    assert torch.allclose(dst2.cpu(), pageable_src, atol=1e-6), "pageable H2D data broken"
    logging.info("pageable H2D OK")

    # ---- 3. pinned D2H: bitwise copy ----
    src_npu = torch.randn(512, device=device)
    pinned_dst = torch.empty(512).pin_memory()
    handle3 = staging.submit_d2h(src_npu, pinned_dst)
    handle3.wait()
    assert torch.equal(pinned_dst, src_npu.cpu()), "pinned D2H data broken"
    logging.info("pinned D2H OK")

    # ---- 4. pageable D2H: handle.wait must join the host-side commit ----
    src_npu2 = torch.randn(777, device=device)  # odd size -> partial bounce
    expected2 = src_npu2.cpu()
    pageable_dst = torch.empty(777)
    handle4 = staging.submit_d2h(src_npu2, pageable_dst)
    handle4.wait()  # no device-wide synchronize: covers event AND executor
    assert torch.equal(pageable_dst, expected2), "pageable D2H commit not joined by wait()"
    assert staging_threads() == base_threads + 1, "pageable D2H did not spawn the executor"
    logging.info("pageable D2H commit join OK")

    # ---- 5. D2H producer ordering: copy must observe current-stream work ----
    src5 = torch.randn(4096, device=device)
    expected5 = src5 * 3.0
    src5.mul_(3.0)  # producer op on the current (default) stream
    dst5 = torch.empty(4096).pin_memory()
    handle5 = staging.submit_d2h(src5, dst5)  # producer defaults to current stream
    handle5.wait()
    assert torch.equal(dst5, expected5.cpu()), "D2H ignored producer ordering"
    logging.info("producer ordering OK")

    # ---- 6. ring reuse: more transfers than bounce slots ----
    ring = []
    for i in range(10):
        cpu_dst = torch.empty(256)
        h = staging.submit_d2h(torch.full((256,), float(i), device=device), cpu_dst)
        ring.append((cpu_dst, h))
    for i, (cpu_dst, h) in enumerate(ring):
        h.wait()
        assert torch.all(cpu_dst == float(i)), f"ring reuse corrupted transfer {i}"
    logging.info("bounce-ring reuse OK")

    # ---- 7. track=False: caller-owned handle, absent from staging.wait ----
    before = len(staging._pending)
    dst7 = torch.empty(128, device=device)
    src7 = torch.arange(128, dtype=torch.float32)
    handle7 = staging.submit_h2d(src7, dst7, track=False)
    assert len(staging._pending) == before, "track=False handle leaked into _pending"
    staging.wait()
    handle7.wait()
    assert torch.equal(dst7.cpu(), src7), "track=False handle data broken"
    logging.info("track=False ownership OK")

    # ---- 8. validation rejections ----
    try:
        staging.submit_h2d(torch.empty(4, device=device), torch.empty(4, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("NPU source accepted for H2D")
    try:
        staging.submit_h2d(torch.empty(4), torch.empty(5, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch accepted")
    logging.info("validation OK")

    # ---- 9. terminal close: idempotent, joins thread, rejects use ----
    staging.close()
    staging.close()
    assert staging_threads() == base_threads, "executor thread leaked after close"
    try:
        staging.submit_h2d(torch.empty(4), torch.empty(4, device=device))
    except RuntimeError:
        pass
    else:
        raise AssertionError("submit after close accepted")
    try:
        _ = staging.stream
    except RuntimeError:
        pass
    else:
        raise AssertionError("stream access after close accepted")
    logging.info("terminal close OK")

    logging.info("STAGING PRIMITIVE TEST PASSED")


if __name__ == "__main__":
    main()
