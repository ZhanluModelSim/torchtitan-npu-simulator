# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.distributed import process_group
from torchtitan_npu.distributed.context_parallel.compressor_attention_cp import _WindowExchange
from torchtitan_npu.simulator import meta_env
from torchtitan_npu.simulator.capture import comm_events
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("record", [False, True])
def test_window_exchange_consumes_recorded_receive_buffer(monkeypatch, rank, record):
    group = SimpleNamespace(rank=lambda: rank, size=lambda: 3, group_name="cp")
    monkeypatch.setattr(process_group, "is_fake_process_group", lambda candidate: candidate is group)
    monkeypatch.setattr(meta_env, "_original_window_exchange", None)
    monkeypatch.setattr(meta_env, "_comm_layer", "")
    monkeypatch.setattr(_WindowExchange, "forward", _WindowExchange.forward)
    monkeypatch.setattr(_WindowExchange, "backward", _WindowExchange.backward)
    recorder = comm_events.CommEventRecorder() if record else None
    monkeypatch.setattr(comm_events, "_active_recorder", recorder)
    monkeypatch.setattr(comm_events, "_resolve_comm_ranks", lambda _: [[0, 1, 2]])
    meta_env._patch_window_exchange_for_fake_pg()
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    x = torch.ones(1, 8, 2, device="meta", requires_grad=True)
    with capture:
        y = _WindowExchange.apply(x, 2, group)
        assert y.shape == (1, 8 + (2 if rank > 0 else 0), 2)
        phase[0] = "backward"
        y.backward(torch.ones_like(y))
    assert x.grad.shape == x.shape
    if not record:
        return
    events = capture.memory_events()
    receives = [event for event in events if event.raw_op_type == "comm.p2p_recv"]
    assert len(receives) == int(rank > 0) + int(rank < 2)
    plan = estimate_static_memory(events)
    for recv in receives:
        buffer_id = recv.outputs[0].tensor_id
        consumers = [
            event
            for event in events
            if event.seq_idx > recv.seq_idx and any(ref.tensor_id == buffer_id for ref in event.inputs)
        ]
        assert consumers, (rank, recv.phase)
        expected_op = "aten.cat.default" if recv.phase == "forward" else "aten.add.Tensor"
        consumer = next(event for event in consumers if event.raw_op_type == expected_op)
        ir_consumer = next(event for event in capture._events if event.seq_idx == consumer.seq_idx)
        assert recv.op_id in ir_consumer.predecessors
        lifetimes = [item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{buffer_id}"]
        assert len(lifetimes) == 1
        assert lifetimes[0].num_bytes == 16
