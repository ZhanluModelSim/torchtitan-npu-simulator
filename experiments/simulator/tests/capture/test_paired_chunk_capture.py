# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan_simulator import meta_env
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_reverse_backward_order_captures_the_forward_source_microbatch(monkeypatch, device):
    context = {"stage": 0, "mb_idx": 0, "comp_type": "F", "phase": "forward"}
    monkeypatch.setattr(meta_env, "_pp_context", context)
    capture = OpDispatchCapture(phase_provider=lambda: context["phase"])
    losses = []
    with capture, AutogradSavedTensorCapture():
        for mb in (0, 1):
            context.update(mb_idx=mb, comp_type="F", phase="forward")
            capture.begin_chunk((0, "F"))
            x = torch.ones(4, device=device, requires_grad=True)
            losses.append(x.sin().cos().sum())
            capture.end_chunk()
        for mb in (1, 0):
            context.update(mb_idx=mb, comp_type="B", phase="backward")
            capture.begin_chunk((0, "B"))
            losses[mb].backward()
            capture.end_chunk()
    events = capture.memory_events()
    assert {event.pp_mb_idx for event in events} == {0}
    assert {event.comp_type for event in events} == {"F", "B"}
    assert capture.class_instance_counts == {(0, "F"): 2, (0, "B"): 2}
    assert all(saved.pp_mb_idx == 0 for saved in capture.autograd_saved_tensor_events())
