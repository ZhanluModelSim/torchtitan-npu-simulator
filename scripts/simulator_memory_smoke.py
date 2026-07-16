#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small static-memory smoke test for the simulator.

This avoids the full torchtitan trainer stack and exercises the P0 path:
TorchDispatch capture -> raw memory events -> static liveness estimator ->
human-readable exports.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory
from torchtitan_npu.simulator.memory.export import export_memory_plan


class TinyCheckpointLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 16, device="meta")
        self.out_proj = nn.Linear(16, 4, device="meta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.in_proj(x).relu()
        viewed = hidden.view(hidden.shape[0], 4, 4)
        return self.out_proj(viewed.reshape(hidden.shape[0], 16))


def _print_plan(plan, raw_event_count: int, out_dir: str) -> None:  # noqa: ANN001
    print("Static memory smoke")
    print(f"  raw_memory_events={raw_event_count}")
    print(f"  persistent_param_bytes={plan.persistent_param_bytes}")
    print(f"  active_bytes_peak={plan.peak_active_bytes}")
    print(f"  peak_seq_idx={plan.peak_seq_idx} peak_phase={plan.peak_phase}")
    print("  largest lifetimes:")
    ranked = sorted(plan.tensor_lifetimes, key=lambda item: item.num_bytes, reverse=True)
    for item in ranked[:8]:
        print(
            "   "
            f" {item.kind:<16} bytes={item.num_bytes:<8} "
            f"seq={item.birth_seq}->{item.death_seq:<4} "
            f"producer={item.producer_raw_op}"
        )
    print(f"  exports={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./simulator_memory_smoke_output")
    args = parser.parse_args()

    model = TinyCheckpointLikeModel()
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        x = torch.randn(2, 8, device="meta")
        y = model(x)
        phase["value"] = "backward"
        recomputed = model.in_proj(x).relu()
        grad = torch.empty_like(recomputed)
        capture.record_synthetic_op("aten.relu_backward.default", inputs=[recomputed], outputs=[grad])
        y.sum()

    plan = estimate_static_memory(capture.memory_events(), model_parts=[model])
    os.makedirs(args.output_dir, exist_ok=True)
    export_memory_plan(plan, args.output_dir)
    _print_plan(plan, len(capture.memory_events()), args.output_dir)


if __name__ == "__main__":
    main()
