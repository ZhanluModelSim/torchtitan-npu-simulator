# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a small multi-block inference workload as semantic JSON.

Example (from the repository root):
    python scripts/block_diffusion/export_example.py --tp 2 --ep 4 --output /tmp/block-diff.json

This schema requires a consumer mapping; it is not TorchTitan's native training IR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from scripts.block_diffusion.workload import ParallelSpec, analyze_schedule, build_schedule
    from torchtitan_npu.simulator.raw_model.block_diff.model import ModelConfig
else:
    from raw_reference import ModelConfig
    from workload import ParallelSpec, analyze_schedule, build_schedule


def build_example(tp: int = 1, ep: int = 1) -> dict:
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_shared_experts=1,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    schedule = build_schedule(
        prompt_length=8,
        block_lengths=(4, 4),
        commit_counts=((2, 2), (2, 2)),
        batch_size=2,
        commit_last=False,
    )
    return analyze_schedule(config, schedule, ParallelSpec(tp_size=tp, ep_size=ep)).to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_example(args.tp, args.ep)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "phases": len(report["phases"]),
                    "ranks_per_phase": args.tp * args.ep,
                    "generated_tokens": report["generated_tokens"],
                    "scope": report["scope"],
                },
                indent=2,
            )
        )
