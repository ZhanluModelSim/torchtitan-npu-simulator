# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay a small two-block workload without external simulator dependencies.

Run from the repository root:
    python scripts/block_diffusion/example.py --device cpu
    python scripts/block_diffusion/example.py --device meta --moe

The reveal order is supplied by this example, not inferred from model quality.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torchtitan_npu.simulator.raw_model.block_diff.model import (
        BlockDiffusionModel,
        KVCache,
        ModelConfig,
        RoutingPlan,
    )
else:
    from raw_reference import BlockDiffusionModel, KVCache, ModelConfig, RoutingPlan


def replay(device: str = "cpu", moe: bool = False) -> dict:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        num_experts=4 if moe else 0,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        num_shared_experts=1 if moe else 0,
        dtype=torch.float32,
    )
    torch.manual_seed(7)
    with torch.device(device):
        model = BlockDiffusionModel(config).eval()
    cache: KVCache | None = None
    phases = []

    def run(tokens, phase, positions, attention):
        nonlocal cache
        routing = RoutingPlan.balanced(tokens.numel(), config.num_experts, config.num_experts_per_tok) if moe else None
        prefix = 0 if cache is None else cache.length
        output = model(
            tokens,
            cache,
            phase=phase,
            attention_policy=attention,
            logits_positions=positions,
            routing_plan=routing,
        )
        cache = output.cache
        phases.append(
            {
                "phase": phase,
                "input_shape": list(tokens.shape),
                "prefix_before": prefix,
                "cache_length_after": 0 if cache is None else cache.length,
                "logits_shape": None if output.logits is None else list(output.logits.shape),
            }
        )
        return output

    with torch.inference_mode():
        prompt = torch.zeros((2, 3), dtype=torch.long, device=device)
        run(prompt, "prefill", (), "causal")
        for block in range(2):
            canvas = torch.ones((2, 4), dtype=torch.long, device=device)
            for step in range(2):
                # The two requests reveal opposite ends first. Counts agree,
                # but each request needs its own position map.
                positions = torch.tensor(
                    [list(range(step * 2, 4)), list(range(3 - step * 2, -1, -1))],
                    dtype=torch.long,
                    device=device,
                )
                output = run(canvas, "denoise", positions, "bidirectional")
                predicted = output.logits[:, :2].argmax(-1)
                canvas = canvas.scatter(1, positions[:, :2], predicted)
            # The final block has no later cache consumer in this workload.
            if block == 0:
                run(canvas, "commit", (), "bidirectional")

    assert cache is not None, "prefill must produce a cache"
    return {
        "device": device,
        "architecture": "moe" if moe else "dense",
        "generated_tokens_per_request": 8,
        "generated_tokens_total": 16,
        "parameter_bytes": model.parameter_memory_bytes(),
        "retained_cache_bytes": cache.memory_bytes(),
        "phases": phases,
        "scope": "Fixed reveal-order replay; no quality or hardware latency estimate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "meta"), default="cpu")
    parser.add_argument("--moe", action="store_true")
    options = parser.parse_args()
    print(json.dumps(replay(options.device, options.moe), indent=2))
