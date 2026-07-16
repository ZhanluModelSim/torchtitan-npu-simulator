# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_script_module():  # noqa: ANN202
    script_path = Path(__file__).parents[3] / "scripts" / "analyze_simulator_memory.py"
    spec = importlib.util.spec_from_file_location("analyze_simulator_memory", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_groups_only_tensors_retained_past_forward(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _write_csv(
        memory_dir / "memory_events.csv",
        ["seq_idx", "phase", "op_id", "module_path"],
        [
            {"seq_idx": 4, "phase": "forward", "op_id": 10, "module_path": "layers.0._checkpoint_wrapped_module"},
            {"seq_idx": 7, "phase": "forward", "op_id": 11, "module_path": "layers.1._checkpoint_wrapped_module"},
            {"seq_idx": 8, "phase": "backward", "op_id": 12, "module_path": "layers.1"},
        ],
    )
    lifetime_columns = [
        "tensor_id", "kind", "num_bytes", "birth_seq", "death_seq", "producer_op", "producer_raw_op",
        "producer_phase", "consumer_ops", "consumer_seqs", "consumer_phases", "alias_of", "shape", "dtype", "reason",
    ]
    _write_csv(
        memory_dir / "tensor_lifetimes.csv",
        lifetime_columns,
        [
            {
                "tensor_id": "tensor:1", "kind": "activation", "num_bytes": 1024, "birth_seq": 4, "death_seq": 8,
                "producer_op": 10, "producer_raw_op": "aten.add", "producer_phase": "forward", "consumer_ops": "12",
                "consumer_seqs": "8", "consumer_phases": "backward", "alias_of": "", "shape": "[1,8]",
                "dtype": "torch.bfloat16", "reason": "forward_to_backward",
            },
            {
                "tensor_id": "tensor:2", "kind": "checkpoint_recompute_temp", "num_bytes": 2048, "birth_seq": 7,
                "death_seq": 7, "producer_op": 11, "producer_raw_op": "aten.mul", "producer_phase": "forward",
                "consumer_ops": "11", "consumer_seqs": "7", "consumer_phases": "forward", "alias_of": "",
                "shape": "[1,16]", "dtype": "torch.bfloat16", "reason": "checkpoint_internal_recompute",
            },
        ],
    )
    (memory_dir / "memory_summary.json").write_text(
        json.dumps({"persistent_param_bytes": 4096, "active_bytes_peak": 8192}), encoding="utf-8"
    )

    module = _load_script_module()
    analysis = module.analyze_memory_export(tmp_path)

    assert analysis["forward_end_seq"] == 7
    assert analysis["resident_count"] == 1
    assert analysis["resident_bytes"] == 1024
    assert analysis["by_kind"] == {"activation": 1024}
    assert analysis["groups"][0]["layer"] == "layer.0"
    assert analysis["suspicious_checkpoint_count"] == 1
