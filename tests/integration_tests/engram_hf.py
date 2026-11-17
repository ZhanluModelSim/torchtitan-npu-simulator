# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation-only Trainer entry: paired checkpoints and Engram load checks."""

import math
import os
import re
import sys
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchtitan.train import main

import torchtitan_npu  # noqa: F401
from tests.integration_tests import OverrideDefinitions
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager

_save = CheckpointManager.dcp_save
_load = CheckpointManager.dcp_load


def snapshot(manager):
    result = {}
    state = manager.sd_adapter.to_hf(manager.states["model"].state_dict())
    for key, value in state.items():
        if ".engram." not in key:
            continue
        if hasattr(value, "_engram_valid_rows"):
            value = value[: value._engram_valid_rows]
        elif hasattr(value, "to_local"):
            value = value.to_local()
        result[key] = value.detach().cpu().float().clone()
    if not result:
        raise RuntimeError("No Engram parameters found in checkpoint")
    caches = 0
    for part_index, part in enumerate(manager.states["model"].model):
        for name, module in part.named_modules():
            storage = getattr(module, "_quantized_storage", None)
            if storage is None:
                continue
            rows = module.weight.shape[0]
            mesh = module.lookup_mesh
            shard_rank = mesh.get_local_rank() if mesh is not None else 0
            # HF omits padding; joint storage uses the EFSDP x EP shard rank.
            valid = max(0, min(rows, module.logical_num_embeddings - shard_rank * rows))
            prefix = f"cache.{part_index}.{name}"
            result[f"{prefix}.data"] = storage[:valid].view(torch.uint8).cpu().clone()
            result[f"{prefix}.scale"] = module._quantized_scale[:valid].view(torch.uint8).cpu().clone()
            caches += 1
    if not caches:
        raise RuntimeError("MXFP8 cache was not initialized")
    return result


def snapshot_path():
    root = Path(os.environ["ENGRAM_HF_RUN_DIR"]) / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"rank-{torch.distributed.get_rank()}.pt"


def save(self, state_dict, checkpoint_id, async_mode, enable_garbage_collection=False, to_hf=False):
    if to_hf:
        torch.save(snapshot(self), snapshot_path())
        _save(self, state_dict, str(checkpoint_id) + "-native", async_mode, to_hf=False)
    return _save(self, state_dict, checkpoint_id, async_mode, enable_garbage_collection, to_hf)


def load(self, state_dict, checkpoint_id, from_hf=False, from_quantized=False):
    _load(self, state_dict, checkpoint_id, from_hf, from_quantized)
    actual = snapshot(self)
    expected = torch.load(snapshot_path(), map_location="cpu", weights_only=True)
    if actual.keys() != expected.keys():
        raise RuntimeError("Restored Engram parameter/cache keys differ")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=lambda msg, key=key: f"{key}: {msg}")
    print(f"HF_VERIFY rank={torch.distributed.get_rank()} from_hf={from_hf} tensors={len(actual)} PASS", flush=True)


def build_engram_hf_test_list():
    return [
        OverrideDefinitions(
            test_name=f"dsv41_engram_mxfp8_hf_{layout}",
            test_descr=f"Engram {layout} HF load, MXFP8 cache rebuild and three training steps",
            env_vars={"ENGRAM_HF_JOINT_SHARDING": joint},
            ngpu=4,
            check_loss=False,
            timeout=1800,
            train_script="tests/integration_tests/run_engram_hf.sh",
            train_args=(),
            override_args=[()],
        )
        for layout, joint in (("ep4", "0"), ("ep2_efsdp2", "1"))
    ]


def compare_metrics(path):
    root = Path(path)
    records = {}
    for mode in ("native", "hf"):
        log = (root / f"{mode}.log").read_text()
        ranks = set(re.findall(r"HF_VERIFY rank=(\d+) .* PASS", log))
        if ranks != {"0", "1", "2", "3"}:
            raise RuntimeError(f"{mode}: missing successful per-rank Engram load checks: {ranks}")
        print(f"HF_LOAD_CHECK {mode}: all four ranks PASS")
        records[mode] = {}
        for file in sorted((root / mode / "tb").rglob("events.out.tfevents.*")):
            events = EventAccumulator(str(file), size_guidance={"scalars": 0}).Reload()
            for tag in ("loss_metrics/global_avg_loss", "grad_norm"):
                if tag not in events.Tags()["scalars"]:
                    continue
                for event in events.Scalars(tag):
                    key = (tag, event.step)
                    if key in records[mode] and records[mode][key] != event.value:
                        raise RuntimeError(f"Conflicting events: {mode} {key}")
                    records[mode][key] = event.value
    same = True
    for step in (1, 2, 3):
        for tag in ("loss_metrics/global_avg_loss", "grad_norm"):
            a = records["native"][(tag, step)]
            b = records["hf"][(tag, step)]
            equal = math.isfinite(a) and math.isfinite(b) and a == b
            same = same and equal
            print(f"HF_COMPARE step={step} {tag} native={a:.17g} hf={b:.17g} equal={equal}")
    if not same:
        raise SystemExit("Metrics differ; retain logs for diagnosis.")
    print("HF_ROUNDTRIP PASS: Engram restore and three training steps match.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--compare":
        compare_metrics(sys.argv[2])
    else:
        CheckpointManager.dcp_save = save
        CheckpointManager.dcp_load = load
        main()
