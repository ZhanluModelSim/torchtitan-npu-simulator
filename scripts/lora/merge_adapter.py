# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Merge LoRA into a floating-point checkpoint without changing its format."""

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from torchtitan_npu.models.deepseek_v4.lora import build_lora_merge_plan, merge_lora_weight


def _load_adapter_config(adapter_dir: str) -> dict:
    path = os.path.join(adapter_dir, "adapter_config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Expected peft_type=LORA, got {config.get('peft_type')!r}")
    if config.get("use_dora") or config.get("use_rslora"):
        raise NotImplementedError("Only plain LoRA (alpha/r) is supported; DoRA and RSLoRA are not")
    if config.get("rank_pattern") or config.get("alpha_pattern") or config.get("fan_in_fan_out"):
        raise NotImplementedError("Per-target rank/alpha and transposed LoRA weights are not supported")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    return config


def _load_base_weight_map(base_dir: str) -> dict[str, str]:
    """Return {tensor_name: shard_filename}, for sharded or single-file checkpoints."""
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]

    single_path = os.path.join(base_dir, "model.safetensors")
    if os.path.exists(single_path):
        with safe_open(single_path, framework="pt") as handle:
            return dict.fromkeys(handle.keys(), "model.safetensors")

    raise FileNotFoundError(f"Found neither model.safetensors.index.json nor model.safetensors under {base_dir}")


def _validate_output_directory(base_dir: str, adapter_dir: str, output_dir: str) -> None:
    resolved_output = os.path.realpath(output_dir)
    protected_inputs = {
        os.path.realpath(base_dir),
        os.path.realpath(adapter_dir),
    }
    if any(os.path.commonpath((resolved_output, input_dir)) == input_dir for input_dir in protected_inputs):
        raise ValueError("output directory must be outside the base model and adapter directories")
    if os.path.exists(output_dir):
        raise FileExistsError(f"output directory already exists: {output_dir}")


def _shard_shapes(handle, weight_map, shard):
    shapes, dtypes = {}, {}
    available = set(handle.keys())
    for key in (key for key in weight_map if weight_map[key] == shard):
        if key not in available:
            raise ValueError(f"Checkpoint index refers to missing tensors: {key}")
        value = handle.get_slice(key)
        shapes[key] = tuple(value.get_shape())
        dtypes[key] = value.get_dtype()
    return shapes, dtypes


def _base_shapes(base_dir, weight_map):
    shapes, dtypes = {}, {}
    for shard in sorted(set(weight_map.values())):
        if os.path.basename(shard) != shard:
            raise ValueError(f"Expected a checkpoint shard filename, got {shard!r}")
        with safe_open(os.path.join(base_dir, shard), framework="pt") as handle:
            metadata = handle.metadata() or {}
            if any("_weight_qdata" in key or "_weight_scale" in key for key in list(handle.keys())) or any(
                "_type" in value and "_tensor_data_names" in value for value in metadata.values()
            ):
                raise ValueError("Decode the quantized base with its backend into plain FP16/BF16/FP32 logical weights")
            shard_shapes, shard_dtypes = _shard_shapes(handle, weight_map, shard)
            shapes.update(shard_shapes)
            dtypes.update(shard_dtypes)
    return shapes, dtypes


def _adapter_plans(handle, weight_map, rank, base_shapes, expected_experts):
    # Meta tensors validate every pair without allocating adapter storage.
    adapter = {key: torch.empty(handle.get_slice(key).get_shape(), device="meta") for key in list(handle.keys())}
    options = dict(rank=rank, base_shapes=base_shapes, expected_experts=expected_experts)
    build_lora_merge_plan(adapter, weight_map, **options)
    plans = {}
    for a_key in sorted(key for key in adapter if key.endswith(".lora_A.weight")):
        b_key = a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        targets = build_lora_merge_plan({key: adapter[key] for key in (a_key, b_key)}, weight_map, **options)
        plans[(a_key, b_key)] = set(targets)
    return plans, options


@dataclass
class _MergeJob:
    weight_map: dict
    plans: dict
    options: dict
    scaling: float


def _write_merged_shards(base_dir, output_dir, adapter, job):
    weight_map, plans, options, scaling = job.weight_map, job.plans, job.options, job.scaling
    targets = set().union(*plans.values())
    shards = {weight_map[key] for key in targets}
    for name in os.listdir(base_dir):
        source = os.path.join(base_dir, name)
        if name not in shards and os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, name))
    for shard in sorted(shards):
        with safe_open(os.path.join(base_dir, shard), framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in list(handle.keys())}
        for pair, keys in plans.items():
            selected = {key for key in keys if weight_map[key] == shard}
            if not selected:
                continue
            factors = {key: adapter.get_tensor(key) for key in pair}
            plan = build_lora_merge_plan(factors, weight_map, **options)
            for key in selected:
                tensors[key] = merge_lora_weight(tensors[key], *plan[key], scaling)
            del plan, factors
        save_file(tensors, os.path.join(output_dir, shard), metadata=metadata)
        del tensors
    return len(targets)


def merge(base_dir: str, adapter_dir: str, output_dir: str) -> dict:
    _validate_output_directory(base_dir, adapter_dir, output_dir)
    base_config = {}
    config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as handle:
            base_config = json.load(handle)
        if base_config.get("quantization_config") or base_config.get("expert_dtype") in ("fp4", "fp8"):
            raise ValueError("Prepare a floating-point checkpoint with its quantization backend before merging")
    config = _load_adapter_config(adapter_dir)
    weight_map = _load_base_weight_map(base_dir)
    shapes, dtypes = _base_shapes(base_dir, weight_map)
    experts = base_config.get("n_routed_experts", base_config.get("n_experts"))
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as adapter:
        plans, options = _adapter_plans(adapter, weight_map, config["r"], shapes, experts)
        for key in set().union(*plans.values()):
            if dtypes[key] not in ("F16", "BF16", "F32"):
                raise ValueError(f"Decode the quantized base weight before merging: {key}")
        parent = os.path.dirname(os.path.abspath(output_dir))
        os.makedirs(parent, exist_ok=True)
        temporary = tempfile.mkdtemp(prefix=".lora-merge-", dir=parent)
        try:
            count = _write_merged_shards(
                base_dir, temporary, adapter, _MergeJob(weight_map, plans, options, config["lora_alpha"] / config["r"])
            )
            if os.path.exists(output_dir):
                raise FileExistsError(f"output directory already exists: {output_dir}")
            os.rename(temporary, output_dir)
        finally:
            if os.path.exists(temporary):
                shutil.rmtree(temporary)
    return {"merged_count": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
