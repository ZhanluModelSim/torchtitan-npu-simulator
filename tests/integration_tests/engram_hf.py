# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 Engram DCP/HF roundtrip recipe and validation helpers."""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from torchtitan.components.checkpointer.base import MODEL
from torchtitan.components.optim import AdamW
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability.metrics import MetricsProcessor

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.loss_compare import TB_LOSS_TAG, extract_scalar_from_tensorboard
from torchtitan_npu.config import OptimConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.models.deepseek_v4_1.optimizer import EngramTableOptimizerConfig
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _make_trainer_config

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import AsyncMode

    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

_VALID_BACKENDS = ("host", "mxfp8")
_VALID_EXPORT_DTYPES = ("bfloat16", "float32")
_VALID_MODES = ("source", "native", "hf")
_SEQ_LEN = 512
_MXFP8_MAX_TOKENS_PER_RANK = 3072


def _parse_joint_sharding() -> bool:
    raw = os.environ.get("ENGRAM_HF_JOINT_SHARDING", "0")
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"ENGRAM_HF_JOINT_SHARDING must be 0 or 1, got {raw!r}")


def _parse_backend() -> str:
    backend = os.environ.get("ENGRAM_HF_BACKEND", "host")
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"ENGRAM_HF_BACKEND must be one of {_VALID_BACKENDS}, got {backend!r}")
    return backend


def _parse_export_dtype() -> str:
    export_dtype = os.environ.get("ENGRAM_HF_EXPORT_DTYPE", "bfloat16")
    if export_dtype not in _VALID_EXPORT_DTYPES:
        raise ValueError(f"ENGRAM_HF_EXPORT_DTYPE must be one of {_VALID_EXPORT_DTYPES}, got {export_dtype!r}")
    return export_dtype


def snapshot(
    manager: CheckpointManager,
    state_dict: dict[str, Any] | None = None,
    *,
    require_mxfp8_cache: bool = False,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    raw_state = manager.states[MODEL].state_dict() if state_dict is None else state_dict
    assert manager.sd_adapter is not None, "sd_adapter is required for Engram HF snapshots"
    state = manager.sd_adapter.to_hf(raw_state)
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
    for part_index, part in enumerate(manager.states[MODEL].model):
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
            if state_dict is not None and manager.export_dtype != module.weight.dtype:
                # Cast before quantizing so expected cache matches weights reloaded from export_dtype.
                from torchtitan_npu.override.deepseek_v4_1.engram.mxfp8 import _quantize_mxfp8_rows

                if valid > 0:
                    data, scale = _quantize_mxfp8_rows(module.weight[:valid].to(manager.export_dtype))
                else:
                    data, scale = storage[:0], module._quantized_scale[:0]
            else:
                data, scale = storage[:valid], module._quantized_scale[:valid]
            result[f"{prefix}.data"] = data.view(torch.uint8).cpu().clone()
            result[f"{prefix}.scale"] = scale.view(torch.uint8).cpu().clone()
            caches += 1
    if require_mxfp8_cache and not caches:
        raise RuntimeError("MXFP8 cache was not initialized")
    return result


def snapshot_path() -> Path:
    root = Path(os.environ["ENGRAM_HF_RUN_DIR"]) / "snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"rank-{torch.distributed.get_rank()}.pt"


class EngramHFCheckpointManager(CheckpointManager):
    """Validation-only checkpointer: save paired DCP/HF checkpoints and verify restore."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        require_mxfp8_cache: bool = False

    def __init__(self, config: Config, **kwargs: Any):
        self.require_mxfp8_cache = config.require_mxfp8_cache
        super().__init__(config=config, **kwargs)

    @torch.no_grad()
    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ):
        if to_hf:
            torch.save(
                snapshot(self, state_dict, require_mxfp8_cache=self.require_mxfp8_cache),
                snapshot_path(),
            )
            super().dcp_save(dict(state_dict), f"{checkpoint_id}-native", async_mode, to_hf=False)
        return super().dcp_save(
            state_dict,
            checkpoint_id,
            async_mode,
            enable_garbage_collection=enable_garbage_collection,
            to_hf=to_hf,
        )

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf: bool, from_quantized: bool) -> None:
        super()._load_checkpoint(states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized)
        actual = snapshot(self, require_mxfp8_cache=self.require_mxfp8_cache)
        expected = torch.load(snapshot_path(), map_location="cpu", weights_only=True)
        if actual.keys() != expected.keys():
            raise RuntimeError(
                f"Restored Engram parameter/cache keys differ: actual={sorted(actual)}, expected={sorted(expected)}"
            )
        for key in expected:
            torch.testing.assert_close(
                actual[key],
                expected[key],
                rtol=0,
                atol=0,
                msg=lambda msg, key=key: f"{key}: {msg}",
            )
        print(
            f"HF_VERIFY rank={torch.distributed.get_rank()} from_hf={from_hf} tensors={len(actual)} PASS",
            flush=True,
        )


def engram_hf() -> TrainerEx.Config:
    """Build the 4-NPU DeepSeek-V4.1 Engram DCP/HF roundtrip recipe."""
    joint = _parse_joint_sharding()
    backend = _parse_backend()
    export_dtype = _parse_export_dtype()
    mode = os.environ.get("ENGRAM_HF_MODE", "source")
    if mode not in _VALID_MODES:
        raise ValueError(f"ENGRAM_HF_MODE must be one of {_VALID_MODES}, got {mode!r}")

    ep = 2 if joint else 4
    hf_assets_path = os.environ.get("HF_ASSETS_PATH") or "tests/assets/deepseek_v3"
    config = _make_trainer_config(
        "deepseek_v4_1_debugmodel_text",
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=ep,
            fsdp_reshard_after_forward="always",
        ),
        seq_len=_SEQ_LEN,
        hf_assets_path=hf_assets_path,
    )
    config.training.steps = 2 if mode == "source" else 3
    config.training.num_tokens_per_microbatch_per_dp_rank = _SEQ_LEN
    config.training.num_tokens_per_train_step = 4 * _SEQ_LEN
    config.training.max_context_length = _SEQ_LEN
    config.training.enable_cpu_offload = False
    config.compile = None
    config.optim = OptimConfig(
        optimizer=HostSparseOptimizersContainer.Config(
            optimizers=[
                EngramTableOptimizerConfig(
                    pattern=r".*\.engram\.table\.weight$",
                    lr=5e-5,
                    betas=(0.9, 0.95),
                    eps=1e-6,
                ),
                AdamW.Config(pattern=r".*", lr=1e-4, eps=1e-8, fused=True, foreach=False),
            ]
        )
    )
    config.optim.lr_scheduler.total_steps = 20
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.metrics = MetricsProcessor.Config(log_freq=1, enable_tensorboard=True, save_tb_folder="tb")

    if backend == "mxfp8":
        config.override.imports.append(
            (
                "torchtitan_npu.override.deepseek_v4_1.engram.host_offload_mxfp8",
                {"num_max_tokens_per_rank": _MXFP8_MAX_TOKENS_PER_RANK, "shard_over_efsdp": joint},
            )
        )
    elif joint:
        config.override.imports.append("torchtitan_npu.override.deepseek_v4_1.engram.shard_over_efsdp")

    if mode == "source":
        config.checkpointer = EngramHFCheckpointManager.Config(
            interval=100,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype=export_dtype,
            load_only=False,
            require_mxfp8_cache=(backend == "mxfp8"),
        )
    else:
        run_dir = Path(os.environ["ENGRAM_HF_RUN_DIR"])
        ckpt_name = "step-2-native" if mode == "native" else "step-2"
        config.checkpointer = EngramHFCheckpointManager.Config(
            interval=100,
            load_only=True,
            initial_load_model_only=True,
            initial_load_in_hf=(mode == "hf"),
            initial_load_path=str(run_dir / "source" / "checkpoint" / ckpt_name),
            require_mxfp8_cache=(backend == "mxfp8"),
        )
    return config


def build_engram_hf_test_list() -> list[OverrideDefinitions]:
    backend = _parse_backend()
    if backend == "mxfp8":
        descr = "Engram MXFP8 {layout} HF export/load, cache rebuild and three training steps"
    else:
        descr = "Engram Host {layout} HF export/load and three training steps"
    return [
        OverrideDefinitions(
            test_name=f"dsv41_engram_{backend}_hf_{layout}",
            test_descr=descr.format(layout=layout),
            env_vars={"ENGRAM_HF_JOINT_SHARDING": joint, "ENGRAM_HF_BACKEND": backend},
            ngpu=4,
            use_golden=True,
            check_loss=False,
            python_config=True,
            timeout=1800,
            train_script="tests/integration_tests/run_engram_hf.sh",
            train_args=(),
            override_args=[()],
        )
        for layout, joint in (("ep4", "0"), ("ep2_efsdp2", "1"))
    ]


def _verify_hf_safetensors(root: Path, export_dtype: str) -> None:
    expected_dtype = TORCH_DTYPE_MAP[export_dtype]
    rank_snaps = [
        torch.load(root / "snapshots" / f"rank-{rank}.pt", map_location="cpu", weights_only=True) for rank in range(4)
    ]
    embed_keys = sorted(key for key in rank_snaps[0] if key.endswith(".engram.embed.weight"))
    if not embed_keys:
        raise RuntimeError("No exported Engram embed weights found in snapshots")

    hf_dir = root / "source" / "checkpoint" / "step-2"
    safetensor_files = sorted(hf_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(f"No consolidated safetensors files found under {hf_dir}")

    exported: dict[str, torch.Tensor] = {}
    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in embed_keys:
                if key in available:
                    exported[key] = handle.get_tensor(key)

    if sorted(exported) != embed_keys:
        raise RuntimeError(f"Exported HF Engram keys mismatch: got {sorted(exported)}, expected {embed_keys}")

    for key in embed_keys:
        tensor = exported[key]
        if tensor.dtype != expected_dtype:
            raise RuntimeError(f"{key}: expected HF dtype {expected_dtype}, got {tensor.dtype}")
        expected_full = torch.cat([snap[key] for snap in rank_snaps], dim=0)
        torch.testing.assert_close(
            tensor.float(),
            expected_full,
            rtol=0,
            atol=0,
            msg=lambda msg, key=key: f"HF safetensors {key}: {msg}",
        )
    print(f"HF_EXPORT_CHECK: {len(embed_keys)} Engram table(s) in {export_dtype} PASS")


def compare_metrics(path: str | Path) -> None:
    root = Path(path)
    export_dtype = _parse_export_dtype()
    _verify_hf_safetensors(root, export_dtype)

    records: dict[str, dict[tuple[str, int], float]] = {}
    for mode, expected_from_hf in (("native", "False"), ("hf", "True")):
        log = (root / f"{mode}.log").read_text()
        ranks = set(re.findall(rf"HF_VERIFY rank=(\d+) from_hf={expected_from_hf} .* PASS", log))
        if ranks != {"0", "1", "2", "3"}:
            raise RuntimeError(f"{mode}: missing successful per-rank Engram load checks: {ranks}")
        print(f"HF_LOAD_CHECK {mode}: all four ranks PASS")
        records[mode] = {}
        for tag in (TB_LOSS_TAG, "grad_norm"):
            values = extract_scalar_from_tensorboard(root / mode, "tb", tag)
            for step, value in values.items():
                records[mode][(tag, step)] = value

    same = True
    for step in (1, 2, 3):
        for tag in (TB_LOSS_TAG, "grad_norm"):
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
        raise SystemExit("Usage: python -m tests.integration_tests.engram_hf --compare RUN_DIR")
