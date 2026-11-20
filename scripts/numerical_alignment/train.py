# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Native Trainer entry point for shared DCP seed creation and loading.

Selected by loss_compare via TRAIN_FILE. The ordinary example runner still
selects the model, optimizer, overrides, parallelism and training configuration.
Optional checkpointers and the suppressed seed field are set after CLI parsing.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path


def main() -> None:
    if Path(__file__).resolve().parents[2] != Path.cwd().resolve():
        raise ValueError("comparison entry point must run from its own selected repository root")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--comparison-create-seed-checkpoint", action="store_true")
    parser.add_argument("--comparison-checkpoint-folder", type=Path)
    parser.add_argument("--comparison-initial-load-path", type=Path)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--dump_folder", "--dump-folder", dest="output_dir_legacy")
    parser.add_argument("--training.max-context-length", "--training.seq-len", dest="seq_len", type=int)
    parser.add_argument("--training.steps", dest="steps", type=int)
    parser.add_argument("--debug.seed", dest="seed", type=int)
    parser.add_argument("--debug.deterministic", dest="deterministic", action="store_true")
    parser.add_argument(
        "--metrics.enable-tensorboard", "--metrics.enable_tensorboard", dest="enable_tensorboard", action="store_true"
    )
    parser.add_argument("--metrics.log-freq", "--metrics.log_freq", dest="log_freq", type=int)
    parser.add_argument("--metrics.save-tb-folder", "--metrics.save_tb_folder", dest="save_tb_folder")
    controls, training_args = parser.parse_known_args()
    if controls.comparison_create_seed_checkpoint and controls.comparison_initial_load_path:
        parser.error("seed creation and initial checkpoint loading are mutually exclusive")

    seed_only = ("--optim.optimizer.name", "--parallelism.")
    for argument in training_args:
        if argument.startswith(seed_only):
            parser.error(f"{argument.split('=', 1)[0]} is seed-only; choose a Python recipe")

    import torch.distributed as dist
    from torchtitan.components.optim import AdamW
    from torchtitan.config import ConfigLoader
    from torchtitan.observability import structured_logger as sl
    from torchtitan.observability.logging import init_logger

    import torchtitan_npu
    from torchtitan_npu.config import OptimizerConfig

    init_logger()
    logger = logging.getLogger(__name__)
    config = ConfigLoader().load(training_args)
    output_dir = controls.output_dir or controls.output_dir_legacy
    if output_dir is not None:
        config.dump_folder = output_dir
    if controls.seq_len is not None:
        config.training.max_context_length = controls.seq_len
        config.training.num_tokens_per_microbatch_per_dp_rank = controls.seq_len
    if controls.steps is not None:
        config.training.steps = controls.steps
    if controls.seed is not None:
        config.debug.seed = controls.seed
    if controls.deterministic:
        config.debug.deterministic = True
    if controls.enable_tensorboard:
        config.metrics.enable_tensorboard = True
    if controls.log_freq is not None:
        config.metrics.log_freq = controls.log_freq
    if controls.save_tb_folder is not None:
        config.metrics.save_tb_folder = controls.save_tb_folder
    if controls.comparison_create_seed_checkpoint:
        # Build the seed from the same model recipe while avoiding legacy CLI
        # overrides. A one-process AdamW setup is independent of the training
        # topology and does not require DistMuon's distributed parameter mesh.
        config.parallelism = replace(
            config.parallelism,
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        )
        config.optim.optimizer = OptimizerConfig(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-5,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
                    fused=True,
                    foreach=False,
                )
            ]
        )
    if controls.comparison_create_seed_checkpoint or controls.comparison_initial_load_path:
        from torchtitan.components.checkpointer import CheckpointManager

        checkpointer = config.checkpointer or CheckpointManager.Config()
        if not isinstance(checkpointer, CheckpointManager.Config):
            raise ValueError("shared comparison checkpoints require a DCP checkpointer")
        folder = controls.comparison_checkpoint_folder or Path(config.dump_folder) / "checkpoint"
        folder = folder.resolve()
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"comparison checkpoint folder must be empty: {folder}")
        if controls.comparison_create_seed_checkpoint:
            if int(os.environ["WORLD_SIZE"]) != 1:
                raise ValueError("seed creation requires one process")
            if controls.comparison_checkpoint_folder is None:
                parser.error("seed creation requires --comparison-checkpoint-folder")
            config.create_seed_checkpoint = True
            config.checkpointer = replace(
                checkpointer,
                folder=str(folder),
                load_only=False,
                last_save_model_only=True,
                last_save_in_hf=False,
                export_dtype="float32",
                initial_load_path=None,
            )
        else:
            checkpoint = controls.comparison_initial_load_path.resolve()
            if not checkpoint.is_dir():
                raise ValueError(f"initial checkpoint does not exist: {checkpoint}")
            config.checkpointer = replace(
                checkpointer,
                folder=str(folder),
                load_only=True,
                initial_load_path=str(checkpoint),
                initial_load_model_only=True,
                initial_load_in_hf=False,
            )
            logger.info("Comparison loads shared model-only checkpoint: %s", checkpoint)

    import torchtitan

    logger.info("Comparison code: torchtitan=%s; torchtitan_npu=%s", torchtitan.__file__, torchtitan_npu.__file__)
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    trainer = None
    try:
        trainer = config.build()
        if config.create_seed_checkpoint:
            trainer.engine.save_checkpoint(last_step=True)
            logger.info("Created shared seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer is not None:
            trainer.close()
        raise
    else:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
