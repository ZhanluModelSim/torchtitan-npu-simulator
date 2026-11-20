# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen SFT launch options and the published recipe import path."""

__all__ = [
    "qwen3_5_27b_sft_64k_cp8",
    "run_qwen3_5_4k",
    "run_qwen3_5_32k_cp8",
    "run_qwen3_5_64k_cp8",
    "run_qwen3_5_128k_cp8",
]

import argparse
import hashlib
import importlib.util
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torchtitan_npu  # noqa: F401 -- activate NPU patches before upstream attention imports
from torchtitan_npu_recipes.models.qwen3_5 import (
    QwenSFTCheckpointManager,
    qwen3_5_27b_sft_64k_cp8,
    run_qwen3_5_4k,
    run_qwen3_5_32k_cp8,
    run_qwen3_5_64k_cp8,
    run_qwen3_5_128k_cp8,
)


@contextmanager
def config_file_args(args: list[str]) -> Iterator[list[str]]:
    """Make the last explicit Python file importable for the native loader."""
    module_index = None
    module_value = None
    for index, arg in enumerate(args):
        if arg == "--":
            break
        if arg.startswith("--module="):
            module_index = index
            module_value = arg.removeprefix("--module=")
        elif arg == "--module" and index + 1 < len(args):
            module_index = index + 1
            module_value = args[module_index]

    if module_index is None or module_value is None or not module_value.endswith(".py"):
        yield args
        return

    path = Path(module_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--module recipe file does not exist: {path}")
    # Use the complete path so a user file cannot shadow an imported module.
    module_name = "_torchtitan_recipe_" + hashlib.sha256(bytes(path)).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load --module recipe file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_path = sys.path.copy()
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
        replacement = f"--module={module_name}" if args[module_index].startswith("--module=") else module_name
        tail = module_index + 1
        yield [*args[:module_index], replacement, *args[tail:]]
    finally:
        sys.path[:] = original_path
        sys.modules.pop(module_name, None)


_LAUNCH_FLAGS = {"--max-steps", "--smoke-test", "--init-checkpoint", "--resume-step"}
_base_recipe: ContextVar[argparse.Namespace] = ContextVar("sft_base_recipe")


@contextmanager
def sft_launch_args(args: list[str]) -> Iterator[list[str]]:
    if all(arg.partition("=")[0] not in _LAUNCH_FLAGS for arg in args):
        yield args
        return
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--module", required=True)
    parser.add_argument("--config", required=True)
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument("--max-steps", type=int)
    steps.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-step", type=int)
    original, native_args = parser.parse_known_args(args)
    if original.max_steps is not None and original.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if original.init_checkpoint is not None and not original.init_checkpoint.is_dir():
        parser.error("--init-checkpoint must name an existing directory")
    if original.smoke_test and original.resume_step is not None:
        parser.error("--smoke-test requires a fresh run. Use --max-steps for a short resume test.")
    if original.output_dir is not None:
        native_args.extend(("--output-dir", original.output_dir))
    if original.resume_step is not None:
        native_args.extend(("--resume-step", str(original.resume_step)))
    token = _base_recipe.set(original)
    try:
        yield ["--module", __name__, "--config", "launch_recipe", *native_args]
    finally:
        _base_recipe.reset(token)


def launch_recipe():
    from torchtitan.config import ConfigLoader
    from torchtitan.trainer import Trainer

    options = _base_recipe.get()
    config = ConfigLoader._load_config(options.module, options.config)
    trainer = ConfigLoader._trainer_config(config)
    if not isinstance(trainer, Trainer.Config):
        raise TypeError("Qwen SFT trial arguments require a Trainer configuration")
    training = trainer.training
    limit = 1 if options.smoke_test else options.max_steps
    if limit is not None and limit < training.steps:
        scheduler = trainer.optim.lr_scheduler
        if scheduler is not None and scheduler.total_steps is None:
            scheduler.total_steps = training.steps
        training.steps = limit
        if options.max_steps is not None and isinstance(trainer.checkpointer, QwenSFTCheckpointManager.Config):
            trainer.checkpointer.last_save_in_hf = False
            trainer.checkpointer.last_save_model_only = False
    if options.max_steps is not None and isinstance(trainer.checkpointer, QwenSFTCheckpointManager.Config):
        trainer.checkpointer.target_steps = training.steps
    if options.resume_step is not None and options.resume_step >= training.steps:
        raise ValueError(f"Target step {training.steps} must exceed resume step {options.resume_step}.")
    if options.init_checkpoint is not None:
        if trainer.checkpointer is None:
            raise ValueError("The selected recipe has no checkpoint loader for --init-checkpoint")
        trainer.checkpointer.initial_load_path = str(options.init_checkpoint.resolve())
    if options.smoke_test and trainer.checkpointer is not None:
        _require_fresh_checkpoint(trainer.checkpointer, options.output_dir or trainer.dump_folder)
        trainer.checkpointer.load_only = True
    return config


def _require_fresh_checkpoint(checkpointer, output_dir: str) -> None:
    from torchtitan.components.checkpointer.base import BaseCheckpointManager
    from torchtitan.tools import filesystem

    if checkpointer.load_step != -1 or (checkpointer.initial_load_path and not checkpointer.initial_load_model_only):
        raise ValueError("--smoke-test requires model-only initialization. Use --max-steps for a short resume test.")
    folder = filesystem.join(output_dir, checkpointer.folder)
    if not filesystem.isdir(folder):
        return
    # The upstream launcher can create structured logs while peers still load
    # the recipe. Only completed step directories can trigger auto-resume.
    for entry in filesystem.listdir(folder):
        if not re.fullmatch(BaseCheckpointManager._STEP_DIR_PATTERN, entry):
            continue
        step = filesystem.join(folder, entry)
        if any(filesystem.isfile(filesystem.join(step, name)) for name in (".metadata", "metadata.pkl")):
            raise ValueError(
                f"--smoke-test found an existing checkpoint at {step}. Use a new --output-dir to start from weights."
            )


def main() -> None:
    from torchtitan_npu.train import main as train

    original = sys.argv[1:]
    with config_file_args(original) as args, sft_launch_args(args) as native_args:
        sys.argv[1:] = native_args
        try:
            train()
        finally:
            sys.argv[1:] = original


if __name__ == "__main__":
    main()
