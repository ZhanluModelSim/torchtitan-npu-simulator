#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EMA checkpoint smoke coverage for the default model integration suite."""

from pathlib import Path

import torch.distributed.checkpoint as dcp

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.deepseek_v4 import GOLDEN_OVERRIDES

_EMA_CHECKPOINT_PREFIX = "ema_optimizer."


def build_ema_test_list() -> list[OverrideDefinitions]:
    """Return the lightweight EMA feature smoke case run by ``models``."""

    return [
        OverrideDefinitions(
            override_args=[
                (
                    *GOLDEN_OVERRIDES,
                    "--training.steps=2",
                    "--parallelism.expert-parallel-degree=2",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--training.global-batch-size=2",
                    "--ema-weights.enable",
                    "--ema-weights.decay=0.5",
                    "--ema-weights.offload-to-cpu",
                    "--checkpoint.enable",
                    "--checkpoint.interval=1",
                    "--checkpoint.no-last-save-model-only",
                )
            ],
            test_descr="DeepSeek-V4 EMA EP2/FSDP2 CPU-offload checkpoint smoke",
            test_name="dsv4_ema_ep2_fsdp2",
            ngpu=2,
            use_golden=True,
            check_loss=False,
            verify_ema_checkpoint=True,
        )
    ]


def assert_ema_checkpoint_written(test_run_dir: Path) -> None:
    """Assert that the latest complete DCP checkpoint contains EMA state."""

    checkpoint_root = test_run_dir / "checkpoint"
    checkpoint_dirs = sorted(
        path
        for path in checkpoint_root.glob("step-*")
        if path.is_dir() and (path / ".metadata").is_file()
    )
    if not checkpoint_dirs:
        raise AssertionError(f"EMA smoke did not write a complete DCP checkpoint under {checkpoint_root}")

    checkpoint_dir = checkpoint_dirs[-1]
    metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
    if not any(key.startswith(_EMA_CHECKPOINT_PREFIX) for key in metadata.state_dict_metadata):
        raise AssertionError(f"EMA state ({_EMA_CHECKPOINT_PREFIX}*) is missing from DCP checkpoint {checkpoint_dir}")
