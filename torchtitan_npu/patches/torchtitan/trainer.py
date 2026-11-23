# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/6290

"""Patch ``Trainer.Config.maybe_log`` to log compact folded JSON.

Keeps ``debug.save_config_file`` backed by the full ``to_dict()`` payload.
Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os

import torch
import torchtitan.trainer

from torchtitan_npu.patches.torchtitan.config.format import format_config

logger = logging.getLogger(__name__)


def patched_maybe_log(self: torchtitan.trainer.Trainer.Config) -> None:
    """Log compact folded JSON for ``debug.print_config`` while keeping ``save_config_file`` full."""
    if self.debug.print_config:
        torchtitan.trainer.logger.info(f"Running with configs: {format_config(self)}")

    if self.debug.save_config_file is not None:
        config_file = os.path.join(self.dump_folder, self.debug.save_config_file)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, "w") as f:
                    json.dump(self.to_dict(), f, indent=2)
            torchtitan.trainer.logger.info(f"Saved job configs to {config_file}")
        else:
            torchtitan.trainer.logger.warning(
                "Job configs logging is disabled due to torch.distributed not initialized."
            )


def apply() -> None:
    if importlib.util.find_spec("torchtitan.config.format") is not None:
        return
    logger.info("[PATCH] Trainer.Config.maybe_log -> patched_maybe_log (upstream #6290)")
    torchtitan.trainer.Trainer.Config.maybe_log = patched_maybe_log  # type: ignore[method-assign]


apply()
