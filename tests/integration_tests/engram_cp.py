# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in Engram CP training on the upstream stripped V4.1 geometry."""

import dataclasses
import sys
from pathlib import Path

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.loss_compare import extract_scalar_from_tensorboard
from torchtitan_npu.models.deepseek_v4_1 import _attach_engram, _flash_engram_geometry
from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_stripped, _v41_optimizer_config


def engram_cp():
    config = deepseek_v4_1_stripped()
    config.activation_checkpoint = None
    # Preserve the fused attention geometry. Two configured Engram layers fit
    # the stripped stack; only table rows are reduced by the launcher.
    _attach_engram(config.model_spec.model, dataclasses.replace(_flash_engram_geometry(), layer_ids=(0, len(config.model_spec.model.layers) - 1)))
    config.optimizer = _v41_optimizer_config(config.model_spec)
    return config


def build_engram_cp_test_list():
    return [
        OverrideDefinitions(
            test_name=f"dsv41_engram_{backend}_cp2_ep2",
            test_descr=f"Engram {backend} CP1/CP2 training with EP2",
            ngpu=4,
            check_loss=False,
            timeout=1800,
            env_vars={"ENGRAM_CP_BACKEND": backend},
            train_script="tests/integration_tests/run_engram_cp.sh",
            train_args=(),
            override_args=[()],
        )
        for backend in ("torch", "fp32", "mxfp8")
    ]


def summarize(root):
    # Different CP degrees change floating-point reduction order. Report the
    # paired values for review; completion alone is not numerical equivalence.
    for tag in ("loss_metrics/global_avg_loss", "grad_norm"):
        values = [extract_scalar_from_tensorboard(root / mode, "tb", tag) for mode in ("cp1", "cp2")]
        if any(set(record) != {1, 2, 3} for record in values):
            raise RuntimeError(f"{tag}: expected exactly three completed steps")
        for step in (1, 2, 3):
            baseline, sharded = (record[step] for record in values)
            print(f"{tag} step={step} cp1={baseline:.9g} cp2={sharded:.9g} delta={sharded-baseline:.9g}")
    print("CP1/CP2 completed; numerical differences above require review")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
