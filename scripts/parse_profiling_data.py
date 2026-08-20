# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline parser for profiling data collected with enable_online_parse=False.

Usage:
    python3 scripts/parse_profiling_data.py path/to/xxx_ascend_pt
    python3 scripts/parse_profiling_data.py path/to/profiling_data
    python3 scripts/parse_profiling_data.py path/to/save_traces_folder
"""

import glob
import logging
import os
import sys

import torch_npu

logging.basicConfig(level=logging.INFO)


def _find_ascend_pt_dirs(path: str) -> list[str]:
    path = os.path.abspath(path)
    if os.path.basename(path).endswith("_ascend_pt") and os.path.isdir(path):
        return [path]

    candidates = sorted(glob.glob(os.path.join(path, "*_ascend_pt")))
    nested = sorted(glob.glob(os.path.join(path, "profiling_data", "*_ascend_pt")))
    seen = set(candidates + nested)
    return sorted(seen)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error(
            "Usage: python3 scripts/parse_profiling_data.py <path/to/xxx_ascend_pt | path/to/save_traces_folder>"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    ascend_pt_dirs = _find_ascend_pt_dirs(input_path)
    if not ascend_pt_dirs:
        logging.error("No *_ascend_pt directories found under %s", input_path)
        sys.exit(1)

    logging.info(
        "Found %d *_ascend_pt director%s to parse", len(ascend_pt_dirs), "y" if len(ascend_pt_dirs) == 1 else "ies"
    )
    for i, d in enumerate(ascend_pt_dirs, 1):
        logging.info("[%d/%d] Parsing %s", i, len(ascend_pt_dirs), d)
        torch_npu.profiler.profiler.analyse(profiler_path=d)
