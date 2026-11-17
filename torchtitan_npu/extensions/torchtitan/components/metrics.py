# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print elapsed_time_per_step to the console after TorchTitan's step line.

TorchTitan logs time_metrics/end_to_end(s) only to structured metrics; the
step console line omits it. Wrap MetricsProcessor.log so the per-step elapsed
time is printed to the console on its own line, matching the console output
of the deprecated patches/tools/metrics.py.

Importing this module applies the extension.
"""

import functools
import time

import torchtitan.components.metrics
from torchtitan.tools.logging import logger as titan_logger

_original_log = torchtitan.components.metrics.MetricsProcessor.log


@functools.wraps(_original_log)
def _patched_log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
    """Print elapsed_time_per_step in console output."""
    time_delta = time.perf_counter() - self.time_last_log
    _original_log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)
    time_end_to_end = time_delta / self.config.log_freq
    color = self.color
    titan_logger.info(f"{color.yellow}elapsed_time_per_step: {time_end_to_end:.3f}s{color.reset}")


torchtitan.components.metrics.MetricsProcessor.log = _patched_log
