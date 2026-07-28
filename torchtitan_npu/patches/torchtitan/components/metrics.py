# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3634

import functools
import logging

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger as titan_logger

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import (
    collect_aux_loss_metrics,
)

logger = logging.getLogger(__name__)

original_log = MetricsProcessor.log


@functools.wraps(original_log)
def patched_log(
    self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
):
    """Add auxiliary losses to step metrics and structured logs."""
    aux_metrics = collect_aux_loss_metrics(self.model_parts, self.parallel_dims)
    original_log(
        self,
        step,
        global_avg_loss,
        global_max_loss,
        grad_norm,
        extra_metrics={**(extra_metrics or {}), **aux_metrics},
    )

    if aux_metrics:
        sl.log_trace_scalar(aux_metrics)
        values = "  ".join(
            f"{name}: {value:.6f}" for name, value in sorted(aux_metrics.items())
        )
        titan_logger.info(f"aux_loss step: {step:2}  {values}")


def apply() -> None:
    logger.info("[PATCH] MetricsProcessor.log -> patched_log")
    MetricsProcessor.log = patched_log


apply()
