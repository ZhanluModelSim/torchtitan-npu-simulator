import functools
import logging

from torchtitan.components.metrics import MetricsProcessor

from torchtitan_npu_lite.patches.torchtitan.models.common.aux_loss import (
    collect_aux_loss_metrics,
)

logger = logging.getLogger(__name__)

original_log = MetricsProcessor.log


@functools.wraps(original_log)
def patched_log(
    self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
):
    """Objective: merging ``collect_aux_loss_metrics`` into ``extra_metrics``."""
    extra_metrics = {
        **(extra_metrics or {}),
        **collect_aux_loss_metrics(self.model_parts, self.parallel_dims),
    }
    original_log(
        self,
        step,
        global_avg_loss,
        global_max_loss,
        grad_norm,
        extra_metrics=extra_metrics,
    )


def apply() -> None:
    logger.info("[PATCH] MetricsProcessor.log -> patched_log")
    MetricsProcessor.log = patched_log


apply()
