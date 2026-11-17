# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)


class LossSpikeDetector(Configurable):
    """Flag a loss spike and report the step where it started.

    The loss is smoothed with Holt's linear method -- a running level plus a
    running trend -- and each step is scored against the one-step-ahead
    prediction ``level + trend``. A spike is a residual more than
    ``z_threshold`` standard deviations above that prediction.

    Tracking the trend is what makes this work on a real run. Training loss
    declines, and a level-only average always lags a declining series; that
    persistent lag becomes the dominant term in the residual variance and
    widens the band far past the actual noise. Measured on a decaying curve
    with noise of 0.045, a level-only detector reported sigma of 0.31 and
    scored a 1.9-nat spike at only 3.9 sigma -- under a threshold of 6, it
    never fired at any spike amplitude. Predicting the trend leaves residuals
    the size of the noise, and the same spike scores above 18.

    Reporting the *onset* matters more than reporting the trigger: a spike
    usually builds over several steps, and rolling back to a checkpoint taken
    after it began would restore an already-damaged state. The onset is the
    earliest step of the contiguous run that stayed above the looser
    ``onset_z_threshold``.

    Every rank feeds this the same all-reduced loss, so every rank reaches the
    same verdict without a further collective.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        z_threshold: float = 6.0
        """Standard deviations above the predicted loss that count as a spike."""

        warmup_steps: int = 100
        """Observations to collect before the detector may trigger."""

        level_decay: float = 0.97
        """Decay of the running level and residual variance. Closer to 1 is
        slower. The effective window is about ``1 / (1 - level_decay)`` steps."""

        trend_decay: float = 0.9
        """Decay of the running trend estimate. Lower adapts the slope faster."""

        onset_z_threshold: float = 2.0
        """Looser bar used to walk back from the trigger to the spike's start."""

        onset_lookback: int = 64
        """How many recent steps the onset search may walk back through."""

        cooldown_steps: int = 500
        """Minimum actual optimizer updates between triggers, including recovery updates."""

        def __post_init__(self) -> None:
            if not math.isfinite(self.z_threshold) or self.z_threshold <= 0.0:
                raise ValueError("z_threshold must be greater than 0.")
            if not math.isfinite(self.onset_z_threshold) or self.onset_z_threshold <= 0.0:
                raise ValueError("onset_z_threshold must be greater than 0.")
            if self.onset_z_threshold > self.z_threshold:
                raise ValueError(
                    "onset_z_threshold must not exceed z_threshold; the onset "
                    "search uses a looser bar than the trigger."
                )
            if not 0.0 < self.level_decay < 1.0:
                raise ValueError("level_decay must be in (0, 1).")
            if not 0.0 <= self.trend_decay < 1.0:
                raise ValueError("trend_decay must be in [0, 1).")
            if self.warmup_steps < 2:
                raise ValueError("warmup_steps must be at least 2.")
            if self.onset_lookback < 1:
                raise ValueError("onset_lookback must be at least 1.")
            if self.cooldown_steps < 0:
                raise ValueError("cooldown_steps cannot be negative.")

    def __init__(self, config: Config) -> None:
        self.config = config
        self._recent: deque[tuple[int, float]] = deque(maxlen=config.onset_lookback)
        self._completed_updates = 0
        self._last_trigger_update = -config.cooldown_steps - 1
        self.reset()

    def record_optimizer_update(self) -> None:
        """Advance the process-local clock; checkpoint rollback never rewinds it."""
        self._completed_updates += 1

    def reset(self) -> None:
        """Forget the loss history.

        Called after a rollback, and when anticipatory mode ends, because the
        run continues on a different trajectory and the old statistics no
        longer describe it. The cooldown deliberately survives a reset.
        """
        self._level = 0.0
        self._trend = 0.0
        self._var = 0.0
        self._count = 0
        self._frozen_steps = 0
        self._recent.clear()

    @property
    def predicted_loss(self) -> float:
        """One-step-ahead prediction the next observation is scored against."""
        return self._level + self._trend

    @property
    def sigma(self) -> float:
        """Standard deviation of the residual around that prediction."""
        return math.sqrt(max(self._var, 0.0))

    def observe(self, step: int, loss: float) -> int | None:
        """Record one step's global loss and return the onset of a spike.

        Returns ``None`` when there is no spike to act on.
        """
        if not math.isfinite(loss):
            # A non-finite loss already aborts the run in
            # TrainingEngine.optimizer_step; there is nothing to roll back to
            # that this detector could choose better.
            return None

        config = self.config
        onset: int | None = None
        anomalous = False

        if self._count >= config.warmup_steps:
            sigma_floor = max(1e-12, 1e-6 * max(abs(loss), abs(self.predicted_loss)))
            z = (loss - self.predicted_loss) / max(self.sigma, sigma_floor)
            self._recent.append((step, z))
            anomalous = z > config.onset_z_threshold

            beyond_cooldown = self._completed_updates - self._last_trigger_update > config.cooldown_steps
            if z > config.z_threshold and beyond_cooldown:
                onset = step
                for recorded_step, recorded_z in reversed(self._recent):
                    if recorded_z <= config.onset_z_threshold:
                        break
                    onset = recorded_step
                self._last_trigger_update = self._completed_updates
                logger.warning(
                    "Loss spike at step %d: loss %.4f is %.1f sigma above the "
                    "predicted %.4f (sigma %.4g); onset step %d.",
                    step,
                    loss,
                    z,
                    self.predicted_loss,
                    self.sigma,
                    onset,
                )

        self._maybe_update_statistics(loss, anomalous=anomalous)
        return onset

    def _maybe_update_statistics(self, loss: float, *, anomalous: bool) -> None:
        """Fold one observation into the level, trend and residual variance.

        An observation already past the onset bar is held out of the update.
        Otherwise a sustained excursion drags the level and inflates the
        variance, and the band widens to cover the very rise it is supposed to
        catch. The hold is bounded by ``onset_lookback``: once the excursion is
        older than the window the onset search can reach back through, there is
        no spike left to report and the level is accepted as the new normal.
        """
        if anomalous and self._frozen_steps < self.config.onset_lookback:
            self._frozen_steps += 1
            return
        if not anomalous:
            self._frozen_steps = 0

        if self._count == 0:
            # Seed from the first observation. Starting the level at zero would
            # make every early step look like a spike.
            self._level = loss
            self._trend = 0.0
            self._var = 0.0
            self._count = 1
            return

        config = self.config
        residual = loss - self.predicted_loss
        previous_level = self._level
        self._level = (1.0 - config.level_decay) * loss + config.level_decay * (self._level + self._trend)
        self._trend = (1.0 - config.trend_decay) * (self._level - previous_level) + config.trend_decay * self._trend
        self._var = config.level_decay * (self._var + (1.0 - config.level_decay) * residual * residual)
        self._count += 1
