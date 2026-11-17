# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The anticipatory training schedule: when to prefetch, replay, and roll back.

Everything that makes anticipatory routing different from ordinary training
lives here, so the trainer only has to run each optimizer step inside
:meth:`AnticipatorySchedule.training_step_context`.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.observability import structured_logger as sl

from .cache import RoutingSlot, slot_nbytes
from .engine import AnticipatoryTrainingEngine
from .router import build_routing_cache

if TYPE_CHECKING:
    from collections.abc import Iterator


TrainingMicrobatch = Any

logger = logging.getLogger(__name__)


class Phase(enum.Enum):
    """Which schedule the current optimizer step follows.

    Derived from the step budget and the queue rather than stored, so the two
    cannot disagree with a separate phase field.

    Warmup is deliberately absent: it runs inline inside the spike handler
    rather than as outer-loop iterations, so one iteration of ``Trainer.train``
    stays one optimizer step in steady state. On the rollback iteration the
    logical step moves backwards before the upstream validation/profiling tail;
    only checkpoint saving is suppressed, and trace labels still name the
    triggering step until the next outer iteration.
    """

    NORMAL = "normal"
    ACTIVE = "active"
    DRAIN = "drain"


@dataclass
class StepData:
    """One optimizer step's input: its microbatches and how they were paid for.

    ``load_times`` travels with the batch so the data-loading cost is credited
    to the step that trains on it, not the step that fetched it. ``slots`` holds
    the routing indices captured for these microbatches, or ``None`` when the
    step routes live, and ``captured_at_step`` is the number of optimizer steps
    completed when they were captured -- the other half of the staleness check.
    """

    microbatches: list[TrainingMicrobatch]
    load_times: list[float]
    slots: list[RoutingSlot] | None = None
    captured_at_step: int | None = None


class SuppliedMicrobatches:
    """Hand a pre-fetched batch to the base ``Trainer.train_step``.

    ``Trainer.train_step`` takes its data iterator as an argument and pulls
    exactly ``gradient_accumulation_steps * num_pp_microbatches`` microbatches
    from it, so a queued batch can be supplied without touching the base
    implementation. This iterator holds exactly that many and fails loudly in
    either direction, so a future change to the base fetch count cannot quietly
    train on a short batch. A bare ``iter(list)`` would not do: it raises
    ``StopIteration`` inside the base step, and ``Trainer.train`` catches only
    ``DataloaderExhaustedError``, so the failure would surface as noise.
    """

    def __init__(self, microbatches: list[TrainingMicrobatch]) -> None:
        self._microbatches = microbatches
        self._index = 0

    def __iter__(self) -> SuppliedMicrobatches:
        return self

    def __next__(self) -> TrainingMicrobatch:
        if self._index >= len(self._microbatches):
            raise RuntimeError(
                "Trainer.train_step requested more microbatches than the "
                f"{len(self._microbatches)} anticipatory routing pre-fetched for "
                "this optimizer step."
            )
        microbatch = self._microbatches[self._index]
        self._index += 1
        return microbatch

    def verify_consumed(self) -> None:
        if self._index != len(self._microbatches):
            raise RuntimeError(
                "Trainer.train_step consumed "
                f"{self._index} of {len(self._microbatches)} supplied "
                "microbatches. Anticipatory routing pre-fetches exactly one "
                "optimizer step's worth; the rest would be trained on with no "
                "cached routing indices."
            )


class AnticipatorySchedule:
    """Decides what each optimizer step trains on and how it routes.

    The schedule owns one queue of pre-fetched steps. It fills the queue by
    running forward-only passes that cache routing indices, and drains it by
    handing those batches to the trainer ``delay_steps`` later, so a step trains
    with the routing its data was assigned that many optimizer steps ago.

    It runs through three phases; see :class:`Phase`. ``NORMAL`` is ordinary
    training, entered and left automatically:

    - a loss spike rolls the run back and arms the mode (warmup, inline),
    - ``active_steps`` steps then train on stale routing,
    - the queue drains with cached routing, and the run is back to ``NORMAL``.
    """

    def __init__(self, trainer) -> None:
        config = trainer.config.anticipatory
        engine = AnticipatoryTrainingEngine(trainer)
        checkpoint_interval = trainer.config.checkpoint.interval
        keep_latest_k = trainer.config.checkpoint.keep_latest_k
        self.config = config
        self.engine = engine
        self.cache = build_routing_cache(
            engine.trainer.model_parts,
            index_store_dtype=config.index_store_dtype,
            device=trainer.device,
        )
        self.metrics = trainer.metrics_processor
        self.detector = config.detector.build()
        self._microbatches_per_step = trainer.gradient_accumulation_steps

        self._queue: deque[StepData] = deque()
        self._active_steps_left = 0
        self._num_rollbacks = 0
        self._phase_at_step_start = Phase.NORMAL
        self._logged_steady_lag = False

        if keep_latest_k > 0:
            logger.warning(
                "checkpointer.keep_latest_k=%d and interval=%d retain about "
                "%d steps of rollback history. A spike whose onset predates "
                "that has no checkpoint to roll back to and will be reported "
                "without acting on it.",
                keep_latest_k,
                checkpoint_interval,
                keep_latest_k * checkpoint_interval,
            )
        logger.info(
            "Anticipatory routing: delay %d step(s), %d active step(s), armed on a loss spike (max %d rollbacks).",
            config.delay_steps,
            config.active_steps,
            config.max_rollbacks,
        )

    # -- surface the trainer calls ----------------------------------------

    @property
    def data(self):
        """Return the training iterator whose cursor is reset on rollback."""
        return self.engine.data

    def prepare_microbatch(self) -> None:
        """Select routing for the upcoming forward and backward pass."""
        self.cache.select(self.engine._microbatch_index)

    def accumulate_microbatch_loss(self, loss) -> None:
        """Collect loss after a successful microbatch and advance its index."""
        self.engine.accumulate_microbatch_loss(loss)

    @property
    def phase(self) -> Phase:
        """Which schedule the next step follows, derived from budget and queue."""
        if self._active_steps_left > 0:
            return Phase.ACTIVE
        return Phase.DRAIN if self._queue else Phase.NORMAL

    @contextlib.contextmanager
    def training_step_context(self, data_iterator: Iterator[TrainingMicrobatch]) -> Iterator[SuppliedMicrobatches]:
        """Scope one optimizer step, yielding the microbatches it trains on.

        The caller runs the unmodified ``Trainer.train_step`` on what this
        yields. Entering chooses the data and the routing mode; leaving spends
        the step and reacts to the loss it produced. A step that raises skips
        the phase advance but still restores the routing mode, so a failed step
        cannot leave the cache armed for the next forward.
        """
        engine = self.engine
        entry_step = engine.trainer.step
        completed = False
        engine.begin_step()
        engine.in_step = True
        try:
            step_data = self._prepare_training_data_and_routes(data_iterator)
            engine.initialize_routing(self.cache, step_data.microbatches[0])
            self._credit_metrics(step_data)
            supplied = SuppliedMicrobatches(step_data.microbatches)
            try:
                self.cache.begin_training(step_data.slots)
                yield supplied
            finally:
                self.cache.reset()
            supplied.verify_consumed()
            engine.in_step = False
            self.detector.record_optimizer_update()
            engine.reduce_step_loss()
            self._advance_schedule_and_handle_spike(data_iterator)
            completed = True
        except DataloaderExhaustedError:
            # 0.3.0 treats exhaustion as normal termination after incrementing
            # the step counter; the incomplete step must not be counted.
            if engine.trainer.step == entry_step:
                engine.trainer.step -= 1
            raise
        finally:
            engine.in_step = False
            if not completed:
                engine.failed = True
                self._queue.clear()

    # -- step lifecycle ----------------------------------------------------

    def _prepare_training_data_and_routes(self, data_iterator: Iterator[TrainingMicrobatch]) -> StepData:
        """Prefetch if active, then choose this step's data and cached routing."""

        phase = self._phase_at_step_start = self.phase
        if phase is Phase.ACTIVE and len(self._queue) >= self.engine.remaining_steps:
            self._active_steps_left = 0
            phase = self._phase_at_step_start = self.phase
        # Capture at the parameters before this update.
        if phase is Phase.ACTIVE and not self._prefetch_and_capture(data_iterator):
            phase = self._phase_at_step_start = self.phase

        if not self._queue:
            return self._fetch_step_data(data_iterator)
        step_data = self._queue.popleft()
        # ACTIVE and DRAIN both consume the recorded data/route pair. DRAIN
        # stops prefetching but keeps replaying until the queue is empty.

        self._check_staleness(step_data)
        return step_data

    def _check_staleness(self, step_data: StepData) -> None:
        """Verify how old this step's routing indices actually are.

        This is the property the whole feature exists to produce, and nothing
        else states it: it emerges from the queue arithmetic, where warmup
        pushes ``delay_steps`` entries and every step afterwards pushes one and
        pops one. Checking it here means a broken push/pop order fails loudly
        instead of quietly degrading to fresh routing -- which trains perfectly
        well and would pass every loss-based test.

        The lag ramps 0..``delay_steps`` while the warmup backlog drains, then
        holds there through ACTIVE and DRAIN once steady state is reached.
        """
        assert step_data.captured_at_step is not None
        lag = self.engine.num_completed_steps - step_data.captured_at_step
        assert 0 <= lag <= self.config.delay_steps, (
            f"Routing indices are {lag} optimizer steps old, outside the "
            f"0..{self.config.delay_steps} the schedule can produce. The "
            "prefetch queue is out of step with the training loop."
        )
        if lag == self.config.delay_steps and not self._logged_steady_lag:
            self._logged_steady_lag = True
            logger.info(
                "Anticipatory routing at steady state: training step %d is "
                "routing on indices computed %d optimizer step(s) ago.",
                self.engine.num_completed_steps + 1,
                lag,
            )

    def _advance_schedule_and_handle_spike(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        """Advance recovery, handle a detected spike, and update checkpoint eligibility."""
        self._advance_phase()
        self._maybe_handle_spike(data_iterator)
        # A checkpoint is only consistent once the queue has drained: until then
        # the prefetch has run the data stream ahead of the parameters, and
        # resuming from such a checkpoint would skip the queued batches.
        self.engine.suppress_checkpoint_saves = self.phase is not Phase.NORMAL

    def _credit_metrics(self, step_data: StepData) -> None:
        """Credit this step's tokens and data-loading cost at training time."""
        self.metrics.ntokens_since_last_log += sum(labels.numel() for _, labels in step_data.microbatches)
        self.metrics.data_loading_times.extend(step_data.load_times)

    # -- data --------------------------------------------------------------

    def _fetch_step_data(self, data_iterator: Iterator[TrainingMicrobatch]) -> StepData:
        """Advance the sole resumable 0.3.0 stream by one complete step."""
        if data_iterator is not self.engine.data:
            raise ValueError("Use trainer.batch_generator(trainer.dataloader)")
        data = data_iterator.fetch_step(self._microbatches_per_step, self.engine.device)
        return StepData(microbatches=data.batches, load_times=data.loading_times)

    # -- prefetch ----------------------------------------------------------

    def _prefetch_and_capture(self, data_iterator: Iterator[TrainingMicrobatch]) -> bool:
        """Fetch one step ahead and cache its routing indices.

        Returns ``False`` when the data ran out, which ends the active window
        and leaves the queue to drain.
        """
        try:
            step_data = self._fetch_step_data(data_iterator)
        except DataloaderExhaustedError:
            logger.warning(
                "Anticipatory prefetch ran out of data; draining %d queued step(s) before stopping.",
                len(self._queue),
            )
            self._active_steps_left = 0
            return False

        slots: list[RoutingSlot] = []
        with sl.log_trace_span("anticipatory_prefetch"):
            for microbatch in step_data.microbatches:
                self.engine.initialize_routing(self.cache, microbatch)
                slot: RoutingSlot = {}
                try:
                    self.cache.begin_capture(slot)
                    self.engine.forward_only_microbatch(microbatch)
                    self.cache.save_captured_routes()
                finally:
                    self.cache.reset()
                slots.append(slot)

        step_data.slots = slots
        # Parameters advance only at the optimizer step, so every slot captured
        # in this call belongs to the same theta.
        step_data.captured_at_step = self.engine.num_completed_steps
        self._queue.append(step_data)
        return True

    # -- phase transitions --------------------------------------------------

    def _advance_phase(self) -> None:
        """Spend one step of the active budget and log any phase change."""
        phase_before = self._phase_at_step_start
        if phase_before is Phase.ACTIVE:
            self._active_steps_left = max(self._active_steps_left - 1, 0)

        if self.engine.remaining_steps == 0:
            self._active_steps_left = 0

        phase = self.phase
        if phase is phase_before:
            return
        if phase is Phase.DRAIN:
            logger.info(
                "Anticipatory prefetch finished; replaying %d queued step(s) while draining.",
                len(self._queue),
            )
        elif phase is Phase.NORMAL:
            logger.info("Anticipatory routing reverted to standard training; checkpointing resumes.")
            # The run continues on a different trajectory, so the loss history
            # collected before the rollback no longer describes it.
            self.detector.reset()

    def _enter_anticipatory(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        """Warm the index cache, then switch to anticipatory train steps.

        The warmup runs inline rather than as outer-loop iterations so that one
        iteration of ``Trainer.train`` stays exactly one optimizer step. No
        optimizer step happens here, so every index the warmup caches is
        computed at the same parameters.

        With ``delay_steps=0`` the warmup is empty and each active step captures
        and consumes the same batch, which reproduces standard training exactly.
        """
        config = self.config
        self._logged_steady_lag = False
        self._active_steps_left = config.active_steps
        with sl.log_trace_span("anticipatory_warmup"):
            for _ in range(min(config.delay_steps, self.engine.remaining_steps)):
                if not self._prefetch_and_capture(data_iterator):
                    break
        per_step_mib = sum(slot_nbytes(slot) for slot in (self._queue[0].slots or ())) / 1024**2 if self._queue else 0.0
        logger.info(
            "Anticipatory routing armed: %d step(s) of indices cached "
            "(%.1f MiB each), replaying routing from %d step(s) back. "
            "Checkpointing is paused until the queue drains, because the data "
            "stream now runs ahead of the training step.",
            len(self._queue),
            per_step_mib,
            config.delay_steps,
        )

    # -- spike response ------------------------------------------------------

    def _maybe_handle_spike(self, data_iterator: Iterator[TrainingMicrobatch]) -> None:
        """Feed the detector and, on a spike, roll back and arm the mode.

        Only observed during standard training: while the mode is active the run
        is deliberately on a different trajectory, and retriggering there would
        stack rollbacks on top of each other.
        """
        if self.phase is not Phase.NORMAL:
            return
        loss = self.engine.last_global_loss
        if loss is None:
            return
        step = self.engine.num_completed_steps
        onset = self.detector.observe(step, loss)
        if onset is None:
            return

        config = self.config
        if self._num_rollbacks >= config.max_rollbacks:
            logger.warning(
                "Loss spike at step %d (onset %d) but the rollback budget of %d is spent; continuing without rollback.",
                step,
                onset,
                config.max_rollbacks,
            )
            return

        target = self.engine.find_rollback_target(onset)
        if target is None:
            logger.warning(
                "Loss spike at step %d (onset %d) but no resumable checkpoint "
                "before step %d survives retention; continuing without "
                "rollback.",
                step,
                onset,
                onset,
            )
            return

        logger.warning(
            "Loss spike at step %d (onset %d, loss %.4f): rolling back to step %d and arming anticipatory routing.",
            step,
            onset,
            loss,
            target,
        )
        self.engine.rollback_to(target)
        self._num_rollbacks += 1
        self.detector.reset()
        self._restart_metrics_window()
        self._enter_anticipatory(data_iterator)

    def _restart_metrics_window(self) -> None:
        """Begin a fresh logging window at the rolled-back step.

        Mirrors the reset block at the end of ``MetricsProcessor.log``. Without
        it the next log record spans the rollback and reports a throughput and
        token delta for steps that were undone.
        """
        self.metrics.step_last_log = None
        self.metrics.ntokens_since_last_log = 0
        self.metrics.data_loading_times.clear()
        self.metrics.time_last_log = time.perf_counter()
