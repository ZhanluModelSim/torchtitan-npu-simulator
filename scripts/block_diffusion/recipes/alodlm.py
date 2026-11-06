# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ALoDLM-specific data, gate, objective and inference policies outside raw.

The equations are independently expressed in this repository; no author code is
copied. This recipe composes the generic LoopedModel begin/step interface. All
Transformer, position encoding, recurrent normalization and coda/readout execution
belongs to that primitive. The recipe owns two-stream corruption, teacher/token
feedback, gate sampling, practical loss, commitment and halt decisions.

Old recipe checkpoints named ``backbone.*`` require the explicit
load_legacy_recipe_state_dict conversion to ``loop.backbone.*``. The backbone
property is a non-registering compatibility accessor; current state dictionaries
contain one physical backbone, not two aliases of its modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torchtitan_npu.simulator.raw_model.block_diff.model import (
        BlockDiffusionModel,
        KVCache,
        LoopedModel,
        LoopSpec,
        ModelConfig,
    )
elif __package__ and "." in __package__:
    from ..raw_reference import BlockDiffusionModel, KVCache, LoopedModel, LoopSpec, ModelConfig
else:
    from raw_reference import BlockDiffusionModel, KVCache, LoopedModel, LoopSpec, ModelConfig

ALODLM_RECIPE_VERSION = 1
# The old public recipe version remains available here, not in the raw contract.
ALODLM_CONTRACT_VERSION = ALODLM_RECIPE_VERSION


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def load_legacy_recipe_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Explicitly convert old ALoDLM checkpoint keys; then load with strict=True.

    Tensor values, aliases and dtypes are retained. This accepts only the old
    backbone/exit_gate namespaces and never silently accepts a mixed new layout.
    Strict load checks tensor keys/shapes, not the recurrent execution policy.
    Callers must retain the original model/recipe configuration, including the
    loop interval and depth. Optimizer-state migration is not performed here.
    """
    converted = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("recipe state entries must be string keys and tensors")
        if name.startswith("backbone."):
            destination = "loop." + name
        elif name.startswith("exit_gate."):
            destination = name
        else:
            raise ValueError(f"unsupported legacy recipe state key: {name}")
        if destination in converted:
            raise ValueError(f"duplicate converted recipe state key: {destination}")
        converted[destination] = value
    return converted


@dataclass(frozen=True)
class ALoDLMSpec:
    """Dense recurrent paradigm, independently specified from arXiv:2610.04198v1.

    Layer interval [loop_start, loop_end) is shared across recurrent passes.
    ``feedback_policy`` selects inference feedback only: cumulative implements
    the paper-style accumulated commitments; newly_committed matches only the
    released inference feedback rule. Training always uses cumulative gold feedback.
    Neither profile implements the approximate depth-aware serving caches.
    """

    loop_start: int
    loop_end: int
    max_depth: int
    mask_token_id: int
    feedback_policy: str = "cumulative"
    normalize_carry: bool = True
    qk_norm: bool = True
    gate_init_mass: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _positive_int("loop_start", self.loop_start, allow_zero=True)
        _positive_int("loop_end", self.loop_end)
        _positive_int("max_depth", self.max_depth)
        _positive_int("mask_token_id", self.mask_token_id, allow_zero=True)
        if self.loop_start >= self.loop_end:
            raise ValueError("loop_start must precede loop_end")
        if self.feedback_policy not in ("cumulative", "newly_committed"):
            raise ValueError("feedback_policy must be cumulative or newly_committed")
        if type(self.normalize_carry) is not bool or type(self.qk_norm) is not bool:
            raise ValueError("normalize_carry and qk_norm must be bool")
        if self.gate_init_mass is not None:
            mass = tuple(self.gate_init_mass)
            if (
                len(mass) != self.max_depth
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in mass
                )
                or not math.isclose(sum(mass), 1.0, rel_tol=1e-7, abs_tol=1e-7)
            ):
                raise ValueError("gate_init_mass must contain max_depth positive probabilities summing to one")
            object.__setattr__(self, "gate_init_mass", mass)

    def validate_model(self, model: ModelConfig) -> None:
        if model.num_experts:
            raise ValueError("the ALoDLM reference currently supports dense backbones only")
        if self.loop_end > model.num_hidden_layers:
            raise ValueError("loop_end exceeds the number of physical layers")
        if self.mask_token_id >= model.vocab_size:
            raise ValueError("mask_token_id must be inside the model vocabulary")


@dataclass(frozen=True)
class ALoDLMBatch:
    """Independent sequences represented as clean + reordered-noisy streams.

    Token fields have shape [batch, 2 * source_length]. ``targets`` holds original
    token IDs, not shifted labels. ``ar_labels`` holds the caller's clean-stream
    AR targets and ignore_index on the noisy stream. ``attention_mask`` is shared
    physical-row visibility; ``positions`` retains original logical positions.
    Packed documents and random corruption are deliberately outside this helper.
    """

    input_ids: torch.Tensor
    targets: torch.Tensor
    positions: torch.Tensor
    masked: torch.Tensor
    mask_probability: torch.Tensor
    ar_labels: torch.Tensor
    attention_mask: torch.Tensor
    source_length: int
    block_size: int
    ignore_index: int = -100


def alodlm_attention_mask(source_length: int, block_size: int, *, device=None) -> torch.Tensor:
    """Exact two-stream visibility, using physical order inside each block.

    A row sees preceding clean blocks and the causal portion of its own block
    in its own stream. No noisy row can see its same-block clean target.
    """
    _positive_int("source_length", source_length)
    _positive_int("block_size", block_size)
    rows = torch.arange(2 * source_length, device=device)
    stream = rows >= source_length
    blocks = rows.remainder(source_length) // block_size
    same_local_block = (stream[:, None] == stream[None, :]) & (blocks[:, None] == blocks[None, :])
    ordered_rows = rows[:, None] >= rows[None, :]
    prior_clean = (blocks[None, :] < blocks[:, None]) & ~stream[None, :]
    return (same_local_block & ordered_rows) | prior_clean


def build_alodlm_batch(
    clean_ids: torch.Tensor,
    labels: torch.Tensor,
    masked: torch.Tensor,
    mask_probability: torch.Tensor | float,
    *,
    block_size: int,
    mask_token_id: int,
    ignore_index: int = -100,
) -> ALoDLMBatch:
    """Deterministically construct two streams from caller-supplied corruption.

    The stable within-block permutation places observed rows before masked rows.
    Probability tensors may be scalar or [B,S]; selected entries must lie in
    (0,1] and have a non-ignored label. Tensor assertions check real inputs;
    these remain caller value invariants for meta inputs. Input token IDs and labels
    are supplied explicitly: this function neither samples corruption nor shifts
    AR labels. Each batch row is an independent sequence; partial last blocks
    are allowed. All tensor transformations have static output shapes.
    """
    _positive_int("block_size", block_size)
    _positive_int("mask_token_id", mask_token_id, allow_zero=True)
    if type(ignore_index) is not int:
        raise ValueError("ignore_index must be an integer")
    if clean_ids.ndim != 2 or clean_ids.dtype != torch.int64 or min(clean_ids.shape) < 1:
        raise ValueError("clean_ids must be a nonempty int64 [batch, sequence] tensor")
    if labels.shape != clean_ids.shape or labels.dtype != torch.int64 or labels.device != clean_ids.device:
        raise ValueError("labels must be int64 with the clean input shape and device")
    if masked.shape != clean_ids.shape or masked.dtype != torch.bool or masked.device != clean_ids.device:
        raise ValueError("masked must be bool with the clean input shape and device")
    if isinstance(mask_probability, torch.Tensor):
        if not mask_probability.is_floating_point() or mask_probability.device != clean_ids.device:
            raise ValueError("mask_probability must be floating point on the input device")
        if mask_probability.ndim != 0 and mask_probability.shape != clean_ids.shape:
            raise ValueError("mask_probability must be scalar or match clean_ids")
        probability = mask_probability.to(torch.float32).expand(clean_ids.shape)
    else:
        if (
            isinstance(mask_probability, bool)
            or not isinstance(mask_probability, (int, float))
            or not (math.isfinite(mask_probability) and 0 < mask_probability <= 1)
        ):
            raise ValueError("scalar mask_probability must lie in (0,1]")
        probability = torch.full(clean_ids.shape, mask_probability, dtype=torch.float32, device=clean_ids.device)
    # These tensor guards are traceable: real inputs are checked without Python
    # scalar extraction, while meta execution carries only their shape contract.
    torch._assert_async((~masked | (labels != ignore_index)).all(), "masked rows require supervised labels")
    valid_probability = torch.isfinite(probability) & (probability > 0) & (probability <= 1)
    torch._assert_async((~masked | valid_probability).all(), "active mask_probability must be finite and in (0,1]")
    batch, length = clean_ids.shape
    chunks = []
    for start in range(0, length, block_size):
        stop = min(start + block_size, length)
        local_order = masked[:, start:stop].to(torch.int32).argsort(dim=1, stable=True)
        chunks.append(local_order + start)
    order = torch.cat(chunks, dim=1)
    original_positions = torch.arange(length, device=clean_ids.device).expand(batch, length)
    selected = masked.gather(1, order)
    noisy_ids = torch.where(masked, mask_token_id, clean_ids).gather(1, order)
    probability = torch.where(selected, probability.gather(1, order), 0.0)
    return ALoDLMBatch(
        input_ids=torch.cat((clean_ids, noisy_ids), dim=1),
        targets=torch.cat((clean_ids, clean_ids.gather(1, order)), dim=1),
        positions=torch.cat((original_positions, original_positions.gather(1, order)), dim=1),
        masked=torch.cat((torch.zeros_like(masked), selected), dim=1),
        mask_probability=torch.cat((torch.zeros_like(probability), probability), dim=1),
        ar_labels=torch.cat((labels, torch.full_like(labels, ignore_index)), dim=1),
        attention_mask=alodlm_attention_mask(length, block_size, device=clean_ids.device),
        source_length=length,
        block_size=block_size,
        ignore_index=ignore_index,
    )


@dataclass(frozen=True)
class ALoDLMOutput:
    """Full-depth training readouts; exit depths are one-based (observed rows: K)."""

    logits: torch.Tensor
    gate_logits: torch.Tensor
    exit_depths: torch.Tensor
    final_carry: torch.Tensor


@dataclass(frozen=True)
class ALoDLMOuterOutput:
    """One uncached inference replay; commit depths are not gate halt decisions."""

    logits: torch.Tensor
    gate_logits: torch.Tensor
    commit_depths: torch.Tensor
    committed: torch.Tensor
    tokens: torch.Tensor
    final_carry: torch.Tensor
    executed_steps: int


class ALoDLMExitGate(nn.Module):
    """Learnable conditional hazards initialized from positive exit-depth mass.

    The terminal projection and depth bias are still evaluated. Consumers force
    its *effective* hazard to one; they must not discard the executed gate op.
    """

    def __init__(self, hidden_size: int, spec: ALoDLMSpec, dtype: torch.dtype):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1, bias=True, dtype=dtype)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        mass = spec.gate_init_mass or (1.0 / spec.max_depth,) * spec.max_depth
        # P(exit at depth d | survived d-1) = mass[d] / sum(mass[d:]).
        biases = [math.log(mass[depth] / sum(mass[depth + 1 :])) for depth in range(spec.max_depth - 1)]
        self.depth_bias = nn.Parameter(torch.tensor([*biases, 0.0], dtype=dtype))

    def forward(self, features: torch.Tensor, depth: int) -> torch.Tensor:
        return self.projection(features.detach()).squeeze(-1) + self.depth_bias[depth]


def _alodlm_depth_plan(
    depths: torch.Tensor | tuple[tuple[int, ...], ...],
    input_ids: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    """Validate plan metadata; callers add traceable mask-specific value guards."""
    if isinstance(depths, torch.Tensor):
        if depths.dtype != torch.int64 or depths.shape != input_ids.shape or depths.device != input_ids.device:
            raise ValueError("depth plan must be int64 with the input shape and device")
        return depths
    rows = tuple(tuple(row) for row in depths)
    if len(rows) != input_ids.shape[0] or any(len(row) != input_ids.shape[1] for row in rows):
        raise ValueError("static depth plan must match [batch, sequence]")
    if any(type(depth) is not int or depth < 0 or depth > max_depth for row in rows for depth in row):
        raise ValueError("static depth plan values must be between zero and the executed depth")
    return torch.tensor(rows, dtype=torch.int64, device=input_ids.device)


class ALoDLMModel(nn.Module):
    """Concrete ALoDLM recipe composed over the generic recurrent primitive.

    This implements recurrence equations and explicit dense replay, not the
    authors' code or checkpoint format. It intentionally has no KV cache, no
    approximate depth fallback, no streaming driver and no implicit token-pruning.
    Heads run over every physical row at every executed depth. A new outer call
    always rebuilds the embedding/prelude state; recurrent carry is local to it.
    """

    def __init__(self, config: ModelConfig, spec: ALoDLMSpec):
        super().__init__()
        spec.validate_model(config)
        self.config = replace(config, qk_norm=spec.qk_norm)
        self.spec = spec
        self.loop = LoopedModel(
            self.config,
            LoopSpec(
                loop_start=spec.loop_start,
                loop_end=spec.loop_end,
                max_depth=spec.max_depth,
                normalize_carry=spec.normalize_carry,
            ),
        )
        self.exit_gate = ALoDLMExitGate(self.config.hidden_size, spec, self.config.dtype)

    @property
    def backbone(self) -> BlockDiffusionModel:
        """Compatibility access without registering the backbone twice."""
        return self.loop.backbone

    def parameter_memory_bytes(self) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def layer_evaluations(self, steps: int | None = None) -> int:
        """Count physical-layer calls only, not FLOPs/token or latency."""
        steps = self.spec.max_depth if steps is None else steps
        _positive_int("steps", steps)
        if steps > self.spec.max_depth:
            raise ValueError("steps cannot exceed max_depth")
        return self.loop.layer_evaluations(steps=steps, readout_count=steps)

    def logical_buffer_count(self) -> int:
        """Depth-aware KV *pair* count only; this reference allocates no caches."""
        return self.config.num_hidden_layers + (self.spec.max_depth - 1) * (self.spec.loop_end - self.spec.loop_start)

    def _validate_inputs(self, input_ids, position_ids, attention_mask, masked, cache) -> None:
        if cache is not None:
            raise ValueError("ALoDLM uncached reference does not support ordinary or depth-aware KV caches")
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
        if (
            position_ids.dtype != torch.int64
            or position_ids.shape != input_ids.shape
            or position_ids.device != input_ids.device
        ):
            raise ValueError("position_ids must be int64 with the input shape and device")
        torch._assert_async((position_ids >= 0).all(), "position_ids must be nonnegative")
        if masked.dtype != torch.bool or masked.shape != input_ids.shape or masked.device != input_ids.device:
            raise ValueError("masked must be bool with the input shape and device")
        length = input_ids.shape[1]
        if (
            attention_mask.dtype != torch.bool
            or tuple(attention_mask.shape) != (length, length)
            or attention_mask.device != input_ids.device
        ):
            raise ValueError("attention_mask must be explicit bool [sequence, sequence] on the input device")

    def forward(
        self,
        batch: ALoDLMBatch,
        *,
        exit_depths: torch.Tensor | tuple[tuple[int, ...], ...] | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOutput:
        return self.rollout_training(batch, exit_depths=exit_depths, cache=cache)

    def rollout_training(
        self,
        batch: ALoDLMBatch,
        *,
        exit_depths: torch.Tensor | tuple[tuple[int, ...], ...] | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOutput:
        """Full-K differentiable rollout with cumulative masked gold feedback.

        Replay exit depths at masked rows must be in [1,K]; observed entries are
        ignored and reported as K. Without replay, independent conditional gate
        draws determine exits, with terminal exit forced. All K readout/gate
        tensors are retained, including readouts after a token's commitment.
        Gate features are detached; carry and teacher embeddings are not.
        """
        self._validate_inputs(batch.input_ids, batch.positions, batch.attention_mask, batch.masked, cache)
        if (
            batch.targets.shape != batch.input_ids.shape
            or batch.targets.dtype != torch.int64
            or batch.targets.device != batch.input_ids.device
        ):
            raise ValueError("batch targets must be int64 with the input shape and device")
        plan = None if exit_depths is None else _alodlm_depth_plan(exit_depths, batch.input_ids, self.spec.max_depth)
        if plan is not None:
            in_range = (plan >= 1) & (plan <= self.spec.max_depth)
            torch._assert_async((~batch.masked | in_range).all(), "masked exit depths must be in [1,K]")
            observed_sentinel = (plan == 0) | (plan == self.spec.max_depth)
            torch._assert_async((batch.masked | observed_sentinel).all(), "observed exit depths must be zero or K")
        state = self.loop.begin(batch.input_ids, batch.positions, batch.attention_mask)
        teacher = self.backbone.embed_tokens(batch.targets)
        exits = torch.full_like(batch.input_ids, self.spec.max_depth)
        logits, gates = [], []
        for depth in range(self.spec.max_depth):
            feedback_mask = batch.masked & (exits <= depth) if depth else None
            step = self.loop.step(
                state,
                feedback=teacher if depth else None,
                feedback_mask=feedback_mask,
                readout=True,
                project_logits=True,
            )
            state = step.state
            assert step.features is not None and step.logits is not None
            prediction = step.logits
            gate = self.exit_gate(step.features, depth)
            logits.append(prediction)
            gates.append(gate)
            if depth + 1 < self.spec.max_depth:
                with torch.no_grad():
                    chosen = (
                        torch.rand_like(gate.float()) < gate.float().sigmoid() if plan is None else plan == depth + 1
                    )
                    chosen = chosen & batch.masked & (exits == self.spec.max_depth)
                exits = torch.where(chosen, depth + 1, exits)
        return ALoDLMOutput(torch.stack(logits), torch.stack(gates), exits, state.carry)

    @torch.no_grad()
    def run_outer(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked: torch.Tensor,
        *,
        commit_depths: torch.Tensor | tuple[tuple[int, ...], ...],
        steps: int | None = None,
        feedback_tokens: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ALoDLMOuterOutput:
        """Replay an uncached outer window with explicit commit and stop schedules.

        Zero denotes no commit, positive depths are one-based. ``steps`` controls
        executed depth independently of gate hazards and confidence commitments;
        it is not an implementation of adaptive halting. Supplied feedback tokens
        replace argmax predictions at their planned commit depth. Committed rows
        remain in all subsequent dense core/coda/head computations.
        """
        self._validate_inputs(input_ids, position_ids, attention_mask, masked, cache)
        steps = self.spec.max_depth if steps is None else steps
        self.layer_evaluations(steps)
        plan = _alodlm_depth_plan(commit_depths, input_ids, steps)
        torch._assert_async(((plan >= 0) & (plan <= steps)).all(), "commit depths must be in [0,steps]")
        torch._assert_async((masked | (plan == 0)).all(), "observed positions cannot have a commit depth")
        if feedback_tokens is not None and (
            feedback_tokens.dtype != torch.int64
            or feedback_tokens.shape != input_ids.shape
            or feedback_tokens.device != input_ids.device
        ):
            raise ValueError("feedback_tokens must be int64 with the input shape and device")
        state = self.loop.begin(input_ids, position_ids, attention_mask)
        tokens = input_ids
        committed = torch.zeros_like(masked)
        newly_committed = torch.zeros_like(masked)
        actual_depths = torch.zeros_like(input_ids)
        logits, gates = [], []
        for depth in range(steps):
            feedback_mask = None
            replacement = None
            if depth:
                feedback_mask = committed if self.spec.feedback_policy == "cumulative" else newly_committed
                replacement = self.backbone.embed_tokens(tokens)
            step = self.loop.step(
                state,
                feedback=replacement,
                feedback_mask=feedback_mask,
                readout=True,
                project_logits=True,
            )
            state = step.state
            assert step.features is not None and step.logits is not None
            prediction = step.logits
            gate = self.exit_gate(step.features, depth)
            logits.append(prediction)
            gates.append(gate)
            selected_tokens = prediction.argmax(-1) if feedback_tokens is None else feedback_tokens
            newly_committed = masked & ~committed & (plan == depth + 1)
            tokens = torch.where(newly_committed, selected_tokens, tokens)
            actual_depths = torch.where(newly_committed, depth + 1, actual_depths)
            committed = committed | newly_committed
        return ALoDLMOuterOutput(
            torch.stack(logits),
            torch.stack(gates),
            actual_depths,
            committed,
            tokens,
            state.carry,
            steps,
        )


@dataclass(frozen=True)
class ALoDLMObjective:
    """Released-practical surrogate components, not the unbiased basic NELBO.

    Probabilities and detached denoising weights have shape [depth,batch,tokens]
    and are zero outside the initially supervised masked positions.
    """

    loss: torch.Tensor
    denoising: torch.Tensor
    actor: torch.Tensor
    autoregressive: torch.Tensor
    masked_tokens: torch.Tensor
    exit_probabilities: torch.Tensor
    denoising_weights: torch.Tensor


def alodlm_training_objective(
    output: ALoDLMOutput,
    batch: ALoDLMBatch,
    *,
    depth_prior_c: float = 0.4,
    kl_individual: float = 0.1,
    kl_aggregate: float = 1.0,
    ignore_index: int | None = None,
) -> ALoDLMObjective:
    """Compute the dense, unpacked released-practical ALoDLM objective.

    The recurrence supplies all depth readouts. This function implements
    detached pre-exit supervision, sequence-level score-function credit with a
    first-pass baseline, and clean-stream AR supervision averaged over depths.
    The release divides their sum by two; this convention is explicit here.
    The original Eq. 7 unbiasedness claim does not apply to this relaxed loss.
    A caller replaying fixed exits gets a conditional trajectory loss, not a
    claim that those exits were sampled from the current gate distribution.
    """
    for name, value in (
        ("depth_prior_c", depth_prior_c),
        ("kl_individual", kl_individual),
        ("kl_aggregate", kl_aggregate),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(batch.ignore_index) is not int:
        raise ValueError("batch.ignore_index must be an integer")
    if ignore_index is not None and (type(ignore_index) is not int or ignore_index != batch.ignore_index):
        raise ValueError("objective ignore_index must agree with the batch")
    ignore_index = batch.ignore_index
    scores = output.logits
    gate = output.gate_logits.float()
    if scores.ndim != 4 or gate.shape != scores.shape[:-1]:
        raise ValueError("logits must be [depth,batch,tokens,vocab] with matching gate logits")
    depth_count, batch_count, tokens, vocab = scores.shape
    if min(depth_count, batch_count, tokens, vocab) < 1:
        raise ValueError("objective dimensions must be positive")
    expected = (batch_count, tokens)
    if (
        batch.targets.shape != expected
        or batch.targets.dtype != torch.int64
        or batch.masked.shape != expected
        or batch.masked.dtype != torch.bool
        or batch.mask_probability.shape != expected
        or not batch.mask_probability.is_floating_point()
        or output.exit_depths.shape != expected
        or output.exit_depths.dtype != torch.int64
        or batch.ar_labels.shape != expected
        or batch.ar_labels.dtype != torch.int64
    ):
        raise ValueError("batch targets, masks, probabilities, AR labels and exit depths must match [batch,tokens]")
    if tokens != 2 * batch.source_length:
        raise ValueError("practical loss requires an unpacked clean/noisy two-stream batch")
    for tensor in (gate, batch.targets, batch.masked, batch.mask_probability, output.exit_depths, batch.ar_labels):
        if tensor.device != scores.device:
            raise ValueError("all objective tensors must share a device")

    valid = batch.masked
    torch._assert_async(
        (
            ~valid
            | (torch.isfinite(batch.mask_probability) & (batch.mask_probability > 0) & (batch.mask_probability <= 1))
        ).all(),
        "supervised mask probabilities must be finite and lie in (0,1]",
    )
    torch._assert_async(
        (~valid | ((output.exit_depths >= 1) & (output.exit_depths <= depth_count))).all(),
        "supervised exit depths must lie in [1, max_depth]",
    )
    count = valid.sum()
    denominator = count.clamp_min(1).float()
    per_depth_ce = F.cross_entropy(
        scores.float().reshape(-1, vocab),
        batch.targets.unsqueeze(0).expand(depth_count, -1, -1).reshape(-1),
        reduction="none",
    ).reshape(depth_count, batch_count, tokens)
    depth_ids = torch.arange(1, depth_count + 1, device=scores.device).view(-1, 1, 1)
    log_survival = torch.cat((torch.zeros_like(gate[:1]), F.logsigmoid(-gate).cumsum(0)[:-1]), dim=0)
    # Force the terminal depth to absorb all remaining survival mass.
    log_exit_profile = torch.where(depth_ids == depth_count, log_survival, log_survival + F.logsigmoid(gate))
    exits = output.exit_depths
    # Observed positions have no exit event; safe gather indices are discarded
    # by the mask. A rollout must validate supervised exits in [1, max_depth].
    indices = torch.where(valid, exits - 1, 0).unsqueeze(0)
    chosen_log_probability = log_exit_profile.gather(0, indices).squeeze(0)
    exit_cost = per_depth_ce.gather(0, indices).squeeze(0)

    log_supervision = torch.where(depth_ids == exits.unsqueeze(0), log_survival, log_survival + F.logsigmoid(gate))
    before_feedback = (depth_ids <= exits.unsqueeze(0)) & valid.unsqueeze(0)
    supervision = torch.where(before_feedback, log_supervision.exp(), 0.0).detach()
    inverse_probability = torch.where(valid, (batch.mask_probability.float() + 1e-8).reciprocal(), 0.0)
    normalized_inverse = inverse_probability / inverse_probability.sum().clamp_min(1e-12)
    denoising = ((per_depth_ce * supervision).sum(0) * normalized_inverse).sum()

    probabilities = log_exit_profile.exp() * valid.unsqueeze(0)
    with torch.no_grad():
        aggregate = probabilities.detach().sum((1, 2)) / denominator
        aggregate_log = aggregate.clamp_min(1e-12).log()
        prior_log = F.log_softmax(
            -depth_prior_c * torch.arange(1, depth_count + 1, device=scores.device, dtype=torch.float32), dim=0
        )
        selected_aggregate = aggregate_log[indices.squeeze(0)]
        selected_prior = prior_log[indices.squeeze(0)]
        regularization = kl_individual * (chosen_log_probability - selected_aggregate) + kl_aggregate * (
            selected_aggregate - selected_prior
        )
        advantage = (exit_cost - per_depth_ce[0] + regularization) * valid
    # Separate batch rows are independently attended sequences. Packed segment
    # credit is deliberately not inferred from a flat, ambiguous token stream.
    sequence_score = (chosen_log_probability * valid).sum(dim=1)
    actor = (advantage.sum(dim=1) * sequence_score).sum() / denominator

    clean_length = batch.source_length
    if clean_length > 1:
        next_labels = batch.ar_labels[:, 1:clean_length]
        ar_count = (next_labels != ignore_index).sum().clamp_min(1)
        ar_terms = F.cross_entropy(
            scores[:, :, : clean_length - 1].float().reshape(-1, vocab),
            next_labels.unsqueeze(0).expand(depth_count, -1, -1).reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        )
        autoregressive = ar_terms.sum() / (depth_count * ar_count)
    else:
        autoregressive = scores.sum() * 0.0
    loss = (denoising + actor + autoregressive) / 2
    # Keep zero-valued terminal gate/empty-target gradients explicit.
    loss = loss + gate.sum() * 0.0
    return ALoDLMObjective(loss, denoising, actor, autoregressive, count, probabilities, supervision)


@dataclass(frozen=True)
class ALoDLMDecision:
    tokens: torch.Tensor
    newly_committed: torch.Tensor
    committed: torch.Tensor
    unresolved: torch.Tensor
    cumulative_halt: torch.Tensor
    stop: torch.Tensor
    entropy: torch.Tensor


@torch.no_grad()
def alodlm_decision(
    logits: torch.Tensor,
    gate_logits: torch.Tensor,
    unresolved: torch.Tensor,
    committed: torch.Tensor,
    cumulative_halt: torch.Tensor,
    *,
    entropy_threshold: float,
    halt_threshold: float,
    terminal: bool = False,
    strict_entropy: bool = False,
    position_penalty: torch.Tensor | None = None,
) -> ALoDLMDecision:
    """Tensor-only one-pass commitment/halting decision with greedy predictions.

    Confidence commits tokens; the gate stops recurrence. They are separate
    decisions. Defaults use the paper's inclusive entropy threshold. The release
    profile can request a strict threshold and supply its explicit position cost.
    The caller drives outer/inner loops; meta callers can replay their schedule.
    """
    for name, value in (("entropy_threshold", entropy_threshold), ("halt_threshold", halt_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if entropy_threshold < 0 or not 0 <= halt_threshold <= 1:
        raise ValueError("entropy_threshold must be nonnegative and halt_threshold in [0,1]")
    if type(terminal) is not bool or type(strict_entropy) is not bool:
        raise ValueError("terminal and strict_entropy must be bool")
    if logits.ndim != 3 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError("logits must be a nonempty floating [batch,tokens,vocab] tensor")
    shape = logits.shape[:2]
    for tensor in (gate_logits, unresolved, committed, cumulative_halt):
        if tensor.shape != shape or tensor.device != logits.device:
            raise ValueError("decision tensors must match [batch,tokens] and device")
    if unresolved.dtype != torch.bool or committed.dtype != torch.bool:
        raise ValueError("unresolved and committed must be boolean")
    if not gate_logits.is_floating_point() or not cumulative_halt.is_floating_point():
        raise ValueError("gate logits and cumulative halt must be floating")
    if position_penalty is not None and (
        position_penalty.shape != shape
        or position_penalty.device != logits.device
        or not position_penalty.is_floating_point()
    ):
        raise ValueError("position_penalty must be floating [batch,tokens] on the input device")
    torch._assert_async((~(unresolved & committed)).all(), "unresolved and committed masks must be disjoint")
    torch._assert_async(
        (torch.isfinite(cumulative_halt) & (cumulative_halt >= 0) & (cumulative_halt <= 1)).all(),
        "cumulative halt probabilities must be finite and lie in [0,1]",
    )
    if position_penalty is not None:
        torch._assert_async(
            (torch.isfinite(position_penalty) & (position_penalty >= 0)).all(),
            "position penalties must be finite and nonnegative",
        )
    working_logits = logits.float()
    torch._assert_async(
        (~torch.isnan(working_logits) & (working_logits != float("inf"))).all(),
        "logits may mask vocabulary entries with -inf, but cannot contain NaN or +inf",
    )
    torch._assert_async(
        torch.isfinite(working_logits).any(-1).all(),
        "each logits row must contain at least one finite vocabulary value",
    )
    log_probabilities = F.log_softmax(working_logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -torch.xlogy(probabilities, probabilities).sum(-1)
    ranking = entropy if position_penalty is None else entropy + position_penalty
    hazard = torch.ones_like(gate_logits, dtype=torch.float32) if terminal else gate_logits.float().sigmoid()
    cumulative = torch.where(unresolved, cumulative_halt + (1 - cumulative_halt) * hazard, cumulative_halt)
    confident = ranking < entropy_threshold if strict_entropy else ranking <= entropy_threshold
    new = unresolved & confident
    all_committed = committed | new
    residual = unresolved & ~new
    remaining = residual.sum(-1)
    mean_halt = (cumulative * residual).sum(-1) / remaining.clamp_min(1)
    stop = (remaining == 0) | (mean_halt >= halt_threshold) | terminal
    ensure_progress = stop & ~all_committed.any(-1) & unresolved.any(-1)
    lowest = ranking.masked_fill(~unresolved, float("inf")).argmin(-1)
    fallback = F.one_hot(lowest, num_classes=logits.shape[1]).bool() & ensure_progress.unsqueeze(-1)
    new = new | fallback
    all_committed = committed | new
    residual = unresolved & ~new
    return ALoDLMDecision(logits.argmax(-1), new, all_committed, residual, cumulative, stop, entropy)
