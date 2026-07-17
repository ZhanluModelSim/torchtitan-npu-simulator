#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate TorchTitan parallel and batch settings before launching workers.

This module deliberately imports only the Python standard library. A launcher
can therefore call it before importing TorchTitan, initializing distributed, or
allocating accelerator workers. All public validation functions are pure: they
only inspect their arguments and either return normalized values or raise
``ValueError``/``TypeError``.

``parse_parallel_training_args`` can selectively parse the original argument
list after an outer launcher has already parsed it. ``argparse`` does not consume
or mutate ``sys.argv``. Pass the original argument list, not the unknown/remainder
list returned by the outer parser. Unknown launcher arguments are ignored.

Command-line re-parsing can only observe values that are explicitly present in
the argument list. It cannot expand a TorchTitan ``--config`` registry function.
If parallel values come from such a config, the outer launcher must pass its
already-resolved values directly to ``validate_parallel_training_config``.

Validation order and formulas
=============================

1. Scalar domain checks
   ``world_size``, PP, EP, TP, CP, ETP, DP-replicate, local batch size, and PP
   microbatch size must be positive integers. ``dp_shard`` is either ``-1`` or
   a positive integer. ``global_batch_size`` is either ``-1`` or a positive
   integer. ``edp`` is optional; when supplied, it must be positive.

2. Resolve DP-shard
   TorchTitan's dense world mesh uses::

       world_size = pp * dp_replicate * dp_shard * cp * tp

   For ``dp_shard == -1`` this utility first requires ``world_size`` to be
   divisible by ``pp * dp_replicate * cp * tp`` and then assigns all remaining
   ranks to DP-shard. An explicit DP-shard must satisfy the same world formula.

3. Validate the expert mesh and derive EDP
   ETP must be either 1 or TP, matching TorchTitan. EP reinterprets ranks in the
   dense FSDP/TP region; it is not another factor in the dense world formula::

       efsdp = dp_shard * cp * tp / (ep * etp)
       edp   = dp_replicate * efsdp
             = world_size / (pp * ep * etp)

   ``ep * etp`` must exactly divide ``dp_shard * cp * tp``. ``edp`` is normally
   derived; if an upstream scheduler supplies it, the supplied value must equal
   the derived value. Here EDP means the size of TorchTitan's expert data mesh,
   including the DP-replicate dimension.

4. Validate training batches
   CP, TP, PP, and EP do not multiply the number of independent samples. The
   data-loading degree and gradient accumulation count are::

       data_parallel_degree = dp_replicate * dp_shard
       global_batch_size = (local_batch_size * data_parallel_degree
                            * gradient_accumulation_steps)

   ``global_batch_size == -1`` means one gradient accumulation step and resolves
   to ``local_batch_size * data_parallel_degree``. An explicit global batch size
   must be divisible by that value.

5. Validate pipeline microbatches
   PP microbatch size must be positive. When PP is enabled, local batch size must
   be exactly divisible by it::

       pipeline_microbatches = local_batch_size / pp_microbatch_size

   Having fewer microbatches than pipeline stages is inefficient but valid in
   TorchTitan, so this utility does not reject it. PP microbatches are distinct
   from gradient accumulation steps.

Scope
=====

These checks cover every relationship determined solely by the arguments above.
Model-dependent constraints still require separate inputs and checks: examples
include sequence length versus TP/CP, model layers versus PP split points,
expert count versus EP, attention heads/hidden dimensions versus TP, and
schedule-specific virtual pipeline stages.

Example preflight command::

    python3 scripts/validate_parallel_config.py \
        --world-size 256 --pp 2 --ep 32 --tp 2 --cp 1 \
        --dp-replicate 1 --dp-shard -1 --edp 4 \
        --pp-microbatch-size 1 --local-batch-size 1 \
        --global-batch-size -1 \
    && torchrun ...
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import NamedTuple


_TORCHTITAN_DEFAULTS = {
    "pp": 1,
    "ep": 1,
    "tp": 1,
    "cp": 1,
    "etp": 1,
    "dp_replicate": 1,
    "dp_shard": -1,
    "edp": None,
    "pp_microbatch_size": 1,
    "local_batch_size": 8,
    "global_batch_size": -1,
}


class ValidatedParallelConfig(NamedTuple):
    """Normalized values derived by a successful preflight validation."""

    world_size: int
    pp: int
    ep: int
    tp: int
    cp: int
    etp: int
    dp_replicate: int
    dp_shard: int
    efsdp: int
    edp: int
    data_parallel_degree: int
    pp_microbatch_size: int
    pipeline_microbatches: int
    local_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int


def _add_selective_parallel_arguments(parser: argparse.ArgumentParser) -> None:
    """Register TorchTitan dotted names and launcher-friendly aliases."""
    parser.add_argument("--world-size", "--world_size", "--ngpu", dest="world_size", type=int)
    parser.add_argument(
        "--parallelism.pipeline_parallel_degree",
        "--parallelism.pipeline-parallel-degree",
        "--pp",
        dest="pp",
        type=int,
        default=_TORCHTITAN_DEFAULTS["pp"],
    )
    parser.add_argument(
        "--parallelism.expert_parallel_degree",
        "--parallelism.expert-parallel-degree",
        "--ep",
        dest="ep",
        type=int,
        default=_TORCHTITAN_DEFAULTS["ep"],
    )
    parser.add_argument(
        "--parallelism.tensor_parallel_degree",
        "--parallelism.tensor-parallel-degree",
        "--tp",
        dest="tp",
        type=int,
        default=_TORCHTITAN_DEFAULTS["tp"],
    )
    parser.add_argument(
        "--parallelism.context_parallel_degree",
        "--parallelism.context-parallel-degree",
        "--cp",
        dest="cp",
        type=int,
        default=_TORCHTITAN_DEFAULTS["cp"],
    )
    parser.add_argument(
        "--parallelism.expert_tensor_parallel_degree",
        "--parallelism.expert-tensor-parallel-degree",
        "--etp",
        dest="etp",
        type=int,
        default=_TORCHTITAN_DEFAULTS["etp"],
    )
    parser.add_argument(
        "--parallelism.data_parallel_replicate_degree",
        "--parallelism.data-parallel-replicate-degree",
        "--dp-replicate",
        dest="dp_replicate",
        type=int,
        default=_TORCHTITAN_DEFAULTS["dp_replicate"],
    )
    parser.add_argument(
        "--parallelism.data_parallel_shard_degree",
        "--parallelism.data-parallel-shard-degree",
        "--dp-shard",
        dest="dp_shard",
        type=int,
        default=_TORCHTITAN_DEFAULTS["dp_shard"],
    )
    parser.add_argument(
        "--parallelism.expert_data_parallel_degree",
        "--parallelism.expert-data-parallel-degree",
        "--edp",
        dest="edp",
        type=int,
        default=_TORCHTITAN_DEFAULTS["edp"],
    )
    parser.add_argument(
        "--parallelism.pipeline_parallel_microbatch_size",
        "--parallelism.pipeline-parallel-microbatch-size",
        "--pp-microbatch-size",
        dest="pp_microbatch_size",
        type=int,
        default=_TORCHTITAN_DEFAULTS["pp_microbatch_size"],
    )
    parser.add_argument(
        "--training.local_batch_size",
        "--training.local-batch-size",
        "--local-batch-size",
        dest="local_batch_size",
        type=int,
        default=_TORCHTITAN_DEFAULTS["local_batch_size"],
    )
    parser.add_argument(
        "--training.global_batch_size",
        "--training.global-batch-size",
        "--global-batch-size",
        dest="global_batch_size",
        type=int,
        default=_TORCHTITAN_DEFAULTS["global_batch_size"],
    )


def parse_parallel_training_args(
    argv: Sequence[str] | None = None,
    *,
    world_size: int | None = None,
) -> dict[str, int | None]:
    """Extract parallel settings from an original argv, ignoring other flags.

    ``world_size`` is commonly known by the outer resource launcher rather than
    TorchTitan. It may be passed explicitly or supplied through ``--world-size``,
    ``--world_size``, or ``--ngpu``. Conflicting values are rejected.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_selective_parallel_arguments(parser)
    parsed, _unknown = parser.parse_known_args(argv)

    parsed_world_size = parsed.world_size
    if world_size is None and parsed_world_size is None:
        raise ValueError(
            "world_size is required; pass it to parse_parallel_training_args "
            "or provide --world-size"
        )
    if (
        world_size is not None
        and parsed_world_size is not None
        and world_size != parsed_world_size
    ):
        raise ValueError(
            "conflicting world_size values: "
            f"launcher supplied {world_size}, command line supplied {parsed_world_size}"
        )

    values = vars(parsed)
    values["world_size"] = (
        world_size if world_size is not None else parsed_world_size
    )
    return values


def parse_and_validate_parallel_training_args(
    argv: Sequence[str] | None = None,
    *,
    world_size: int | None = None,
) -> ValidatedParallelConfig:
    """Extract dotted TorchTitan arguments and run the complete preflight."""
    values = parse_parallel_training_args(argv, world_size=world_size)
    return validate_parallel_training_config(**values)  # type: ignore[arg-type]


def require_integer(name: str, value: object) -> int:
    """Return ``value`` as an int, rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def require_positive_integer(name: str, value: object) -> int:
    """Return a positive integer or raise a parameter-specific error."""
    integer = require_integer(name, value)
    if integer < 1:
        raise ValueError(f"{name} must be >= 1, got {integer}")
    return integer


def resolve_dp_shard(
    *,
    world_size: int,
    pp: int,
    tp: int,
    cp: int,
    dp_replicate: int,
    dp_shard: int,
) -> int:
    """Resolve ``dp_shard=-1`` and validate the dense world mesh formula."""
    world_size = require_positive_integer("world_size", world_size)
    pp = require_positive_integer("pp", pp)
    tp = require_positive_integer("tp", tp)
    cp = require_positive_integer("cp", cp)
    dp_replicate = require_positive_integer("dp_replicate", dp_replicate)
    dp_shard = require_integer("dp_shard", dp_shard)
    fixed_degree = pp * dp_replicate * cp * tp
    if dp_shard == -1:
        if world_size % fixed_degree != 0:
            raise ValueError(
                "world_size must be divisible by "
                "pp * dp_replicate * cp * tp when dp_shard=-1: "
                f"{world_size} % ({pp} * {dp_replicate} * {cp} * {tp}) != 0"
            )
        resolved = world_size // fixed_degree
        if resolved < 1:
            raise ValueError(
                "dp_shard=-1 leaves no rank for DP sharding: "
                f"world_size={world_size}, fixed parallel degree={fixed_degree}"
            )
    else:
        resolved = require_positive_integer("dp_shard", dp_shard)

    expected_world_size = fixed_degree * resolved
    if expected_world_size != world_size:
        raise ValueError(
            "parallel degrees do not match world_size: "
            f"pp({pp}) * dp_replicate({dp_replicate}) * "
            f"dp_shard({resolved}) * cp({cp}) * tp({tp}) "
            f"= {expected_world_size}, expected {world_size}"
        )
    return resolved


def validate_expert_parallelism(
    *,
    pp: int,
    ep: int,
    tp: int,
    cp: int,
    etp: int,
    dp_replicate: int,
    dp_shard: int,
    edp: int | None = None,
) -> tuple[int, int]:
    """Validate the sparse mesh and return ``(efsdp, derived_edp)``."""
    pp = require_positive_integer("pp", pp)
    ep = require_positive_integer("ep", ep)
    tp = require_positive_integer("tp", tp)
    cp = require_positive_integer("cp", cp)
    etp = require_positive_integer("etp", etp)
    dp_replicate = require_positive_integer("dp_replicate", dp_replicate)
    dp_shard = require_positive_integer("dp_shard", dp_shard)
    if etp not in (1, tp):
        raise ValueError(f"etp must be 1 or equal tp({tp}), got {etp}")

    expert_region = dp_shard * cp * tp
    expert_partition = ep * etp
    if expert_region % expert_partition != 0:
        raise ValueError(
            "ep * etp must divide dp_shard * cp * tp exactly: "
            f"({dp_shard} * {cp} * {tp}) % ({ep} * {etp}) != 0"
        )

    efsdp = expert_region // expert_partition
    derived_edp = dp_replicate * efsdp
    if edp is not None:
        supplied_edp = require_positive_integer("edp", edp)
        if supplied_edp != derived_edp:
            raise ValueError(
                "edp does not match the TorchTitan expert data mesh: "
                f"got {supplied_edp}, expected {derived_edp} "
                "(dp_replicate * dp_shard * cp * tp / (ep * etp))"
            )

    # This equivalent identity makes accidental formula changes explicit.
    world_size = pp * dp_replicate * dp_shard * cp * tp
    if pp * derived_edp * ep * etp != world_size:
        raise ValueError("internal expert mesh identity check failed")
    return efsdp, derived_edp


def validate_batch_sizes(
    *,
    local_batch_size: int,
    global_batch_size: int,
    dp_replicate: int,
    dp_shard: int,
) -> tuple[int, int, int]:
    """Return data degree, effective global batch, and accumulation steps."""
    local_batch_size = require_positive_integer(
        "local_batch_size", local_batch_size
    )
    global_batch_size = require_integer("global_batch_size", global_batch_size)
    dp_replicate = require_positive_integer("dp_replicate", dp_replicate)
    dp_shard = require_positive_integer("dp_shard", dp_shard)
    if global_batch_size < -1 or global_batch_size == 0:
        raise ValueError(
            "global_batch_size must be -1 or >= 1, "
            f"got {global_batch_size}"
        )
    data_parallel_degree = dp_replicate * dp_shard
    batch_per_accumulation = local_batch_size * data_parallel_degree

    if global_batch_size == -1:
        effective_global_batch_size = batch_per_accumulation
    else:
        effective_global_batch_size = require_positive_integer(
            "global_batch_size", global_batch_size
        )

    if effective_global_batch_size % batch_per_accumulation != 0:
        raise ValueError(
            "global_batch_size must be divisible by "
            "local_batch_size * dp_replicate * dp_shard: "
            f"{effective_global_batch_size} % "
            f"({local_batch_size} * {dp_replicate} * {dp_shard}) != 0"
        )

    gradient_accumulation_steps = (
        effective_global_batch_size // batch_per_accumulation
    )
    return (
        data_parallel_degree,
        effective_global_batch_size,
        gradient_accumulation_steps,
    )


def validate_pipeline_microbatches(
    *,
    pp: int,
    local_batch_size: int,
    pp_microbatch_size: int,
) -> int:
    """Return the number of pipeline microbatches after validation."""
    pp = require_positive_integer("pp", pp)
    local_batch_size = require_positive_integer(
        "local_batch_size", local_batch_size
    )
    pp_microbatch_size = require_positive_integer(
        "pp_microbatch_size", pp_microbatch_size
    )
    if pp == 1:
        return 1
    if local_batch_size % pp_microbatch_size != 0:
        raise ValueError(
            "local_batch_size must be divisible by pp_microbatch_size when PP is enabled: "
            f"{local_batch_size} % {pp_microbatch_size} != 0"
        )
    return local_batch_size // pp_microbatch_size


def validate_parallel_training_config(
    *,
    world_size: int,
    pp: int,
    ep: int,
    tp: int,
    cp: int,
    dp_replicate: int,
    dp_shard: int,
    pp_microbatch_size: int,
    local_batch_size: int,
    global_batch_size: int,
    edp: int | None = None,
    etp: int = 1,
) -> ValidatedParallelConfig:
    """Run the complete parameter-only preflight validation in documented order."""
    world_size = require_positive_integer("world_size", world_size)
    pp = require_positive_integer("pp", pp)
    ep = require_positive_integer("ep", ep)
    tp = require_positive_integer("tp", tp)
    cp = require_positive_integer("cp", cp)
    etp = require_positive_integer("etp", etp)
    dp_replicate = require_positive_integer("dp_replicate", dp_replicate)
    dp_shard = require_integer("dp_shard", dp_shard)
    local_batch_size = require_positive_integer(
        "local_batch_size", local_batch_size
    )
    pp_microbatch_size = require_positive_integer(
        "pp_microbatch_size", pp_microbatch_size
    )
    global_batch_size = require_integer("global_batch_size", global_batch_size)
    if global_batch_size < -1 or global_batch_size == 0:
        raise ValueError(
            "global_batch_size must be -1 or >= 1, "
            f"got {global_batch_size}"
        )
    if edp is not None:
        edp = require_positive_integer("edp", edp)

    resolved_dp_shard = resolve_dp_shard(
        world_size=world_size,
        pp=pp,
        tp=tp,
        cp=cp,
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
    )
    efsdp, derived_edp = validate_expert_parallelism(
        pp=pp,
        ep=ep,
        tp=tp,
        cp=cp,
        etp=etp,
        dp_replicate=dp_replicate,
        dp_shard=resolved_dp_shard,
        edp=edp,
    )
    (
        data_parallel_degree,
        effective_global_batch_size,
        gradient_accumulation_steps,
    ) = validate_batch_sizes(
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        dp_replicate=dp_replicate,
        dp_shard=resolved_dp_shard,
    )
    pipeline_microbatches = validate_pipeline_microbatches(
        pp=pp,
        local_batch_size=local_batch_size,
        pp_microbatch_size=pp_microbatch_size,
    )

    return ValidatedParallelConfig(
        world_size=world_size,
        pp=pp,
        ep=ep,
        tp=tp,
        cp=cp,
        etp=etp,
        dp_replicate=dp_replicate,
        dp_shard=resolved_dp_shard,
        efsdp=efsdp,
        edp=derived_edp,
        data_parallel_degree=data_parallel_degree,
        pp_microbatch_size=pp_microbatch_size,
        pipeline_microbatches=pipeline_microbatches,
        local_batch_size=local_batch_size,
        global_batch_size=effective_global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    _add_selective_parallel_arguments(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.world_size is None:
        parser.error("world_size is required; provide --world-size")
    try:
        result = validate_parallel_training_config(
            world_size=args.world_size,
            pp=args.pp,
            ep=args.ep,
            tp=args.tp,
            cp=args.cp,
            etp=args.etp,
            dp_replicate=args.dp_replicate,
            dp_shard=args.dp_shard,
            edp=args.edp,
            pp_microbatch_size=args.pp_microbatch_size,
            local_batch_size=args.local_batch_size,
            global_batch_size=args.global_batch_size,
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"parallel config validation failed: {error}") from error

    print(json.dumps(result._asdict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
