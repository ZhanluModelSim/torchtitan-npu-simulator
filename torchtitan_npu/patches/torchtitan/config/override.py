# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/6180
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/6315

"""Patch ``torchtitan.config.apply_overrides`` to group logs and fold FQN ranges.

Backports grouped override success logs from upstream PR #6180 and indexed FQN
range folding from PR #6315 to TorchTitan 0.4.0 while keeping every replaced
line in the return list. Remove this module after the TorchTitan dependency
includes the PRs.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import logging
import sys
from collections import defaultdict
from typing import TYPE_CHECKING

import torchtitan.config

from torchtitan_npu.patches.torchtitan.config.format import fold_indexed_paths

if TYPE_CHECKING:
    from torchtitan.config.configurable import Configurable
    from torchtitan.config.override import OverrideConfig

logger = logging.getLogger(__name__)

# ``torchtitan.config`` exports the ``@override`` decorator, shadowing the
# submodule name; resolve the engine module explicitly via ``importlib``.
_override = importlib.import_module("torchtitan.config.override")
_original_apply_overrides = _override.apply_overrides

_CALLER_MODULES = (
    "torchtitan.trainer",
    "torchtitan.experiments.torchft.trainer",
    "torchtitan.experiments.graph_trainer.precompile_main",
    "torchtitan.rl.controller.controller",
    "torchtitan.rl.generate",
)


def _upstream_has_folded_override_logs() -> bool:
    if importlib.util.find_spec("torchtitan.config.format") is None:
        return False
    upstream_format = importlib.import_module("torchtitan.config.format")
    return hasattr(upstream_format, "fold_indexed_paths")


@functools.wraps(_original_apply_overrides)
def patched_apply_overrides(
    override_config: OverrideConfig,
    config_root: Configurable.Config,
) -> list[str]:
    """Apply overrides with grouped INFO logs and folded indexed FQN ranges."""
    entries = [(e, {}) if isinstance(e, str) else (e[0], e[1]) for e in override_config.imports]
    targets = [(*_override._resolve_target(target), kwargs) for target, kwargs in entries]

    resolved = _override._resolve_active(targets)
    claims = _override._collect_claims(resolved, config_root)
    _override._check_node_conflicts(claims)

    replacements: list[str] = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in claims:
        # ``**{}`` for a bare (no-kwargs) entry is just ``factory(cfg)``; a kwarg
        # the factory does not accept raises TypeError here, naming the factory.
        new_cfg = c.ov.factory(c.cfg, **c.kwargs)
        if isinstance(c.parent, list) and isinstance(c.attr, int):
            c.parent[c.attr] = new_cfg
        elif isinstance(c.attr, str):
            setattr(c.parent, c.attr, new_cfg)
        else:
            raise ValueError(f"Override '{c.ov.name}' claims root config '{c.fqn}', which cannot be replaced in place.")
        kwargs_note = " with " + ", ".join(f"{k}={v!r}" for k, v in c.kwargs.items()) if c.kwargs else ""
        transition = f"{type(c.cfg).__qualname__} -> {type(new_cfg).__qualname__}{kwargs_note}"
        replacements.append(f"[Override] {c.ov.name}: {c.fqn} {transition}")
        groups[c.ov.name, transition].append(c.fqn)

    if replacements:
        for (factory, transition), fqns in groups.items():
            _override.logger.info(
                "[Override] %s: %s matched=%d fqns=%s",
                factory,
                transition,
                len(fqns),
                fold_indexed_paths(fqns),
            )
        _override.logger.info("Applied %d override(s)", len(replacements))
    elif override_config.imports:
        _override.logger.warning(
            f"override.imports={override_config.imports} produced no "
            "replacements. Check that the imported modules register overrides "
            "whose `target` and `fqns` match nodes in the config tree."
        )

    return replacements


def apply() -> None:
    """Install the override logger backport across public and cached bindings."""
    if _upstream_has_folded_override_logs():
        return
    if getattr(sys.modules["torchtitan.config.override"], "apply_overrides", None) is patched_apply_overrides:
        return
    logger.info("[PATCH] torchtitan.config.apply_overrides -> grouped folded logs (upstream #6180, #6315)")
    sys.modules["torchtitan.config.override"].apply_overrides = (  # pyrefly: ignore [missing-attribute]
        patched_apply_overrides
    )
    torchtitan.config.apply_overrides = patched_apply_overrides
    for module_name in _CALLER_MODULES:
        if module_name in sys.modules:
            sys.modules[module_name].apply_overrides = (  # pyrefly: ignore [missing-attribute]
                patched_apply_overrides
            )


apply()
