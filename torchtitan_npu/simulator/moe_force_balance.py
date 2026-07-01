# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forces deterministic (data-independent) behavior needed for meta-device
capture: MoE round-robin load balancing (reusing the existing
`debug.moe_force_load_balance` flag already wired through
`torchtitan_npu.models.deepseek_v4.moe.TokenChoiceTopKRouter`) and a fixed
RNG seed (see design doc §5.3 and §9)."""

from __future__ import annotations

import warnings
from typing import Any

DEFAULT_SIMULATION_SEED = 42


def force_moe_load_balance(config: Any) -> None:
    """Force `config.debug.moe_force_load_balance = True`, warning if the
    caller's config had it disabled. The simulator always needs
    deterministic MoE routing so the captured compute graph (in particular
    `num_tokens_per_expert`, and therefore the EP all-to-all split sizes)
    does not depend on real token data -- see design doc §5.3 for the proof
    that this makes every rank's routing decision identical."""
    debug_config = config.debug
    if not getattr(debug_config, "moe_force_load_balance", False):
        warnings.warn(
            "torchtitan_npu.simulator: forcing debug.moe_force_load_balance=True "
            "(config had it disabled). The simulator always uses deterministic "
            "round-robin MoE routing so the captured compute graph does not "
            "depend on real token data.",
            stacklevel=2,
        )
    debug_config.moe_force_load_balance = True


def force_deterministic_seed(config: Any, seed: int = DEFAULT_SIMULATION_SEED) -> None:
    """Force a fixed `config.debug.seed` if the caller left it unset.

    `torchtitan.distributed.utils.set_determinism()` broadcasts a seed
    derived from `torch.get_rng_state()` whenever `world_size > 1` and
    `debug.seed is None`; that code path calls `.to("cpu")` on a tensor
    living on the trainer's device and then `.item()`, which raises
    `NotImplementedError: Cannot copy out of meta tensor; no data!` under
    the simulator's meta-device execution. Supplying an explicit seed
    means `set_determinism()` takes the "already have a seed" branch and
    never touches that code path.
    """
    if config.debug.seed is None:
        config.debug.seed = seed
