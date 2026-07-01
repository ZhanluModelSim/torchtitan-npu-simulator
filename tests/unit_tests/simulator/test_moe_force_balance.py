# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan_npu.simulator.moe_force_balance import (
    DEFAULT_SIMULATION_SEED,
    force_deterministic_seed,
    force_moe_load_balance,
)


def _config(moe_force_load_balance: bool = False, seed=None) -> SimpleNamespace:
    return SimpleNamespace(debug=SimpleNamespace(moe_force_load_balance=moe_force_load_balance, seed=seed))


def test_force_moe_load_balance_sets_true_and_warns_when_disabled():
    config = _config(moe_force_load_balance=False)
    with pytest.warns(UserWarning, match="moe_force_load_balance"):
        force_moe_load_balance(config)
    assert config.debug.moe_force_load_balance is True


def test_force_moe_load_balance_no_warning_when_already_enabled():
    config = _config(moe_force_load_balance=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        force_moe_load_balance(config)  # must not raise/warn
    assert config.debug.moe_force_load_balance is True


def test_force_deterministic_seed_sets_default_when_none():
    config = _config(seed=None)
    force_deterministic_seed(config)
    assert config.debug.seed == DEFAULT_SIMULATION_SEED


def test_force_deterministic_seed_respects_existing_seed():
    config = _config(seed=123)
    force_deterministic_seed(config)
    assert config.debug.seed == 123


def test_force_deterministic_seed_accepts_custom_seed():
    config = _config(seed=None)
    force_deterministic_seed(config, seed=7)
    assert config.debug.seed == 7
