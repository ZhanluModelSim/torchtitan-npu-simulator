# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():  # noqa: ANN202
    script_path = Path(__file__).parents[3] / "scripts" / "validate_parallel_config.py"
    spec = importlib.util.spec_from_file_location("validate_parallel_config", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_script_module()


def _valid_config(**overrides):  # noqa: ANN003, ANN202
    config = {
        "world_size": 256,
        "pp": 2,
        "ep": 32,
        "tp": 2,
        "cp": 1,
        "etp": 1,
        "dp_replicate": 1,
        "dp_shard": -1,
        "edp": 4,
        "pp_microbatch_size": 1,
        "local_batch_size": 2,
        "global_batch_size": -1,
    }
    config.update(overrides)
    return config


def test_selective_parser_extracts_dotted_flags_and_ignores_outer_args() -> None:
    argv = [
        "--module",
        "torchtitan_npu.simulator",
        "--parallelism.pipeline_parallel_degree=2",
        "--parallelism.expert-parallel-degree",
        "32",
        "--parallelism.tensor_parallel_degree",
        "2",
        "--parallelism.data_parallel_shard_degree=-1",
        "--parallelism.pipeline_parallel_microbatch_size",
        "1",
        "--training.local-batch-size=2",
        "--training.global_batch_size",
        "-1",
        "--unrelated-launcher-option",
        "value",
    ]
    original_argv = list(argv)

    values = validator.parse_parallel_training_args(argv, world_size=256)

    assert argv == original_argv
    assert values == _valid_config(edp=None)


def test_parse_and_validate_uses_torchtitan_defaults_for_omitted_flags() -> None:
    result = validator.parse_and_validate_parallel_training_args(
        ["--world-size", "8", "--training.local_batch_size", "2"]
    )

    assert result.dp_shard == 8
    assert result.pp == result.ep == result.tp == result.cp == 1
    assert result.local_batch_size == 2
    assert result.global_batch_size == 16


def test_selective_parser_rejects_missing_or_conflicting_world_size() -> None:
    with pytest.raises(ValueError, match="world_size is required"):
        validator.parse_parallel_training_args([])
    with pytest.raises(ValueError, match="conflicting world_size"):
        validator.parse_parallel_training_args(
            ["--world-size", "16"], world_size=8
        )


def test_resolves_auto_dp_shard_and_all_derived_values() -> None:
    result = validator.validate_parallel_training_config(**_valid_config())

    assert result.dp_shard == 64
    assert result.efsdp == 4
    assert result.edp == 4
    assert result.data_parallel_degree == 64
    assert result.global_batch_size == 128
    assert result.gradient_accumulation_steps == 1
    assert result.pipeline_microbatches == 2


def test_supports_hsdp_and_explicit_gradient_accumulation() -> None:
    result = validator.validate_parallel_training_config(
        **_valid_config(
            dp_replicate=2,
            dp_shard=32,
            edp=4,
            global_batch_size=512,
        )
    )

    assert result.data_parallel_degree == 64
    assert result.global_batch_size == 512
    assert result.gradient_accumulation_steps == 4


@pytest.mark.parametrize("value", [0, -2, 1.5, True])
def test_rejects_invalid_positive_degrees(value: object) -> None:
    error = TypeError if isinstance(value, (float, bool)) else ValueError
    with pytest.raises(error):
        validator.validate_parallel_training_config(**_valid_config(tp=value))


def test_rejects_auto_dp_shard_when_fixed_degrees_do_not_divide_world() -> None:
    with pytest.raises(ValueError, match="world_size must be divisible"):
        validator.validate_parallel_training_config(
            **_valid_config(world_size=250, edp=None)
        )


def test_rejects_explicit_dense_mesh_world_size_mismatch() -> None:
    with pytest.raises(ValueError, match="parallel degrees do not match"):
        validator.validate_parallel_training_config(
            **_valid_config(dp_shard=63, edp=None)
        )


def test_rejects_etp_that_is_neither_one_nor_tp() -> None:
    with pytest.raises(ValueError, match="etp must be 1 or equal tp"):
        validator.validate_parallel_training_config(
            **_valid_config(etp=3, edp=None)
        )


def test_rejects_non_integral_expert_mesh() -> None:
    with pytest.raises(ValueError, match=r"ep \* etp must divide"):
        validator.validate_parallel_training_config(
            **_valid_config(ep=30, edp=None)
        )


def test_rejects_supplied_edp_that_differs_from_derived_mesh() -> None:
    with pytest.raises(ValueError, match="edp does not match"):
        validator.validate_parallel_training_config(**_valid_config(edp=8))


@pytest.mark.parametrize("global_batch_size", [0, -2])
def test_rejects_global_batch_values_other_than_minus_one_or_positive(
    global_batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="global_batch_size must be -1 or >= 1"):
        validator.validate_parallel_training_config(
            **_valid_config(global_batch_size=global_batch_size)
        )


def test_rejects_global_batch_not_divisible_by_local_times_data_degree() -> None:
    with pytest.raises(ValueError, match="global_batch_size must be divisible"):
        validator.validate_parallel_training_config(
            **_valid_config(global_batch_size=129)
        )


def test_rejects_non_positive_pipeline_microbatch_even_without_pp() -> None:
    with pytest.raises(ValueError, match="pp_microbatch_size must be >= 1"):
        validator.validate_parallel_training_config(
            world_size=1,
            pp=1,
            ep=1,
            tp=1,
            cp=1,
            dp_replicate=1,
            dp_shard=1,
            pp_microbatch_size=0,
            local_batch_size=1,
            global_batch_size=-1,
        )


def test_rejects_local_batch_not_divisible_by_pipeline_microbatch() -> None:
    with pytest.raises(ValueError, match="local_batch_size must be divisible"):
        validator.validate_parallel_training_config(
            **_valid_config(local_batch_size=3, pp_microbatch_size=2)
        )


def test_pp_microbatch_is_ignored_for_splitting_when_pp_is_disabled() -> None:
    result = validator.validate_parallel_training_config(
        world_size=8,
        pp=1,
        ep=1,
        tp=1,
        cp=1,
        dp_replicate=1,
        dp_shard=8,
        pp_microbatch_size=16,
        local_batch_size=1,
        global_batch_size=-1,
    )

    assert result.pipeline_microbatches == 1


def test_individual_validators_enforce_their_own_domain_checks() -> None:
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        validator.resolve_dp_shard(
            world_size=0,
            pp=1,
            tp=1,
            cp=1,
            dp_replicate=1,
            dp_shard=-1,
        )
    with pytest.raises(ValueError, match="local_batch_size must be >= 1"):
        validator.validate_batch_sizes(
            local_batch_size=0,
            global_batch_size=-1,
            dp_replicate=1,
            dp_shard=1,
        )
    with pytest.raises(ValueError, match="pp_microbatch_size must be >= 1"):
        validator.validate_pipeline_microbatches(
            pp=2,
            local_batch_size=1,
            pp_microbatch_size=0,
        )
