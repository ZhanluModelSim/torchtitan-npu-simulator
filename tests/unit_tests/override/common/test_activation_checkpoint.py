# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the selective-AC save-op edit override."""

from dataclasses import dataclass, field

import pytest
import torch
from torchtitan.config import Configurable, OverrideConfig, apply_overrides
from torchtitan.config.override import parse_cli_imports
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC

from torchtitan_npu.override.common.activation_checkpoint import (
    SelectiveACWithSaveOpEdits,
    update_save_ops,
)

pytestmark = pytest.mark.cpu

_OVERRIDE_TARGET = "torchtitan_npu.override.common.activation_checkpoint.update_save_ops"


class _Root(Configurable):
    """Minimal config root holding the activation-checkpoint node."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        activation_checkpoint: SelectiveAC.Config = field(default_factory=SelectiveAC.Config)


def _default_save_ops() -> set:
    return SelectiveAC.Config().build().get_save_ops()


def _edited_save_ops(edits: list[str]) -> set:
    config = update_save_ops(SelectiveAC.Config(), edits=edits)
    return config.build().get_save_ops()


def test_update_save_ops_override_replaces_the_selective_policy() -> None:
    imports = parse_cli_imports([f'{_OVERRIDE_TARGET}={{"edits": "-torch.ops.aten.bmm.default"}}'])
    config = _Root.Config(
        activation_checkpoint=SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=["moe.router.gate"])
    )

    replacements = apply_overrides(OverrideConfig(imports=imports), config)

    assert len(replacements) == 1
    replaced = config.activation_checkpoint
    assert isinstance(replaced, SelectiveACWithSaveOpEdits.Config)
    assert replaced.save_op_edits == ["-torch.ops.aten.bmm.default"]
    assert replaced.force_recompute_mm_shapes_by_fqns == ["moe.router.gate"]


def test_update_save_ops_override_without_edits_keeps_the_default_save_set() -> None:
    config = _Root.Config()

    apply_overrides(OverrideConfig(imports=[_OVERRIDE_TARGET]), config)

    assert isinstance(config.activation_checkpoint, SelectiveACWithSaveOpEdits.Config)
    assert config.activation_checkpoint.save_op_edits == []
    assert config.activation_checkpoint.build().get_save_ops() == _default_save_ops()


def test_update_save_ops_override_does_not_replace_full_ac() -> None:
    config = _Root.Config(activation_checkpoint=FullAC.Config())

    replacements = apply_overrides(OverrideConfig(imports=[_OVERRIDE_TARGET]), config)

    assert replacements == []
    assert type(config.activation_checkpoint) is FullAC.Config


def test_empty_edits_keep_the_default_save_set() -> None:
    assert _edited_save_ops([]) == _default_save_ops()


def test_drop_edit_removes_only_the_named_op() -> None:
    dropped = _edited_save_ops(["-torch.ops.aten.bmm.default"])

    assert torch.ops.aten.bmm.default not in dropped
    assert dropped == _default_save_ops() - {torch.ops.aten.bmm.default}


def test_add_edit_adds_only_the_named_op() -> None:
    added = _edited_save_ops(["+torch.ops.aten.relu.default"])

    assert torch.ops.aten.relu.default in added
    assert added == _default_save_ops() | {torch.ops.aten.relu.default}


def test_edits_apply_in_order_so_the_last_edit_wins() -> None:
    dropped_then_added = _edited_save_ops(["-torch.ops.aten.mm.default", "+torch.ops.aten.mm.default"])
    added_then_dropped = _edited_save_ops(["+torch.ops.aten.mm.default", "-torch.ops.aten.mm.default"])

    assert torch.ops.aten.mm.default in dropped_then_added
    assert torch.ops.aten.mm.default not in added_then_dropped


def test_factory_parses_a_comma_separated_edit_string() -> None:
    config = update_save_ops(
        SelectiveAC.Config(),
        edits="-torch.ops.aten.bmm.default,+torch.ops.aten.relu.default",
    )

    assert config.save_op_edits == ["-torch.ops.aten.bmm.default", "+torch.ops.aten.relu.default"]


@pytest.mark.parametrize(
    "edit",
    [
        "torch.ops.aten.bmm.default",
        "+torch.ops.aten.bmm",
        "+torch.ops.aten",
        "+torch.ops.aten.not_an_op.default",
        "",
    ],
    ids=["missing-sign", "overload-packet", "namespace", "unknown-op", "empty"],
)
def test_factory_rejects_an_invalid_edit(edit: str) -> None:
    with pytest.raises(ValueError):
        update_save_ops(SelectiveAC.Config(), edits=edit)
