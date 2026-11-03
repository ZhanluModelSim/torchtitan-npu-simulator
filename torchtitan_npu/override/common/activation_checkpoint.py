# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: selective AC with caller-selected edits to the saved-op set.

The selective-AC save set is fixed in torchtitan, and some of its entries are
always saved (``bmm``, matmul shapes not covered by
``force_recompute_mm_shapes_by_fqns``). This override keeps the selected
policy's knobs -- including ``force_recompute_mm_shapes_by_fqns`` -- and
applies a list of edits to that set, in order:

    +torch.ops.aten.bmm.default   add the op (always save its outputs)
    -torch.ops.aten.bmm.default   drop the op (always recompute its outputs)

Each entry must start with ``+`` or ``-`` followed by a fully qualified op
overload such as ``torch.ops.aten.bmm.default``. The path is resolved when the
override is installed, so it must be registered by then; namespaces, overload
packets and unknown paths are rejected there instead of silently doing
nothing. The default is an empty list, so installing the override is a no-op
baseline.

Import it through ``--override.imports`` together with
``activation-checkpoint:selective``, e.g.:

    'torchtitan_npu.override.common.activation_checkpoint.update_save_ops={"edits": "-torch.ops.aten.bmm.default"}'
"""

from dataclasses import dataclass, field
from typing import cast

import torch
from torchtitan.config import derive, override
from torchtitan.distributed.activation_checkpoint import SelectiveAC

_OP_PREFIX = "torch.ops."


def _parse_edits(edits: list[str] | str | None) -> list[str]:
    """Validate and resolve the ``edits`` override kwarg.

    Override kwargs are parsed from JSON by torchtitan (``parse_cli_imports``),
    not by tyro, so a comma-separated string and a JSON array are both accepted
    here. Each entry is resolved so an unparseable op fails at installation,
    before the model is built.
    """
    if edits is None:
        return []
    entries = edits.split(",") if isinstance(edits, str) else edits
    parsed = []
    for entry in entries:
        entry = str(entry).strip()
        if not entry or entry[0] not in ("+", "-"):
            raise ValueError(f"save-op edit must be '+torch.ops.<path>' or '-torch.ops.<path>', got {entry!r}.")
        name = entry[1:]
        if not name.startswith(_OP_PREFIX) or name == _OP_PREFIX:
            raise ValueError(f"save-op edit must use a fully qualified 'torch.ops.' name, got {entry!r}.")
        _resolve_op(name)
        parsed.append(entry)
    return parsed


def _resolve_op(qualified: str):
    """Resolve a ``torch.ops.<path>`` name to a specific op overload."""
    obj = torch.ops
    for part in qualified[len(_OP_PREFIX) :].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(f"save-op edit {qualified!r} is not a registered torch op.") from exc
    if not isinstance(obj, torch._ops.OpOverload):
        raise ValueError(
            f"save-op edit {qualified!r} must name a specific op overload "
            f"(e.g. 'torch.ops.aten.mm.default'), got {type(obj).__name__}."
        )
    return obj


class SelectiveACWithSaveOpEdits(SelectiveAC):
    """Selective AC whose save set is updated by the configured op edits."""

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        save_op_edits: list[str] = field(default_factory=list)

    def get_save_ops(self) -> set:
        save_ops = set(super().get_save_ops())
        config = cast("SelectiveACWithSaveOpEdits.Config", self.config)
        for entry in config.save_op_edits:
            op = _resolve_op(entry[1:])
            if entry[0] == "+":
                save_ops.add(op)
            else:
                save_ops.discard(op)
        return save_ops


@override(
    target=SelectiveAC.Config,
    description="Add or drop ops in the selective activation-checkpointing save set",
)
def update_save_ops(
    cfg: SelectiveAC.Config,
    edits: list[str] | str | None = None,
) -> SelectiveACWithSaveOpEdits.Config:
    """Install the editing policy; ``edits`` defaults to empty (a no-op)."""
    return derive(cfg, SelectiveACWithSaveOpEdits.Config, save_op_edits=_parse_edits(edits))
