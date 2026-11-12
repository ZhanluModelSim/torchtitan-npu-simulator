# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic config-tree filtering and mutation helpers (model-agnostic)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torchtitan.models.base.module import Module


class ConfigFilterFn(Protocol):
    """Predicate over a TorchTitan model-config node and its config-tree FQN."""

    def __call__(self, config: Module.Config, fqn: str) -> bool:
        raise NotImplementedError


def any_config_filter(*filters: ConfigFilterFn) -> ConfigFilterFn:
    """Combine config-tree filters with logical OR."""

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(filter_fn(config, fqn) for filter_fn in filters)

    return _filter


def match_config_fqn_suffix(*suffixes: str) -> ConfigFilterFn:
    """Match config-tree FQNs by a dotted suffix or an exact root name.

    ``Config.traverse`` reports nested fields as dotted FQNs, but a field on
    the model config itself has no leading component (for example, a token
    dispatcher node).  Accepting both forms keeps filters useful for root-owned
    extension configs as well as nested modules.
    """
    normalized = tuple(suffix.lstrip(".") for suffix in suffixes)

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(fqn == suffix or fqn.endswith(f".{suffix}") for suffix in normalized)

    return _filter


def _match_fqn_suffix(fqn: str, pattern: str) -> bool:
    """Match a config-tree FQN by a dotted suffix pattern (``.attention.wkv``)."""
    normalized = pattern.lstrip(".")
    return fqn == normalized or fqn.endswith(f".{normalized}")


def _replace_config(model_config, parent: object | None, attr: str | int | None, replacement):
    """Replace one config-tree node, including the uncommon root-node case."""

    if parent is None:
        return replacement
    if isinstance(parent, list):
        assert isinstance(attr, int)
        parent[attr] = replacement
    else:
        assert isinstance(attr, str)
        setattr(parent, attr, replacement)
    return model_config
