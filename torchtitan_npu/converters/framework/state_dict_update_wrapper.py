# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import TYPE_CHECKING, Any

from torchtitan.protocols.model_spec import ModelSpec

if TYPE_CHECKING:
    from torchtitan_npu.converters.model_custom_interface import StateDictUpdater

logger = logging.getLogger(__name__)


def get_state_dict_adapter_wrapper(cls):
    class StateDictUpdateWrapper(cls):
        """Wrapper for state dict update functionality"""

        _updater_cls_list: list[type["StateDictUpdater"]] = []

        @classmethod
        def add_state_dict_updater(cls, updater_cls: type["StateDictUpdater"]) -> bool:
            """Register an updater once and report whether it was added."""
            if updater_cls in cls._updater_cls_list:
                return False
            cls._updater_cls_list.append(updater_cls)
            return True

        def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
            """Apply to_hf transformation"""
            for updater_cls in self._updater_cls_list:
                state_dict = updater_cls.to_hf(state_dict)
            state_dict = super().to_hf(state_dict)
            return state_dict

        def from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
            """Apply from_hf transformation"""
            state_dict = super().from_hf(state_dict)
            for updater_cls in self._updater_cls_list:
                state_dict = updater_cls.from_hf(state_dict)
            return state_dict

    return StateDictUpdateWrapper


def apply_state_dict_update(updater_cls: type["StateDictUpdater"], model_spec: ModelSpec):
    if not hasattr(model_spec, "state_dict_adapter"):
        raise RuntimeError("[StateDictUpdateWrapper] TrainSpec does not have state_dict_adapter.")

    state_dict_adapter = model_spec.state_dict_adapter
    if state_dict_adapter is None:
        raise RuntimeError("[StateDictUpdateWrapper] TrainSpec.state_dict_adapter is None.")

    if not hasattr(state_dict_adapter, "add_state_dict_updater"):
        adapter_wrapper = get_state_dict_adapter_wrapper(state_dict_adapter)
        model_spec.state_dict_adapter = adapter_wrapper
        state_dict_adapter = adapter_wrapper

    # pyrefly: ignore [missing-attribute]
    if not state_dict_adapter.add_state_dict_updater(updater_cls):
        logger.info(f"[StateDictUpdateWrapper] Skip duplicate StateDictUpdater {updater_cls}.")
        return

    logger.info(f"[StateDictUpdateWrapper] Add StateDictUpdater {updater_cls}.")
