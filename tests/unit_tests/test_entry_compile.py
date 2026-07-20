# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torchtitan_npu.entry as entry_mod
from torchtitan_npu.entry import (
    _compile_requires_bypass_triton_codegen,
    _has_model_converter,
    _uses_inductor_npu_ext,
)


def test_has_model_converter_accepts_model_config_name_entries():
    model_config = SimpleNamespace(name="npu_bypass_triton_codegen")
    owner = SimpleNamespace(_model_config=model_config)
    config = SimpleNamespace(converters=[SimpleNamespace(_owner=owner)])

    assert _has_model_converter(config, "npu_bypass_triton_codegen")


def test_has_model_converter_handles_missing_config():
    assert not _has_model_converter(None, "npu_bypass_triton_codegen")


def test_compile_strategy_helpers_document_model_groups():
    assert _uses_inductor_npu_ext("vlm")
    assert _uses_inductor_npu_ext("deepseek_v4")
    assert not _uses_inductor_npu_ext("llama3")

    assert not _compile_requires_bypass_triton_codegen("vlm")
    assert _compile_requires_bypass_triton_codegen("llama3")


def test_parse_config_uses_current_sys_argv(monkeypatch):  # noqa: ANN001
    parsed = object()
    received_args = []

    class FakeConfigManager:
        def parse_args(self, args):  # noqa: ANN001, ANN202
            received_args.extend(args)
            return parsed

    monkeypatch.setattr(entry_mod, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(
        entry_mod.sys,
        "argv",
        ["embedded", "--module", "torchtitan_npu.simulator", "--config", "demo"],
    )

    assert entry_mod._parse_config() is parsed
    assert received_args == [
        "--module",
        "torchtitan_npu.simulator",
        "--config",
        "demo",
    ]
