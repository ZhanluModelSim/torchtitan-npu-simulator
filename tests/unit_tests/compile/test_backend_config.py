# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

torch_npu = pytest.importorskip("torch_npu")  # registers the config key

import torch._inductor.config as inductor_config

from torchtitan_npu.compile import configure_npu_backend

_NPU_BACKEND_ENV = "TORCHINDUCTOR_NPU_BACKEND"


def _npu_backend() -> str:
    """Read the npu_backend config key, registering it on first access."""
    inductor_config.get_config_copy()  # pyrefly: ignore [missing-attribute]
    return getattr(inductor_config, "npu_backend", "default")


@pytest.fixture
def restore_backend(monkeypatch):
    monkeypatch.delenv(_NPU_BACKEND_ENV, raising=False)
    before = _npu_backend()
    yield
    inductor_config.npu_backend = before  # pyrefly: ignore [missing-attribute]


def test_configure_defaults_to_ascendc(restore_backend):
    configure_npu_backend()
    assert _npu_backend() == "ascendc"


def test_configure_is_idempotent(restore_backend):
    configure_npu_backend()
    configure_npu_backend()
    assert _npu_backend() == "ascendc"


def test_env_escape_hatch_leaves_config_untouched(restore_backend, monkeypatch):
    inductor_config.npu_backend = "default"  # pyrefly: ignore [missing-attribute]
    monkeypatch.setenv(_NPU_BACKEND_ENV, "default")
    configure_npu_backend()
    # Once set, the config layer would shadow the env variable in torch_npu's
    # resolution order, so configure must not write it when the env pins a
    # backend — even to pin "default" explicitly.
    assert _npu_backend() == "default"
    assert os.environ[_NPU_BACKEND_ENV] == "default"
