# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract of the backported backend-owned attention-metadata seam."""

import torch

import torchtitan_npu  # noqa: F401  imported for its patch chain: the seam under test
from torchtitan.models.common.attention import BaseAttention, VarlenAttention
from torchtitan.protocols.module import Module


def test_default_contract_is_the_backend_class_and_no_metadata():
    config = BaseAttention.Config(n_heads=1, inner_attention=Module.Config())

    assert config.attention_metadata_key is BaseAttention  # not overridden: the backend class
    assert config.build_attention_metadata(torch.zeros((1, 4), dtype=torch.int64)) is None


def test_build_stamps_the_key_so_a_rewritten_class_cannot_lose_it():
    """SimpleFSDP rewrites ``__class__``; the instance keeps the key its config built the map with."""
    instance = VarlenAttention.Config().build()

    assert instance.__dict__["_attention_metadata_key"] is VarlenAttention  # the default key is stamped too
    instance.__class__ = type("SimpleFSDPVarlenAttention", (VarlenAttention,), {})
    assert instance.attention_metadata_key is VarlenAttention
