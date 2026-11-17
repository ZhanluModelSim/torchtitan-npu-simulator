# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Engram sparse table: model side, host side and token compression.

``core`` owns the module and its table, ``config`` the recipe-facing configs,
``host``/``lookup`` the host-side table and its sparse gather, ``token_map`` the
tokenizer compression the table is keyed by.
"""

from .config import _make_engram_configs
from .core import Engram, EngramContextGate, EngramTable
from .host import HostEngramTable
from .lookup import HostEngramLookup

__all__ = [
    "Engram",
    "EngramContextGate",
    "EngramTable",
    "HostEngramLookup",
    "HostEngramTable",
    "_make_engram_configs",
]
