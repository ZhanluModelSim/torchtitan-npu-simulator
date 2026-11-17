# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect TorchFT tests only when their optional distribution is installed."""

from importlib.util import find_spec

collect_ignore_glob = ["test_*.py"] if find_spec("torchft") is None else []
