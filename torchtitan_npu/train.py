# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU-aware torchtitan training entry point."""

from torchtitan.train import main

import torchtitan_npu  # noqa: F401

if __name__ == "__main__":
    main()
