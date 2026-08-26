# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU-aware torchtitan training entry point."""

import torchtitan_npu  # noqa: F401  # Install NPU config and package patches first.


def main() -> None:
    from torchtitan.train import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
