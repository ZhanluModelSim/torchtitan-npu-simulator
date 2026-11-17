#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the tokenizer-compression lookup used by DeepSeek Engram."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from torchtitan_npu.models.deepseek_v4_1.engram.token_map import build_token_id_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tokenizer",
        type=Path,
        help="HF tokenizer directory or tokenizer.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .npy path (default: TOKENIZER_DIR/engram_token_id_map.npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer
    tokenizer_json = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Tokenizer JSON does not exist: {tokenizer_json}")
    output = args.output or tokenizer_json.parent / "engram_token_id_map.npy"
    if output.suffix != ".npy":
        raise ValueError(f"Engram token map output must use the .npy suffix, got {output}.")

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    mapping = build_token_id_map(tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, mapping, allow_pickle=False)
    print(f"Saved {mapping.size} token IDs -> {int(mapping.max()) + 1} canonical IDs to {output}")


if __name__ == "__main__":
    main()
