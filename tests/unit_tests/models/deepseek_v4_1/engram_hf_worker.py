# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU HF export and EP / joint EFSDP-EP shard restore with native padding."""

import json
import sys
from datetime import timedelta
from itertools import product
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files_on_every_rank

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.state_dict_adapter import DeepSeekV41StateDictAdapter


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        config = model_registry("deepseek_v4_1_debugmodel_text").model
        table = config.layers[1].engram.table
        table.head_vocab_sizes = (2, 3, 5, 7, 11, 13)
        table.num_embeddings = 48
        table.embedding_dim = 32
        root = Path(sys.argv[1])
        assets = root / "assets"
        if rank == 0:
            assets.mkdir()
            (assets / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.1.engram.embed.weight": "model-00001-of-00002.safetensors",
                            "layers.1.engram.q_weight": "model-00002-of-00002.safetensors",
                        }
                    }
                )
            )
        dist.barrier()
        weight = torch.arange(48 * 32, dtype=torch.float32).reshape(48, 32)
        gate = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        # Four-rank local runs cover EP2 replicas and four distinct joint shards.
        # The default two-rank UT remains small while checking both key formats.
        for layout, mapped in product(("ep", "efsdp_ep"), (False, True)):
            shards = 2 if layout == "ep" else dist.get_world_size()
            shard_rank = rank % shards
            rows = 48 // shards
            offset = shard_rank * rows
            key = f"layers.1.engram.table.weight.{layout}_shard_{shard_rank:05d}_of_{shards:05d}"
            adapter = DeepSeekV41StateDictAdapter(config, str(assets) if mapped else None)
            native = {key: weight[offset : offset + rows].clone(), "layers.1.engram.gate.q_weight": gate}
            # Trainer casts native state before calling to_hf for last-save export.
            native = {name: value.to(torch.bfloat16) for name, value in native.items()}
            hf = adapter.to_hf(native)
            assert type(hf["layers.1.engram.embed.weight"]) is torch.Tensor
            path = root / f"{layout}_{'mapped' if mapped else 'unmapped'}"
            mapping = adapter.fqn_to_index_mapping
            dcp.save(
                hf,
                storage_writer=HuggingFaceStorageWriter(
                    str(path / "sharded" if mapping else path),
                    save_distributed=True,
                    fqn_to_index_mapping=mapping,
                    enable_consolidation=not mapping,
                ),
            )
            if mapping:
                consolidate_safetensors_files_on_every_rank(
                    input_dir=str(path / "sharded"),
                    output_dir=str(path),
                    fqn_to_index_mapping=mapping,
                )
            restored = {
                "layers.1.engram.embed.weight": torch.empty(41, 32, dtype=torch.bfloat16),
                "layers.1.engram.q_weight": torch.empty_like(gate, dtype=torch.bfloat16),
            }
            dcp.load(restored, storage_reader=HuggingFaceStorageReader(str(path)))
            torch.testing.assert_close(restored["layers.1.engram.embed.weight"], weight[:41].bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(restored["layers.1.engram.q_weight"], gate.bfloat16(), rtol=0, atol=0)
            target = adapter.to_hf({key: torch.full((rows, 32), -1.0)})
            dcp.load(target, storage_reader=HuggingFaceStorageReader(str(path)))
            loaded = adapter.from_hf(target)
            assert set(loaded) == {key}
            valid = max(0, min(rows, 41 - offset))
            expected = weight[offset : offset + valid].bfloat16().float()
            torch.testing.assert_close(loaded[key][:valid], expected, rtol=0, atol=0)
            assert torch.count_nonzero(loaded[key][valid:]) == 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
