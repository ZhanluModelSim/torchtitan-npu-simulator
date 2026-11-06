# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture support for DeepSeek-V4.1's host-offload Engram table.

The production Engram table is CPU-resident (``uses_host_offload``) and its
initialization asserts a CPU weight. Meta capture builds the model on the meta
device, where the assertion is wrong but nothing else changes: initialization
performs no allocation and the table's real-device residency stays CPU, which
is a memory-budget fact rather than a graph fact.
"""

import torch


def install_deepseek_v4_1_meta_support(scope):
    """Allow the host-offload Engram table to initialize on meta."""
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

    def _meta_init_self_parameters(self):
        # Mirror the production seed derivation exactly; only the device
        # assertion is relaxed for meta, where init allocates nothing.
        if self.weight.device.type not in {"cpu", "meta"}:
            raise RuntimeError(f"Host-offload Engram weight must initialize on CPU or meta, got {self.weight.device}.")
        shard_rank = self._table_mesh.get_local_rank() if self._table_mesh is not None else self._ep_rank
        seed = (torch.initial_seed() + 1000003 * self.layer_id + 9176 * shard_rank) % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._init_param("weight", self.weight)

    scope.replace(HostEngramTable, "_init_self_parameters", _meta_init_self_parameters)


def _build_attention_masks(self, tokens, labels, extra_inputs, *, cp_mesh=None, load_balancer_type=None):
    """The runner's transitional seam for the V4.1 attention-metadata map.

    V4.1 has no model-level ``build_attention_masks``; the pinned decoder
    patch constructs the ``AttentionMetadataMap`` itself through
    ``_get_attention_metadata``. The adapter reuses exactly that builder so
    the capture path matches the production prepare-input branch, and keeps
    CP out of scope as for the other families.
    """
    if cp_mesh is not None:
        if load_balancer_type is not None:
            raise NotImplementedError("DeepSeek-V4.1 simulator capture does not use a CP load balancer")
        raise NotImplementedError("DeepSeek-V4.1 simulator capture does not support context parallel")
    from torchtitan_npu.patches.torchtitan.models.common.decoder import _get_attention_metadata

    metadata = _get_attention_metadata(self, extra_inputs.get("positions"))
    extra = dict(extra_inputs)
    if metadata:
        extra["attention_metadata"] = metadata
    return tokens, labels, extra


def install_deepseek_v4_1_input_adapter(scope):
    """Bind the runner's transitional input contract to the metadata map."""
    from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model

    scope.replace(DeepSeekV41Model, "build_attention_masks", _build_attention_masks)


__all__ = ["install_deepseek_v4_1_input_adapter", "install_deepseek_v4_1_meta_support"]
