# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan_npu.models import deepseek_v4 as dsv4
from torchtitan_npu.models.deepseek_v4 import state_dict_adapter


N_HASH_LAYERS = 3


N_LAYERS = 4


NUM_EXPERTS = 4


def _build_model_config(num_experts=NUM_EXPERTS, num_layers=N_LAYERS):
    return dsv4._make_v4_config(
        dim=32, n_layers=num_layers, vocab_size=512, n_heads=4, head_dim=8, rope_head_dim=4, q_lora_rank=16,
        o_lora_rank=8, n_groups=2, compress_ratios=(1,) * num_layers, window_size=32, norm_eps=1e-6, index_n_heads=2,
        index_head_dim=8, index_topk=2, moe_inter_dim=16, num_experts=num_experts, num_shared_experts=1, top_k=1,
        n_hash_layers=min(N_HASH_LAYERS, num_layers), route_norm=False, route_scale=1.0, load_balance_coeff=1e-3, hc_mult=2,
        sinkhorn_iters=2, hc_eps=1e-6, max_seq_len=32, compress_rope_theta=10000.0, original_seq_len=32,
        num_mtp_layers=1,
    )


def _adapter_with_converter(converter_config):
    model_config = converter_config.build().convert(_build_model_config())
    return state_dict_adapter.DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
