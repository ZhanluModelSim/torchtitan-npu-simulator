// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "../_aclnn_common.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_lightning_indexer_grad_kl_loss(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &query_index, const at::Tensor &key_index,
    const at::Tensor &weight, const at::Tensor &sparse_indices, const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, const c10::optional<std::vector<int64_t>> &actual_seq_lengths_query,
    const c10::optional<std::vector<int64_t>> &actual_seq_lengths_key, std::string layout, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, double scale_value, bool deterministic) {
  const char *layout_ptr = layout.c_str();

  auto seq_q_tmp = actual_seq_lengths_query.value_or(std::vector<int64_t>{});
  auto seq_k_tmp = actual_seq_lengths_key.value_or(std::vector<int64_t>{});
  c10::optional<at::IntArrayRef> seq_q(seq_q_tmp);
  c10::optional<at::IntArrayRef> seq_k(seq_k_tmp);

  at::Tensor d_query_index = at::zeros(query_index.sizes(), query_index.options());
  at::Tensor d_key_index = at::zeros(key_index.sizes(), key_index.options());
  at::Tensor d_weight = at::zeros(weight.sizes(), weight.options());
  at::Tensor loss = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(query.device()));

  ACLNN_CMD(aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index, weight, sparse_indices,
            softmax_max, softmax_sum, query_rope, key_rope, seq_q, seq_k, scale_value, layout_ptr, sparse_mode,
            pre_tokens, next_tokens, deterministic, cmp_ratio, d_query_index, d_key_index, d_weight, loss);

  return {d_query_index, d_key_index, d_weight, loss};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("npu_sparse_lightning_indexer_grad_kl_loss", &npu_sparse_lightning_indexer_grad_kl_loss,
        "Sparse Lightning Indexer Grad KL Loss");
}
