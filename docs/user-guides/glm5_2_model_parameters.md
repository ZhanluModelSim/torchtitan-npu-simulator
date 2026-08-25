# GLM-5.2 训练建模参数

GLM-5.2 通过 `torchtitan_npu.models.glm5_2` 注册，复用 DSV3.2 的 MLA、DSA、MoE、MTP、TP/EP/CP/PP 和 simulator 执行框架。

## 官方结构映射

| 参数 | GLM-5.2 | 建模位置 |
| --- | ---: | --- |
| vocab size | 154880 | `tok_embeddings` / `output` |
| hidden size | 6144 | `dim` |
| main layers | 78 | `n_layers` |
| dense prefix | 3 | `n_dense_layers` |
| routed experts | 256 | `num_experts` |
| active experts/token | 8 | `router_top_k` |
| shared experts | 1 | `num_shared_experts` |
| dense FFN width | 12288 | `dense_hidden_dim` |
| routed expert width | 2048 | `moe_hidden_dim` |
| attention heads | 64 | `n_heads` |
| Q LoRA rank | 2048 | `q_lora_rank` |
| KV LoRA rank | 512 | `kv_lora_rank` |
| Q/K no-PE / RoPE dim | 192 / 64 | `qk_nope_head_dim` / `qk_rope_head_dim` |
| V head dim | 256 | `v_head_dim` |
| indexer heads / dim | 32 / 128 | `index_n_heads` / `index_head_dim` |
| DSA top-k | 2048 | `index_topk` |
| MTP | 1 | `num_mtp_modules` |
| max sequence length | 1048576 | `rope_max_seq_len` |

官方 IndexShare schedule 为：第 0–2 层 full，之后每 4 层一个 full indexer，其余层 shared；MTP iteration 复用最后一个主干层的 top-k。shared 层不创建 indexer 参数，也不计算 indexer auxiliary loss。

## 配置入口

```powershell
& 'D:\HW_project\.conda-dsv32\python.exe' -m torchtitan_npu.entry `
  --module torchtitan_npu.models.glm5_2 `
  --config glm5_2_smoketest `
  --training.steps 1
```

TTNS meta 仿真：

```powershell
& 'D:\HW_project\.conda-dsv32\python.exe' scripts/run_simulator_spawn.py `
  --config glm5_2_smoketest `
  --simulation.world-size 1 `
  --simulation.no-enable-memory-tracking
```

完整 78 层配置名为 `glm5_2_78layers_1mtp`。它用于结构、参数、并行规划和内存模型验收；实际训练仍需要匹配的 GLM-5.2 checkpoint、tokenizer、CANN 与 `torch_npu` 运行时。
