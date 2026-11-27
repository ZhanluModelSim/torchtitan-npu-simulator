# DeepSeek-V4 系列上下文并行

DeepSeek-V4 和 V4.1 的 BF16 文本配方支持逐文档上下文并行，拓扑使用 `TP=PP=1`。配方需同时启用对应模型的 AscendC `sparse_attn.asc` 和 `lightning_indexer.asc`，保证 teacher 的生产端和消费端配套。

## 配置

在 Python recipe 中设置 `ParallelismConfig.context_parallel_degree` 和 `context_parallel_load_balancer`。默认使用逐文档连续切分。设置 `DocumentCPLoadBalancer.Config(head_tail=True)` 后，每个 rank 持有每篇文档的首尾配对片段。该配置来自 `torchtitan_npu.override.deepseek_v4.cp_metadata`。

| 布局 | CP rank `r` 持有的 token | 每篇文档的对齐倍数 |
| --- | --- | --- |
| 连续切分 | 第 `r` 个片段 | `CP × lcm(compress_ratios)` |
| HeadTail | 第 `r` 个片段及其镜像片段 | `2 × CP × lcm(compress_ratios)` |

文本配方根据布局和压缩比设置 `AlignedTextProcessor.Config.per_doc_alignment`。手动设置该字段时，取表中对应值的整数倍。融合 override 和模型版本需要匹配。参考、量化和多模态配方使用 CP=1。

| 1M 文本配方 | 默认 CP | 默认每篇文档对齐 |
| --- | --- | --- |
| `deepseek_v4_flash_cpt_1024k_a5` | 128 | 16,384 |
| `deepseek_v4_1_flash_cpt_1m_a5` | 128 | 256 |

V4.1 的 1M 配方使用 `FullActivationRematConfig`。配方选择和启动参数见 [Python 配置指南](../developer_guides/python_recipes.md)。

## 切分与通信

`DeepSeekContextParallelTransform` 在构造训练配置时替换 `Compressor`、`inner_attention` 和 `inner_indexer`。`shard_tensors` 按同一置换切分输入、标签、位置和 padding mask。V4 的 MTP 深度输入和 V4.1 的 Engram 前序 token 使用同一置换。

- 滑窗 KV 通过 all-to-all 取回本 rank 缺少的历史 token。
- 各 `Compressor` 通过 all-to-all 恢复自己的连续输入区间。V4 的 ratio-4 压缩器额外交换前一个四 token 块用于重叠池化，并丢弃该块对应的压缩输出。
- 压缩 KV 和 indexer key 通过 `cp_all_gather_tensors` 收集。默认使用 FP32 执行反向的 reduce-scatter。
- AscendC 后端直接根据本 rank 的 `q`、`swa_k` 和 `cmp_k` frame 构建 AI_CPU 记录。arch22 使用 `A3KeyWorkaround` 压实 key。

## 双卡集成用例

`deepseek_cp` 套件包含 V4 和 V4.1 的 CP=2 连续切分用例，各执行两步。`check_loss=False` 表示用例检查训练流程能否完成。HeadTail 的置换和 frame 由 CPU 单测覆盖。

```bash
python -m tests.integration_tests.run_tests ./test_reports/deepseek-cp \
  --test_suite deepseek_cp --ngpu 2 --no-parallel
```

精度验证需要固定初始权重、输入和全局有效 token 计数。比较方法见 [数值对齐指南](../developer_guides/numerical_alignment.md)。
