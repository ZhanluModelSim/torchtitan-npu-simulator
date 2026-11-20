# Qwen3.5 系列模型

Qwen3.5 系列包括 Qwen3.5、Qwen3.6 和 Qwen3.8。同架构的官方权重使用同一套 SFT 入口。训练从各自的 HF 目录读取权重、tokenizer 和 chat template。以下用 27B 配方演示。

## 1. 安装训练环境

完成[基础安装](../../../docs/user-guides/installation.md)并加载 CANN 环境。在仓库根目录执行：

```bash
python -m pip install 'jinja2>=3.1' 'transformers>=5.13,<6'
export CC="$ASCEND_HOME_PATH/tools/ccec_compiler/bin/bisheng" CCC_OVERRIDE_OPTIONS="# +-std=c++20"
```

默认训练使用原生 Triton-Ascend GDN 和 CANN varlen attention。需要 FLA 时，按 [master 的安装指南](https://gitcode.com/cann/torchtitan-npu/blob/3734a27ed73012a0b190d394f23067147fc3dc07/torchtitan_npu/models/qwen3_5/README.md)在独立环境安装配套版本：

```bash
python -m pip install --only-binary=:all: \
  'flash-linear-attention==0.6.0' 'fla-core==0.6.0' \
  'triton==3.6.0' 'triton-ascend==3.6.0.post2' 'transformers==5.22.1' 'tokenizers==0.23.2' \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

## 2. 快速上手

样本取自 Stanford Alpaca 2023 发布版的前 16 条。转换时将 `instruction` 和非空的 `input` 合并为 user 消息，将 `output` 写为 assistant 消息。该样本按 CC BY-NC 4.0 提供。[JSONL 样本](../../../tests/assets/qwen3_5/alpaca.jsonl) · [Stanford Alpaca 原始数据](https://github.com/tatsu-lab/stanford_alpaca/blob/f13496211289def0ff88ae673389ae14a818b4b3/alpaca_data.json) · [数据许可](https://creativecommons.org/licenses/by-nc/4.0/)

在仓库根目录执行下面的命令，将模型路径换成官方 HF 权重所在目录，并用 `--output-dir` 指定本次短测的新输出目录。`--smoke-test` 执行一个优化步。

```bash
bash examples/qwen3_5/run_qwen_sft.sh \
  --hf-assets-path /models/Qwen3.5-27B \
  --dataset-path tests/assets/qwen3_5/alpaca.jsonl \
  --output-dir ./outputs/qwen-alpaca-smoke \
  --smoke-test
```

## 3. 启动训练并选择 loss mask

```bash
bash examples/qwen3_5/run_qwen_sft.sh \
  --hf-assets-path /models/Qwen3.5-27B \
  --dataset-path /data/train.jsonl \
  --output-dir /outputs/qwen-sft \
  --loss-mask ignore_empty_think
```

`ignore_empty_think` 可以直接配置，是 `default+ignore_empty_think` 的简写，也是省略 `--loss-mask` 时的行为。空 thinking 指模板生成的 `<think>\n\n</think>\n\n` 一类只有空白的块。处理器将屏蔽位置的标签设为 `-100`，对应内容作为输入上下文。

- `--loss-mask ignore_empty_think` 选择所有有监督的 assistant 轮次，屏蔽空 thinking，监督真实推理、工具调用、答复及对应的 EOS。
- `--loss-mask last_round+ignore_empty_think` 把范围收至最后一条 user 后的 assistant 轮次。消息上的 `"loss": true` 或 `false` 可以覆盖轮次选择。
- `--loss-mask ignore_think_prefix` 只屏蔽 thinking 开始标签及其紧随的换行，推理正文与结束标签参与 loss。
- `--loss-mask ignore_think` 将整个 thinking 块作为上下文，监督工具调用、答复和 EOS。也可使用 `last_round+ignore_think`。

下例包含多轮对话、推理和工具调用。用 `"loss": false` 将第一轮 assistant 消息保留为上下文。工具回执通过 `tool_call_id` 对应 `tool_calls` 中的调用。选择 `ignore_think` 时，`reasoning_content` 作为上下文，两次工具调用和最终答复参与 loss。

```jsonl
{"tools":[{"type":"function","function":{"name":"get_temperature","description":"查询城市示例气温","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"messages":[{"role":"user","content":"报气温用什么单位？"},{"role":"assistant","content":"用摄氏度。","loss":false},{"role":"user","content":"查询上海和深圳的示例气温。"},{"role":"assistant","content":"","reasoning_content":"两个城市可以并行查询。","tool_calls":[{"id":"sh","type":"function","function":{"name":"get_temperature","arguments":{"city":"上海"}}},{"id":"sz","type":"function","function":{"name":"get_temperature","arguments":{"city":"深圳"}}}]},{"role":"tool","content":"上海测站返回 21 ℃","tool_call_id":"sh"},{"role":"tool","content":"深圳测站返回 29 ℃","tool_call_id":"sz"},{"role":"assistant","content":"上海 21 ℃，深圳 29 ℃。"}]}
```

启用已安装的 FLA 时，追加 `--override torchtitan_npu.override.qwen3_5.fla.operators`。使用长上下文配方时，运行 `bash examples/qwen3_5/run_qwen_sft.sh --help` 查看配置列表。

## 4. 定制模板、短测和导出

自定义模板用 `--chat-template /data/team.jinja --loss-mask template`，在要监督的内容周围加 `{% generation %}…{% endgeneration %}`。只训练推理时标记非空推理正文，混训时给无推理的样本保留答复监督。使用模型自带 generation 标记时，移动边界即可。团队自定义工具格式应配套推理模板和解析器。

已验证的 Qwen3.5 模板渲染最后一条 user 后的推理。需要较早轮次的推理时，复制该模板，将 assistant 分支条件改为 `loop.index0 > ns.last_query_index or reasoning_content`。已验证的 Qwen3.6 和 Qwen3.8 模板支持 `preserve_thinking`，训练用它保留历史推理。

- 续训沿用原输出目录、训练配置、数据、模板和运行环境，使用 `--resume-step N` 指定步数。`--max-steps M` 截短时保存完整 DCP，目标步数必须大于已恢复步数。旧版本游标请使用原提交续训。
- 自定义 Python 配方用 `--module /jobs/config_registry.py --config train`。可导入的模型配方也能通过 `scripts/run_train.sh --module torchtitan_npu_recipes.models.qwen3_5 --config run_qwen3_5_4k` 启动。
- 正常训练结束时，配方可用 `checkpointer.last_save_in_hf = True`、`checkpointer.last_save_model_only = True` 和 `checkpointer.export_dtype = "bfloat16"` 导出。保留初始化所用的 HF 权重目录，续训后的导出也从该目录补齐未训练的辅助权重。推理目录使用同一步的 safetensors 及其索引，配上同源模型配置、tokenizer 和处理器文件。
