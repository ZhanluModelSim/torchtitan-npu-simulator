# GraphTrainer EP overlap 适配

EP overlap 是 torchtitan 上游 `experiments/graph_trainer/` 为 MoE + Expert Parallel（EP）训练提供的通用 graph pass：把 MoE 区域按 token 维切成两个 chunk，使一个 chunk 的 all-to-all 通信与另一个 chunk 的专家计算重叠。该能力的配置接口、pass pipeline 和本文的临时补丁均不绑定具体模型名称。本文描述启用该能力所需的四个 `patches/torchtitan/` 临时补丁；这些问题最先由 DeepSeek-V4（DSV4）验证图暴露，当前整网验证也使用 DSV4。

| 序号 | pass名称 | 内容 |
| --- | --- | --- |
| 1 | mark_unbacked（前置操作，不属于pass） | 在 trace 前制造 chunk 动态维度及 hint |
| 2 | populate_chunk_dim_metadata_pass | trace 后识别该符号，并登记“这是 EP chunk symbol” |
| 3 | ep_overlap_chunk_pass | 利用 EP chunk symbol 符号切图 |
| 4 | ep_overlap_schedule_pass | 根据两个 chunk 重排通信和计算 |
| 5 | concretize_ep_chunk_symbolic_shapes_pass | 调度结束后，使用筛选后的 hint 具体化相关符号 |

EP overlap 默认不启用。它只在 GraphTrainer 编译路径（`graph_trainer_*` 配置）下生效，普通 eager 训练不走这些 graph pass；下文配置项中的 `strategy=eager` 指 GraphTrainer 的另一种 chunk 实现方式。DSV4 验证样例所依赖的模型侧 GraphTrainer 适配见 [`deepseek_v4_graph_trainer.md`](deepseek_v4_graph_trainer.md)。

## 能力范围、依赖与验证边界

| 项 | 值 |
| --- | --- |
| 能力对象 | 使用 GraphTrainer、MoE 和真实 EP token exchange 的训练图；EP degree > 1 |
| 当前整网验证模型 | DeepSeek-V4，使用 `graph_trainer_deepseek_v4_flash_43layers_16experts` 配置 |
| 训练路径 | GraphTrainer `aot_fx_trace`；eager 路径不涉及 |
| torchtitan 基线 | `0.3.0`（`requirements.txt` 固定版本） |
| 环境 | Ascend NPU + `torch-npu` + CANN + HCCL；EP degree > 1 |
| 默认状态 | 关闭，需显式传 `--compile.ep_overlap.enabled` |

## 实现原理

### 四个上游临时补丁

补丁位于 `torchtitan_npu/patches/torchtitan/experiments/graph_trainer/`，针对 `requirements.txt` 固定的 torchtitan 0.3.0，在 `import torchtitan_npu` 时经 `patches/torchtitan/__init__.py` 自动 `apply()`。它们修正的是上游 graph pass 的行为，代码不依赖 NPU 专有算子，也不检查模型类型。DSV4 验证图只是这些问题的首个整网复现场景；任何产生相同符号关系、节点标注或调度依赖的 GraphTrainer MoE 图都会进入相同处理逻辑。

四个 patch 共替换或包装五个上游函数，均有标记防止重复 `apply()`。它们的执行顺序是：第四个作用于 chunk pass，第二、第三个作用于 overlap schedule pass，第一个作用于调度后的 concretization pass。

| 补丁模块 | 替换或包装的上游符号 | 当前修复方案 |
| --- | --- | --- |
| `ep_chunk_concretization` | `ep_pass_utils._placeholder_symbol_hints`、`_concretize_value` | 筛选含 chunk symbol 的 placeholder 维度；与传入 hint 符号无交集的值保持原样 |
| `ep_overlap_shape_queries` | `ep_overlap_pass._collect_token_exchanges` | 清理 body 内 AllToAll launch 的 size query 误继承的 exchange 标注，再调用上游收集器 |
| `ep_ready_nodes_dedup` | `ep_overlap_pass._ready_nodes` | 保留候选遍历和依赖检查，用本轮 `selected` 集合跨 chunk 去重 |
| `ep_shape_live_out` | `ep_chunk_pass._split_live_out_users` | 将同一 root 内直接查询 AllToAll dim-0 的 full user 移到 chunked |

#### 1. ep_chunk_concretization.py

报错示例：

```text
ValueError: Chunk pass could not derive a base hint for symbol u0 from
arg2594_1.shape[0] extent Max(0, u0 - Min(1, u0)) + 1.
```

FX 图中既有 EP chunk symbol，也有独立的数据相关 metadata symbol。社区 `ep_pass_utils.py` 中的 `_placeholder_symbol_hints` 遍历所有 tensor placeholder 的 shape 维度，对每个维度调用 `_record_symbols_from_extent`。因此，与 chunk 无关的非线性 metadata 维度也会进入 hint 推导，可能因无法推导 base hint 而报错。

社区具体化流程先合并 placeholder hint 和 chunk hint，再调用 `_concretize_value`。原 `_concretize_value` 还可能通过值自身的 hint 把不属于目标范围的动态值折叠成常量。

**Patch 方案：**

| 社区函数 | 当前实现 |
| --- | --- |
| `_placeholder_symbol_hints` | 替换为 `_chunk_placeholder_symbol_hints`：取得 `chunk_symbol_hints_for_mode(gm).keys()`，只对自由符号与它有交集的 placeholder shape 维度调用上游 `_record_symbols_from_extent` |
| `_concretize_value` | 包装原函数：值的自由符号与传入 `symbol_hints.keys()` 无交集时原样返回，否则委托上游原函数具体化 |

placeholder hint 收集逻辑：

```python
def _chunk_placeholder_symbol_hints(gm: fx.GraphModule) -> dict[object, int]:
    chunk_symbols = ep_pass_utils.chunk_symbol_hints_for_mode(gm).keys()
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder" or (val := ep_pass_utils.tensor_meta(node)) is None:
            continue
        for dim, extent in enumerate(val.shape):
            symbols = ep_pass_utils.free_symbols(extent)
            if not symbols & chunk_symbols:
                continue
            ep_pass_utils._record_symbols_from_extent(hints, extent, source=f"{node.name}.shape[{dim}]")
    return hints
```

值具体化的包装逻辑：

```python
def _concretize_chunk_value(value: object, symbol_hints: dict[object, int]) -> object:
    if not ep_pass_utils.free_symbols(value) & symbol_hints.keys():
        return value
    return original_concretize_value(value, symbol_hints)
```

这里有两个不同的筛选集合：placeholder 维度按 **EP chunk symbol** 筛选；值具体化按调用方传入的 **合并后 hint 字典**筛选。若选中的维度包含其他符号且上游能推导其 hint，这些符号也可能进入具体化范围，不能将第二层条件表述为“值必须直接含有 EP chunk symbol”。

当前实现不扫描全图的 shape、stride、storage offset，也不计算符号依赖连通分量。选中维度的 hint 推导仍使用上游规则：缺少 hint、无法推导 base hint、hint 冲突都会继续报错，不会被忽略。`apply()` 检查六个所需上游辅助符号，缺失时记录 warning 并跳过此 patch。

#### 2. ep_overlap_shape_queries.py

报错示例：

```text
ValueError: ep_overlap found EP token-exchange metadata on non-marker
node sym_size_int (aten.sym_size.int). Only all_to_all launches
and their wait_tensor nodes may carry this annotation.
```

错误发生在 `ep_overlap_schedule_pass` 的通信节点收集阶段。AllToAll 输出的接收 token 数是数据相关的，trace 会生成 `aten.sym_size.int(all_to_all_output, 0)` 来绑定其首维长度。该节点在通信标注的作用域内生成，可能继承 `EP_token_exchange=dispatch`。社区 `_collect_token_exchanges` 只接受带此标注的 token-exchange launch 及其 wait，遇到 size query 就报错。

**Patch 方案：**在原 `_collect_token_exchanges` 前遍历当前 `body.nodes`，仅处理同时满足以下条件的节点：

- `op == "call_function"`、target 为 `aten.sym_size.int`，且恰有两个位置参数。
- 第一个参数是当前 body 内的 FX 节点，并且上游 `_is_token_exchange_launch` 将其识别为 token-exchange launch。

包装函数的主体如下；只在 `custom` 中存在 `EP_token_exchange` 时复制字典并删除这个键：

```python
node_set = set(body.nodes)
for node in body.nodes:
    if not _is_token_exchange_shape_query(node, node_set):
        continue
    custom = ep_overlap_pass._custom_meta(node)
    if _EP_TOKEN_EXCHANGE in custom:
        custom = dict(custom)
        del custom[_EP_TOKEN_EXCHANGE]
        node.meta["custom"] = custom
return current(body, order=order)
```

其余 metadata 保留；删除后即使 `custom` 为空也保留该容器；没有该标签时不改写 metadata。匹配条件不限制查询维度为 0，也不显式检查查询值是否 unbacked。launch/wait 的识别、唯一 wait 检查、顺序检查和 wait 标注归一化仍由上游完成，其他非目标节点的错误继续抛出。

当前正式 patch 仍采用“删除 shape query 继承的标签”方案，没有改成“保留标签、仅绕过报错”。

#### 3. ep_ready_nodes_dedup.py

报错示例及调用关系：

```text
AssertionError: stable topological sort failed

ep_overlap_schedule_pass()
  -> _schedule_ep_overlap_regions()
     -> _plan_region()
        -> _build_region_phases()
           -> _append_ready_blocks()
              -> _ready_nodes()       # 可能返回重复 node
     -> _apply_schedule()
        -> _phase_order_deps()         # 顺序约束可能构成环
        -> _stable_topological_sort()
```

节点有确定的 chunk 所有权，但两个 chunk 的调度候选集合可能有交集。例如，节点 N 属于 chunk0，同时又是 chunk1 后续 AllToAll 的依赖，它可能同时出现在两个 chunk 的候选集中。

社区 `_ready_nodes` 对每个 chunk 仅排除此前已经 `emitted` 的节点；`_append_ready_blocks` 要等整个 ready 元组生成后才更新 `emitted`，因此无法排除本轮内部的重复项。若返回结果为 `[N, A, N]`，后续顺序依赖会形成 `N -> A -> N`。

**Patch 方案：**当前提交保留 `_ready_nodes` 的遍历实现，在一次调用内新增 `selected` 集合；不修改传入的 `emitted`。替换函数的主体如下：

```python
ready = []
selected: set[fx.Node] = set()
for chunk_id in chunk_order:
    body = region.bodies_by_chunk[chunk_id]
    candidates = sorted(candidates_by_chunk.get(chunk_id, set()) - emitted, key=order.__getitem__)
    for node in candidates:
        if node in selected:
            continue
        if not include_waits and ep_overlap_pass._is_c10d_functional_node(node):
            continue
        deps = ep_overlap_pass._body_deps(node, body=body, owner_by_node=owner_by_node)
        if all(dep in emitted for dep in deps):
            ready.append(node)
            selected.add(node)
return tuple(ready)
```

chunk 遍历顺序、chunk 内候选排序、`include_waits` 过滤以及依赖必须已经 `emitted` 的条件都保留；只有实际追加到 ready 的节点才记入 `selected`。每个节点在一次返回中最多出现一次，保留首次被选中的位置。当前代码没有调用原 `_ready_nodes` 后再对返回值去重。

#### 4. ep_shape_live_out.py

报错示例：

```text
ValueError: Chunk pass cannot materialize live-out without chunk dimension;
all_to_all_single_default from 'layers.0.moe' has full users
['sym_size_int:sym_size.int:fqn=layers.0.moe.dispatcher']
and requires a forward accumulation proof or a backward parameter-gradient consumer.
```

live-out 是 chunk body 内产生、但仍被 body 外节点使用的值。社区 `_split_live_out_users` 将消费者分成 `chunked`（跟随对应 chunk 的值）和 `full`（需要重建完整值）。某些 dispatcher size query 带有 MoE 模块 FQN，却位于 body 外；它们需要查询各 chunk 自己接收的 token 数，但上游分类将其归入 `full`。

社区 `_materialize_live_out` 只能在识别到 chunk 维时沿该维 `cat`，或在有加法重建依据时 `add`。AllToAll 输出首维是独立的接收 token symbol，通常不含选定的 chunk symbol，因而无法识别拼接维度；正向也没有适用的加法依据，最终报错。修复此场景的关键是调整消费者归属，使它无需完整值重建。

**Patch 方案：**只包装 `_split_live_out_users`。先调用社区原函数，再从 `full` 中筛选：

- user 是 `call_function` 节点，target 为 `aten.sym_size.int`，恰有两个位置参数，第二个参数为 `0`。
- 第一个参数直接指向 `call_function` 类型的 `_c10d_functional.all_to_all_single.default` 节点。
- user 的模块 FQN 位于 `producer_region.root_fqn` 内。

```python
chunked, full = current(users, plans, producer_region, symbol_hints)
moved = tuple(
    user
    for user in full
    if _is_dim0_all_to_all_size_user(user)
    and ep_chunk_pass.is_module_fqn_inside_root(
        ep_chunk_pass._get_module_fqn(user), producer_region.root_fqn
    )
)
if not moved:
    return chunked, full
moved_set = set(moved)
return (*chunked, *moved), tuple(user for user in full if user not in moved_set)
```

| 函数 | 当前行为 |
| --- | --- |
| `_split_live_out_users` | 将满足上述条件的 user 从 full 移到 chunked；没有匹配者时返回原分类结果 |
| `_materialize_live_out` | **保持社区原函数，不再替换，也不新增 AllToAll dim-0 的 cat fallback** |

匹配不依赖 `EP_token_exchange` 标签，也不覆盖 wait 输出、普通张量或其他维度的 size query。没有剩余 full users 时，社区 chunk pass 跳过这个 live-out 的完整值重建。仍有 full users 时继续执行社区重建规则和校验，不能重建的情况仍报错。

第二个 patch 的标签清理发生在后续 schedule pass，晚于这里的 live-out 分类，因此第四个问题不是删除标签引起的。

父模块、其他 root 或缺失 FQN 的 dim-0 query 仍可能被归为 full；若未来要支持这些场景，需要额外定义重建方案，不能将当前补丁的适用范围外推为所有 AllToAll shape live-out 都受支持。

## 启用方式

EP overlap 通过 `--compile.ep_overlap.*` 打开。满足前述条件的 GraphTrainer MoE recipe 均通过同一组配置项启用；模型入口、dispatcher、并行配置和算子 override 仍由各模型 recipe 负责。当前 A3 GraphTrainer debug 脚本已设置 `seq` 分块、`graph` 策略和 NPU auto overlap pipeline，调用它时无需重复传这些配置项。自动调度机制见 [`graph_trainer_auto_overlap.md`](graph_trainer_auto_overlap.md)。

下面给出当前 DSV4 A3 GraphTrainer 入口脚本的调用示例：flash 43layers/16experts、8 卡、EP=8。此脚本默认启用 NPU auto overlap；下文 2026-09-16 的整网验证对应旧提交和固定调度，不是此命令的验证结果。

```bash
NGPU=8 \
HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh
```

`EpOverlapConfig`（`torchtitan/experiments/graph_trainer/configs.py`）字段：

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `enabled` | `bool`，默认 `False` | 是否启用 EP overlap |
| `strategy` | `eager` / `graph`，默认 `graph` | `eager` 在 trace 前包装模块 forward；`graph` trace 原模型后用 FX pass 切分 |
| `chunk_dim` | `batch` / `seq`，默认 `batch` | 切分的逻辑输入维 |
| `module_fqn` | `layers.*` / `layers.*.moe`，默认 `layers.*` | 被切分的模块 FQN 模式 |
| `disable_early_grad_accumulation` | `bool`，默认 `False` | 关闭 graph chunking 的早期参数梯度累加，用于需要与 eager chunking 保持严格顺序的验证场景 |

上游 `validate_ep_overlap_config` 约束：`chunk_dim=seq` 仅在 `module_fqn=layers.*.moe` 时可用——序列切分要求完整 K/V 上下文，因此只对 MoE block 根成立。

本仓 `ep_chunk_concretization` 补丁对 `strategy=graph` 新增运行时约束：所选 `batch` 或 `seq` 输入维长度必须不小于 2 且为偶数。Graph chunking 会将该维固定拆成两个等长 chunk；若长度为 1 或奇数，trace 输入准备阶段会直接抛出 `ValueError`，避免生成不等长 chunk 后再进入符号 shape 推导。`strategy=eager` 不经过该输入标记流程，不受此项补丁校验影响。

若需把 EP overlap 固化为某个配置的默认值，在 `config_registry.py` 的对应工厂中对 `config.compile.ep_overlap` 赋 `EpOverlapConfig(...)`。

## DSV4 整网实跑验证（2026-09-16）

本节只记录 DSV4 作为验证载体的结果，用于证明通用 EP overlap 能力及临时补丁在这一具体组合上可以完成整网训练。

在同一台机器上按“关闭 EP overlap -> 开启 EP overlap”的顺序各运行 10 step。两组均使用 8 张 Ascend 910、CANN 9.2.0、TorchTitan `v0.3.0`（`086bf6c166ec85c1298eb5596fa9bf95f6a2d840`）和本仓库 HEAD `ec98b86809e024e105958b50c7714666c8c65ff2`。模型为 flash 43 layers / 16 experts，EP=8、DP shard=8、TP/CP/PP=1，local/global batch size 为 1/64，sequence length 为 4096，seed 为 42。

| 指标 | EP overlap 关闭 | EP overlap 开启 |
| --- | ---: | ---: |
| 结果 | 10/10 step，退出码 0 | 10/10 step，退出码 0 |
| graph pass | 7 个，43.617 s | 13 个，228.628 s |
| EP 图变换 | 无 | seq chunking 86 个 region；scheduling 86 个 region |
| step 2-10 平均耗时 | 64.975 s | 62.923 s |
| step 2-10 平均 TPS | 505.000 | 521.222 |
| loss（step 1 -> 10） | 12.19569 -> 9.91401 | 12.19564 -> 9.94998 |
| 日志记录的峰值显存 | 52.22 GiB（85.23%） | 59.37 GiB（96.90%） |

该次顺序单跑中，开启组 step 2-10 的平均 TPS 比关闭组高约 3.2%。

开启组的代价也较明显：日志峰值显存增加 7.15 GiB。

## 相关文档

- DSV4 模型侧 GraphTrainer 适配：[`deepseek_v4_graph_trainer.md`](deepseek_v4_graph_trainer.md)
- 临时补丁目录约定：[`torchtitan_npu/patches/torchtitan/README.md`](../../torchtitan_npu/patches/torchtitan/README.md)
- Override 注册表：[`torchtitan_npu/override/README.md`](../../torchtitan_npu/override/README.md)
- 上游 GraphTrainer 与 EP overlap 说明：[`torchtitan/experiments/graph_trainer/README.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/README.md)
