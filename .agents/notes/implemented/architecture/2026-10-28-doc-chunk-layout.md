# Agent Note: DeepSeek-V4 / V4.1 的 CP 布局收敛成「每文档分块」

Status: implemented（V4 与 V4.1 同一布局；!1099 合入后 head-tail 退出计划）
Created: 2026-10-28
Last updated: 2026-10-28
Tracking: 本仓 !1099（取代 !1063 / !1064 / !1066 的「连续 slab」临时决定）；上游 #4116 / #4617
Supersedes: `./2026-10-21-temporary-head-tail-removal.md`（已删除）

## Context

两个栈原先各用一套 rank-local 行布局，各有各的代价：

- **V4.1：query 在层内搬。** rank 持有每篇文档的 chunk `r`，但不是切分时就给出这个行序，而是每层把 query
  all-to-all 到 chunk 行、输出再搬回来（每层两次集合通信）；metadata 因此要同时描述 query 与 key 两套坐标
  （`cu_seq_q` / `cu_seq_k` / `positions_q` / `exchange` 一类的四个位置量）。
- **V4：连续 slab。** rank 持有全局行的一段连续区间，metadata 要拿 slab 与文档边界求交来推 rank-local varlen，
  而这些区间跨文档、跨 chunk，无法用一条规则描述「本 rank 的行」。
- 两者的交界处还有两条硬约束：上游 #4617 的 `prepare_cp_metadata` 只用**整行**的 metadata 加一个可选置换重建
  rank-local 副本；融合内核的 frame 是「本 rank 的查询块 + 整行 key」。任何需要把置换折进 varlen、窗口/块计划、
  交换路由与容器打包的布局（head-tail 如此）都要为每一条推导各写一份。
- Engram 要每行**在本文档内**往前 `H` 个 token。连续 slab 下这些前驱是别处的一段连续区间，一个窗口说不出每行
  各自的前驱。

## Decision

CP 的布局定义为整行的**一个置换**：rank `r` 的窗口是每篇文档的 chunk `r`（rank-major、文档内顺序）。

- 布局以 load balancer 的形式进 `cp_shard`：`DocChunkLoadBalancer`
  （`--parallelism.context-parallel-load-balancer doc_chunk`）由**切分前**的整行 positions 构造，
  `_generate_indices` 自己算置换。切分即重排，零通信；调用方只交整行，不预算置换。
- 留空（`None`）解析成该栈 metadata 声明的 `cp_layout`（两栈都是 `doc_chunk`），且这个解析对**所有**
  `cp_shard` 调用点一致（含 MTP 深度那一次）。pinned 布局（headtail / ptrr / 真 `None`）行为不变，只有边界驱动
  布局要求调用方交 positions。
- 查询从输入到输出都不移动；只有 key 投影读的那份 hidden 由 `RestorePlan` 从 chunk 行恢复到本 rank 的 slab。
  KV 与压缩 KV 始终是整行，由内核边界的声明式 `S(1) -> R` gather 取得。
- 布局只依赖整行的文档边界，所以 rank-local metadata 可以只由边界重建，不需要置换、`docs` 表或 slab∩文档求交。
- Engram 的 n-gram 前驱顺着同一个置换推出：切分前按整行算成 `[B, C, H]`（第 `d-1` 项是本行往前第 `d` 个 token），
  文档起点的行由 entry 的 `positions` 掩掉，零通信；`_shift_tokens` 同时接受 `[B, H]`（连续切分的一个窗口）与
  `[B, L, H]`（按行）两种契约。
- head-tail **不再**作为方向：置换感知的计划推导整体放弃，不再保留「先迁到某种模式、再在它上面加 head-tail」的两步。

## Consequences

- 两栈共用一套 metadata：`cu_seq_q` / `cu_seq_k` / `positions_q` / `exchange` 这类别名一起删掉，改成
  `q` / `swa_k` / `cmp_k` 三个 frame 加 `position_k`（V4.1 另有 `restore`）。原先 V3.2 专用的 varlen-CP
  rank-local metadata（`from_global`）已随该栈的 CP 能力退场删除，也没有第二个消费者。
- CLI 取值 `chunk_r` 改名 `doc_chunk`，示例脚本与数值对齐脚本同步；`parallelize` 侧不再为 CP 准备 batch 维搬移。
- arch22（910）+ CP>1 不再编译：融合 metadata 带 device 行数（key 压实的 `nonzero`），编译区无法 guard。判为
  不可编译时告警并跑 eager，训练照常；代价是 arch22 上开编译跑 CP 的作业会变慢。
- 逐位判据（详见 `docs/feature_guides/context_parallel.md` §7）：同一 degree、改动前后逐位（910 双卡）；
  跨 degree 在 A5 单卡模拟器上逐位，CP=1 与 CP=2 的 logits、step 1 `loss` 相同，挂 Engram 的用例同样如此。
  arch22 上跨 degree 只有 V4.1 逐位，V4 因 ratio-4 carry 与长归约在该 arch 上没有确定性档位、落到不同内核
  路径而不逐位——改动前既有，不是这次引入的。
- 负载均衡：各 rank 的 token 数仍然均分，但文档长度不均时注意力计算在文档边界处可能不均；目前没有吞吐对比。
- 失去 head-tail 之后，将来若有需要「按 rank 交错」的布局，必须重新引入置换感知推导、并重新论证它与「由整行边界
  重建 metadata」的关系。这是有意接受的代价。
- ratio-4 的 carry **仍然**跨 rank 取一块（chunk `r` 的第一块池化要用 chunk `r-1` 的末块），与布局无关：
  `CarryExchange` 的 ring 不因这次收敛而消失。

## Alternatives considered

- **把 query 的 all-to-all 留在层内**（V4.1 原形态）：每层两次集合通信，metadata 要描述 query 与 key 两套坐标，
  且输出还要 unpermute 回 slab。
- **连续 slab**（!1063 的临时状态）：实现最简，但 metadata 要按 slab∩文档边界求交推导，Engram 的按行前驱
  无法用一个窗口表达。
- **head-tail 作为布局**：置换要折进每条 rank-local 推导（varlen、窗口/块计划、交换路由、容器打包），与
  「由整行边界重建 metadata」的目标直接冲突；本次连同「恢复 head-tail」的计划一起放弃。
