# Day 04 — 分布式并行：框架素养与故障定位

日期：`2026-07-30`

状态：`in_progress`

强度：工作日 4–5 小时

## 主要目标

把 sharding 学到“会配置、会解释、会初步排障”的深度：能说明 DP、FSDP/ZeRO、TP、PP 切了什么、每个 rank 持有什么状态、为何出现对应 collective，以及 OOM、hang、checkpoint 不兼容时先查哪里。今天不追求设计新并行算法或推导最优网络拓扑。

## 理论（90 分钟）

精读清单：[Day 04 — Sharding 对象、状态与诊断](../SCALING-BOOK-READING-GUIDE.md#day-04)。

- 保留《How To Scale Your Model》Part 3：sharding notation、matmul Case 1–4、AllGather、AllReduce、ReduceScatter、AllToAll。
- 少量预读 Part 5：DP、FSDP、TP、PP 各自解决的显存或计算问题。
- 对每种策略只回答五件事：切分对象、复制对象、rank-local 状态、collective、常见失败。
- 掌握 global batch 关系：

  \[
  B_{\text{global}}=B_{\text{micro}}\times N_{\text{grad-accum}}\times N_{\text{DP}}
  \]

  TP/PP degree 不重复计入 global batch。

## Coding / 配置阅读（90 分钟）

- 从一份 PyTorch FSDP/DeepSpeed 配置和一份 Megatron 配置中，标出 DP/TP/PP degree、micro batch、gradient accumulation、checkpoint 格式。
- 填写 `DP/FSDP/TP/PP -> parameter/gradient/optimizer/activation -> replicated/sharded` 状态表。
- 画一个 `TP=2, PP=2, DP=2` 的 8-rank group 图；CP/SP/EP 只放 Stretch。
- 写三张故障卡：单卡 OOM、多卡 hang、并行度变化后的 checkpoint load failure；每张卡列“首查指标 -> 可能原因 -> 下一步验证”。

## 训练 / 实验（45–60 分钟）

- 不做正式训练、不租 GPU。
- 对两个假想配置做纸面诊断：
  1. global batch 意外翻倍；
  2. FSDP 保存的分片 checkpoint 改用不同 world size 恢复。
- 可选 CPU dry run：验证配置解析出的 world size 与各 parallel group size，不实现 collective。

## Quiz：对象、状态、诊断

互动测验使用 [`DAY-04-READING-QUIZ.md`](./DAY-04-READING-QUIZ.md) 跟踪原始回答、点评和修正版结论。

- [x] Day 04 Reading Quiz：`completed`。

1. **对象/数据题**：FSDP 与 TP 分别切哪个对象？parameter、gradient、optimizer state、activation 在每个 rank 上是什么布局？
2. **状态/训练题**：`A[I,J_X]B[J_X,K]` 为什么产生 partial sums？AllReduce 与 ReduceScatter 分别把它变成什么输出状态？
3. **诊断/判断题**：分别面对 FSDP 后仍 OOM、部分 rank hang、改变 world size/TP degree 后 checkpoint 无法加载，第一轮要核对哪些 shape、buffer、rank 日志、collective 顺序、metadata 与 reshard 条件？

## 资源与租卡

- [Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/)
- [Training at Scale](https://jax-ml.github.io/scaling-book/training/)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- CPU only；今天不为 collective 演示租卡。

## 产物

- [`../artifacts/reports/day-roadmaps/day-04-parallelism-roadmap.svg`](../artifacts/reports/day-roadmaps/day-04-parallelism-roadmap.svg)
- `../artifacts/reports/day04-parallel-state-map.md`
- `../artifacts/reports/day04-rank-groups.mmd`
- `../artifacts/reports/day04-distributed-failure-cards.md`

## 验收

- [ ] 给定 DP/FSDP/TP/PP 配置，能算 world size、global batch 和 group size。
- [ ] 能从 sharding 形状解释四种 collective 的因果，而不是只背名称。
- [ ] 能说清每种策略对四类训练状态的影响。
- [ ] 面对 OOM、hang、checkpoint mismatch，各能给出有顺序的首轮排查。
- [ ] CP/SP/EP 未掌握不阻塞本周；记录为后续选学，不冒充 Core。

## Daily Log

### 今天最容易混淆的对象/状态

### 一个诊断题及证据

### Day 05 第一动作
