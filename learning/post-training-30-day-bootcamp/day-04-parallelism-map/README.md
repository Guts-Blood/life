# Day 04 — 分布式并行地图

日期：`2026-07-30`  
状态：`not_started`  
强度：工作日 4–5 小时

## 今日结果

建立 DP、FSDP/ZeRO、TP、PP、CP、SP、EP 的统一心智模型和 rank-group 图。

## 资料

- [How to Parallelize a Transformer for Training](https://jax-ml.github.io/scaling-book/training/)
- [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)

## 时间安排

- 90 分钟：读 Scaling Book training 前半。
- 60 分钟：读 Megatron 并行策略概览。
- 90 分钟：画 8/16/64 GPU rank groups。
- 45 分钟：填写通信/显存/适用条件对照表。
- 15 分钟：口头讲解并记录不顺的地方。

## 理论

精读清单：[Day 04 — Sharding notation 与 collective 因果链](../SCALING-BOOK-READING-GUIDE.md#day-04)。逐个完成 Part 3 Case 1–4，并为每个 Case 写 global/local shape、collective 与通信 bytes。

- DP/FSDP/TP/PP/CP/SP/EP 的状态切分、process groups、collectives 与通信代价。

## Coding

- 用 Mermaid 或小脚本生成 16/64 GPU rank-group 图。
- 可选：用 `torch.distributed` 写两个 rank 的 all-reduce/all-gather 最小程序，但不要求今天租卡运行。

## 训练 / 实验

- 无正式训练；这是 Day 15 多卡实验的设计日。
- 预注册 Day 18 Megatron 双卡 TP/DP 实验需要测的 memory、tokens/s、rank groups 和 collective。

## 资源与租卡

- CPU only；不要租 GPU。
- 多卡租用集中到 Day 15–16，避免边读理论边计费。

## Core

- [ ] 对每种并行回答：切什么、复制什么、通信什么、何时使用、主要失败模式。
- [ ] 画 `TP=4, PP=2, DP=2` 的 16 GPU rank groups。
- [ ] 画 `TP=2, PP=2, EP=8, DP=2` 的 MoE 拓扑草图。
- [ ] 解释 TP 为什么通常限制在高速互联域内。
- [ ] 解释 PP bubble 与 micro-batch 数量的关系。
- [ ] 解释 CP 与普通 sequence parallel 的区别。

## 必须完成的表

| 策略 | 切分维度 | 每卡持有什么 | 主要 collective | 显存收益 | 性能风险 |
|---|---|---|---|---|---|
| DP | | | | | |
| FSDP | | | | | |
| TP | | | | | |
| PP | | | | | |
| CP | | | | | |
| EP | | | | | |

## 产物

- `../artifacts/reports/parallelism-map.md`
- `../artifacts/reports/rank-groups-16gpu.mmd`
- `../artifacts/reports/rank-groups-64gpu-moe.mmd`

## Definition of Done

- 随机给一个并行配置，能够计算总 GPU 数与每类 group size。
- 能说明增加某一种 parallel degree 后，哪些显存下降、哪些通信上升。
- 不把 ZeRO/FSDP 与 TP 混成同一件事。

## Daily Log

### 最难理解的 group

### 口头讲解卡住在哪里

### Day 05 第一动作
