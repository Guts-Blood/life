# Day 22 — Qwen3.5/3.6-35B-A3B MoE 容量规划

日期：`2026-08-17`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

以 `Qwen/Qwen3.5-35B-A3B-Base` 为可训练目标，完成显存、active compute、EP 拓扑与监控方案；再对照 Qwen3.6-35B-A3B，并与 Day 19 的 32B dense 比较。

## 理论（75 分钟）

精读清单：[Day 22 — 35B MoE capacity 完整推演](../SCALING-BOOK-READING-GUIDE.md#day-22)。依次完成 Part 12 EP、PP、真实 topology、TLDR，并把 Quiz 5 Q2 改写成 Qwen 题。

- Router、top-k、expert capacity、load-balance loss。
- EP/ETP/TP/DP 组合与 all-to-all。
- Straggler、token imbalance、expert collapse。
- Base vs post-trained checkpoint；text-only SFT 是否冻结/加载 vision encoder；hybrid attention/DeltaNet 对传统 Transformer 估算的修正。

## Coding（120 分钟）

- 为 `capacity_plan.py` 加 total/active parameters、num experts、top-k、EP degree。
- 画 token dispatch -> expert compute -> combine 的通信图。
- 输出 per-expert token/load、drop/overflow、router entropy 等监控清单。
- 从两个 model config 自动导出 component/parameter 表，明确哪些公式仍适用、哪些必须用 pilot 实测。

## 训练 / 实验（60–75 分钟设计实验）

- 不实际训练 30B MoE。
- 设计 `EP=8` 起点，比较 8/16/32/64 GPU 下 TP/PP/DP 组合。
- 画 `EP=8, TP=2, PP=2, DP=2` 的 64 GPU groups。
- 对 Dense 32B 与 MoE 35B-A3B 写 memory/compute/communication 对照。

## 资源与租卡

- CPU only；不租 30B MoE 集群。
- 使用 Day 18/19 的 collective 和 profiler 数据作为实际输入。

## 验收

- [ ] 能解释 MoE 为什么 FLOPs 更低但系统不一定更简单/更快。
- [ ] 权重显存、active compute、all-to-all 三者分开估算。
- [ ] 能解释为什么 post-training 设计选择 Qwen3.5 Base，而 Qwen3.6 用作架构/能力参照。
- [ ] 至少五项 MoE 专属监控。
- [ ] 输出 `day22-qwen35b-a3b-capacity.md` 与 topology 图。

## Daily Log

### Dense vs MoE

### 推荐 topology

### Day 23 第一动作
