# Day 06 — 周末 Reading：Tülu 3 × Applied Training

日期：`2026-08-01`

状态：`not_started`

强度：1 小时，仅阅读

## 主要目标

把“post-training 阶段为什么存在”与“训练是否放得下、跑得完、值得跑”放进同一张图。周末只做对读与判断，不新增代码或 GPU 任务。

## 理论（60 分钟）

精读清单：[Day 06 — Tülu 3 与 Applied Training 对读](../SCALING-BOOK-READING-GUIDE.md#day-06)。

- 25 分钟：读 [Tülu 3](https://arxiv.org/abs/2411.15124) 的 pipeline overview、SFT/DPO/RLVR 阶段和 evaluation/decontamination 总览。
- 25 分钟：读 [Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training/) 中参数/FLOPs、minimum memory、training time 的代表性例题；先写判断再展开答案。
- 10 分钟：完成一张 crosswalk：

| 决策 | Tülu 3 提供的依据 | Applied Training 提供的约束 |
|---|---|---|
| 是否需要该训练阶段 | 数据、目标行为、前序 checkpoint | memory/compute/time/cost feasibility |
| 是否扩大规模 | dev eval 与 failure slice | token budget、硬件与并行效率 |
| 是否停止 | 收益、退化、污染风险 | 预算与 wall-clock |

必须回答：

1. SFT、preference tuning、RLVR 各自需要什么数据和上游 checkpoint？
2. memory-feasible、compute-feasible、time/cost-feasible 为什么是三个判断？
3. Scale Your Model 的成本结论如何约束 post-training recipe，而不替代数据与评测决策？

## Coding

无。周末不打开 IDE，不补工作日任务。

## 训练 / 实验

无。

## 资源与租卡

CPU/手机阅读即可；不要启动 AutoDL GPU。

## 验收

- [ ] 写出 3 个可复述结论，每条都同时连接“训练阶段”与“规模约束”。
- [ ] 标出一个可迁移到 Qwen/H100 的方法，以及一个不能直接迁移的 TPU 数字。
- [ ] 留下一个仍不确定、可在 Week 2 用证据回答的问题。

### 三个结论

-
-
-

### 未决问题
