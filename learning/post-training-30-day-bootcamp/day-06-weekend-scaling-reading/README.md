# Day 06 — 周末 Reading：Applied Scaling

日期：`2026-08-01`  
状态：`not_started`  
强度：1 小时，仅阅读

## 主要目标

把前五天的公式和并行概念放进一个真实的大模型训练案例，不新增工程任务。

## 理论（60 分钟）

精读清单：[Day 06 — 真实模型的参数、时间与成本](../SCALING-BOOK-READING-GUIDE.md#day-06)。按问题逐项手算，展开原文答案前先写自己的结果。

- 10 分钟：读 [Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training/) 的 `What does LLaMA 3 look like?`。
- 35 分钟：读 `Counting parameters and FLOPs`，每个隐藏答案展开前先手算。
- 15 分钟：完成 training time/minimum-memory 问题，并只记录以下三个答案：
  1. Memory-feasible 为什么不等于时间/成本可接受？
  2. Compute-feasible 为什么仍可能不值得租卡？
  3. 哪个推理方法可以迁移到 H100/Qwen，而哪些 TPU 数字不能直接迁移？

## Coding

无。周末不打开 IDE，不补工作日欠下的实现。

## 训练 / 实验

无。

## 资源与租卡

CPU/手机阅读即可；不要开 AutoDL GPU 实例。

## Tracking

- [ ] 完成指定阅读。
- [ ] 在下方写 3 个 bullet，不写长总结。

### 三个 bullet

- 
- 
- 
