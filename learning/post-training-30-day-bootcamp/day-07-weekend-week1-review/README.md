# Day 07 — 周末 Reading：Week 1 Review

日期：`2026-08-02`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

压缩 Week 1 的知识，确保下周开 GPU 前没有概念性阻塞。

## 理论（60 分钟）

精读清单：[Day 07 — 为真实模型选 sharding](../SCALING-BOOK-READING-GUIDE.md#day-07)。Core 是 Part 6 的 pure FSDP、FSDP+sequence、FSDP+TP 三次判断，并迁移到 Qwen3-32B。

- 40 分钟：继续 [Training LLaMA 3 on TPUs](https://jax-ml.github.io/scaling-book/applied-training/) 的 `How to shard LLaMA 3-70B for training`；依次判断 pure FSDP、FSDP+sequence、FSDP+TP，先答再展开原文。
- 10 分钟：`Worked Problems` Question 1，只列已知量、公式与 topology，不追完整数值。
- 10 分钟：把 LLaMA config 替换为 Qwen3-32B，写出哪些公式可以迁移、哪些 TPU 数字必须换成 H100 实测。

## Coding

无。

## 训练 / 实验

无；不要为了“预热”开卡。

## 资源与租卡

CPU only。检查 Day 08 的 AutoDL 实例和存储是否可用，但不启动。

## Week 1 Gate

- [ ] 能逐项解释 parameter、gradient、optimizer、activation。
- [ ] 能画出 DP/FSDP/TP/PP/CP/EP 的切分对象。
- [ ] Day 08 的模型、数据、命令、成功条件已经明确。

### 最大阻塞
