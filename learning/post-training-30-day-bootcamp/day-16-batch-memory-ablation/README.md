# Day 16 — Batch、Accumulation 与显存优化

日期：`2026-08-11`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

在一天内回答两个工程问题：怎样选择 micro-batch/accumulation，以及 checkpointing/Flash Attention/LoRA 分别改变什么。

## 理论（60 分钟）

精读清单：[Day 16 — Activation checkpointing 与 Flash Attention](../SCALING-BOOK-READING-GUIDE.md#day-16)。把 checkpointing、Flash Attention、LoRA 分别标到 model states、activation、FLOPs、HBM traffic。

- `global batch = micro batch × accumulation × world size`。
- Local tokens、arithmetic intensity 和显存边界。
- Activation checkpointing 的 compute-memory tradeoff。
- Flash Attention 的 IO 优化；LoRA 对 model states 与 activation 的不同影响。

## Coding（75 分钟）

- 实现 batch/effective-token calculator。
- 扩展 memory estimator：full/LoRA、checkpointing、measured peak。
- 统一 ablation launcher，避免手工改错配置。

## 训练 / 实验（150 分钟）

Core A，固定 effective batch：

| micro batch | accumulation |
|---:|---:|
| 1 | 16 |
| 2 | 8 |
| 4 | 4 |

Core B，只跑能回答问题的最小对照：

- 1.7B Full：checkpointing off/on。
- 稳定配置：Flash Attention off/on。
- 使用 Day 08 数据对照 4B LoRA 的 memory/throughput。

每组 15–25 steady-state steps；如果时间不足，优先 Core A 和 checkpointing。

## 资源与租卡

- 1×H100 80GB，预计 5–7 小时。
- 同一卡、同数据、同 sequence；OOM 是有效边界，不循环盲试。
- 训练结束后保存表格和 configs，再关机。

## 验收

- [ ] 按收益/代价解释每项优化，而不是罗列开关。
- [ ] 比较 allocated/reserved、tokens/s、step time 和 label tokens/s。
- [ ] 更新 Day 03 Roofline 预测误差。
- [ ] 为 Day 18 多卡选定唯一稳定配置。

## Daily Log

### 最优单卡配置

### Roofline 校准

### Day 17 第一动作
