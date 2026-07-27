# Day 03 — Roofline 与 H100

日期：`2026-07-29`  
状态：`done`（用户确认 Day 03 已完成；保留原清单，不补写未记录的 artifact）

强度：工作日 4–5 小时

## 今日结果

把已有 Roofline 理解变成一份能预测训练 step time 和瓶颈的 H100 worksheet。

## 已有基础

先复习你已经写过的记录：[Scaling Book Part 0/1 学习记录](../../daily_work/20260613/20260613_scaling_book_part0_part1_learning_track.md)。不需要重新总结 Part 0/1，重点补上 H100 和真实 SFT 的连接。

## 时间安排

- 30 分钟：复习已有笔记，列出还没验证的假设。
- 60 分钟：完成 Part 1 matmul/network roofline 与 H100 问题。
- 45 分钟：读 Part 12 的 GPU/Memory/specs 并替换为实际 AutoDL H100 数据。
- 90 分钟：计算 Qwen3-1.7B 的理论 step time 上下界。
- 30 分钟：设计 Day 06/09 需要记录的测量指标。

## 理论

精读清单：[Day 03 — Roofline 落到 H100](../SCALING-BOOK-READING-GUIDE.md#day-03)。Core 是 Part 1 `Matrix multiplication`、`Network communication rooflines`、Problems Q3/Q5，以及 Part 12 的 H100 硬件/内存表。

互动测验使用 [`DAY-03-READING-QUIZ.md`](./DAY-03-READING-QUIZ.md) 跟踪原始回答、点评和修正版结论。

- [x] Day 03 Reading Quiz：`completed`（`2026-07-26`–`2026-07-27`）。

- Compute/HBM/network roofline、arithmetic intensity、MFU 与 strong scaling。
- 把已有 TPU/通用结论改写为当前 H100 和后续训练配置的预测。

## Coding

- 为 memory estimator 增加 FLOPs、理论 step time 和 MFU scenario 输出，或单独建立 `roofline_estimator.py`。
- 为 Day 06/09 定义统一 benchmark JSON schema。

## 训练 / 实验

- 不做长训练。若 Day 01 环境需要补测，可用一个小 matmul benchmark 验证 HBM/compute 量级。
- 真实训练预测留到 Day 06、09、12 对照。

## 资源与租卡

- 理论/Coding：CPU only。
- 可选：1×H100 30 分钟做 microbenchmark；已有可靠 H100 数据时不租。

## Core

- [ ] 写清 `T_math`、`T_HBM`、`T_network` 的公式和单位。
- [ ] 从实际 H100 型号/拓扑获得 peak FLOPs、HBM bandwidth、GPU 间连接信息。
- [ ] 计算一个 BF16 matmul 的 arithmetic intensity。
- [ ] 用 local tokens 解释 micro-batch/sequence 对 MFU 的影响。
- [ ] 为 MFU=30%/40%/50% 估算 Qwen3-1.7B SFT step time。
- [ ] 预注册要在真实训练中验证的预测。

## 预注册示例

```text
Hypothesis: 在 effective batch 不变时，提高 micro-batch、降低 accumulation，
会提高 tokens/s，但显存上升；收益在 GPU 已 compute-bound 后变小。

Measure: tokens/s, step_time, peak_memory, label_tokens/s.
```

## 产物

- `../artifacts/reports/qwen17b-h100-roofline.md`
- Day 12 ablation 的预注册假设

## Definition of Done

- 能解释 theoretical peak、MFU 和实际 tokens/s 之间的关系。
- 能说明“GPU utilization 99%”为什么不等于高 MFU。
- 写出至少三个理论模型可能与真实 profiler 不符的原因。

## Daily Log

### 预测

### 尚缺的硬件信息

### 最有价值的公式

### Day 04 第一动作
