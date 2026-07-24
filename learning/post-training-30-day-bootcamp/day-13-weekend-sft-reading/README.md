# Day 13 — 周末 Reading：Qwen3 Training

日期：`2026-08-08`  
状态：`not_started`  
强度：1 小时，仅阅读

## 主要目标

了解 Qwen3 自己如何描述 pretraining/post-training 阶段，并把自制 SFT 放在完整流程中的正确位置。

## 理论（60 分钟）

精读清单：[Day 13 — Qwen post-training 全局图](../SCALING-BOOK-READING-GUIDE.md#day-13)。只读报告的 architecture、pre-training overview、post-training pipeline；benchmark 大表不是 Core。

- 10 分钟：读 [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 的 abstract/introduction。
- 10 分钟：读 model architecture。
- 10 分钟：读 pre-training overview。
- 25 分钟：读 post-training pipeline；benchmark 大表不是 Core。
- 5 分钟：写三个 bullet：
  1. Base、SFT/Instruct、thinking model 的训练阶段差异。
  2. 自己 Day 12 的训练缺少 Qwen production pipeline 哪些部分？
  3. 下周 ablation 最值得验证的假设是什么？

## Coding

无。

## 训练 / 实验

无；即使 Day 12 夜间 run 仍在跑，也只查看是否健康，不做新调参。

## 资源与租卡

阅读不开 GPU。如果 Day 12 长任务已完成，确认备份后关机。

### 三个 bullet

- 
- 
- 
