# Day 27 — 周末 Reading：GRPO 与 Online RL

日期：`2026-08-22`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

理解 GRPO 的 group advantage、on-policy 边界和 reference/KL，并将算法变量映射到 Day 26 的 slime data objects。

## 理论（60 分钟）

精读清单：[Day 27 — GRPO objective 与 slime objects](../SCALING-BOOK-READING-GUIDE.md#day-27)。

- 35 分钟：读 [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) 的 objective/method。
- 25 分钟：对照 slime 的 `Sample`/rollout fields，回答：
  1. 一个 prompt 产生多少 completions，在哪一步组成 group？
  2. Reward、advantage、old/ref logprob 分别由谁产生、谁消费？
  3. 哪些延迟会让训练不再严格 on-policy？

## Coding

无。

## 训练 / 实验

无；不启动 slime、SGLang 或 GPU。

## 资源与租卡

CPU only。只读论文和 Day 26 的源码图。

### 三个答案

1. 
2. 
3. 
