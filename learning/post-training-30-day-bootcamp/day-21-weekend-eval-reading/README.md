# Day 21 — 周末 Reading：LLM Evaluation

日期：`2026-08-16`  
状态：`not_started`  
强度：1 小时，仅阅读

## 主要目标

在下周写 Eval 代码前，明确 benchmark、task eval、人类评测和 LLM judge 的职责与偏差。

## 理论（60 分钟）

精读清单：[Day 21 — Eval/rollout inference](../SCALING-BOOK-READING-GUIDE.md#day-21)。Part 7 的 latency/throughput、memory、continuous batching 用于连接 Eval generation 与后续 GRPO rollout。

- 20 分钟：读 [Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) 的 `Theoretical estimates for LLM latency and throughput`。
- 10 分钟：读 `What about memory?`。
- 15 分钟：读 `Designing an Effective Inference Engine -> Continuous batching`。
- 10 分钟：读 `Distributing Inference -> Prefill/Generation`。
- 5 分钟：写一张 eval generation 与 GRPO rollout 的共同资源表。

## Coding

无。

## 训练 / 实验

无。

## 资源与租卡

CPU only；不要开 Eval GPU。

### 五类风险

1. 
2. 
3. 
4. 
5. 
