# Day 20 — 周末 Reading：MoE

日期：`2026-08-15`  
状态：`not_started`  
强度：1 小时，仅阅读

## 主要目标

建立 total parameters、activated parameters、expert parallel 和 all-to-all 的最小 MoE 心智模型。

## 理论（60 分钟）

精读清单：[Day 20 — MoE active compute 与 EP](../SCALING-BOOK-READING-GUIDE.md#day-20)。Part 4 MoE、Part 12 EP 与 Qwen 35B-A3B config 各回答不同问题，不混成一个“3B 模型”。

- 15 分钟：读 [Transformer Math](https://jax-ml.github.io/scaling-book/transformers/) 的 `Sparsity and Mixture-of-Experts`。
- 20 分钟：读 [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) 的 `Expert Parallelism`。
- 20 分钟：读 [Qwen3.5-35B-A3B-Base model card/config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base)，区分 text/vision components、total/active parameters。
- 5 分钟：写下面三条结论。

只写三条：

1. 为什么权重显存看 35B，而每 token 主要计算只激活约 3B？
2. Token dispatch 产生什么通信？
3. Expert imbalance 会在指标/trace 中如何表现？

## Coding

无。

## 训练 / 实验

无。

## 资源与租卡

CPU only；不要租 GPU。

### 三条结论

- 
- 
- 
