# Day 19 — Megatron Checkpoint/Profiler 与 Qwen3-32B 规划

日期：`2026-08-14`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

补完 Megatron optimizer/checkpoint/runtime 主链路，用 Profiler 验证瓶颈；修改一个观测点或并行配置后重跑，并将实测用于 Qwen3-32B 第一版容量方案。

## 理论（60 分钟）

精读清单：[Day 19 — 用 Profiler 证伪 Roofline](../SCALING-BOOK-READING-GUIDE.md#day-19)。把 Part 9 的 XLA/HLO 例子翻译为 PyTorch op、CUDA kernel、NCCL collective，不学习 JAX 操作本身。

- CPU launch、CUDA kernels、NCCL collective、overlap、idle gap。
- Memory-feasible vs performance-efficient。
- TP/PP/FSDP/CP degree、pipeline bubble、local tokens。

## Repo 通读收尾（75 分钟）

- 顺着 Day 18 的真实 stack 回读 `training.py -> pipeline schedule -> model -> loss -> optimizer`。
- 顺着 save/load 日志回读 distributed checkpoint serialization 与 optimizer state。
- 对每个 `UNKNOWN` edge 做代码搜索或最小断点验证；无法解决的保留具体问题。
- 输出 `megatron-sft-call-chain.md`，每个节点含输入/输出、rank group、collective、state ownership。

## Coding / Profiler（75 分钟）

- 对 Day 18 一个两卡配置抓 3–5 个 steady-state steps 的 profiler trace。
- 按 attention/GEMM/optimizer/NCCL/data/idle 分类时间。
- 把插桩改为一个可开关 hook/logger，或修改一个 reward 无关的 Megatron 观测点；保存 diff 后重跑 5 steps，证明理解调用位置。

## 训练 / 实验与 Capacity（90–120 分钟）

- Profiler 只抓短 active window，避免整段训练开 profiler。
- 找到 top operators、exposed NCCL 和 data gaps。
- 运行修改后的 5-step config，确认数值结果未被观测代码改变。
- 扩展 `capacity_plan.py` 输入 model config、precision、sequence、batch、GPU 数和 parallel degrees。
- 对 8×/16×/32×H100 各提出 Qwen3-32B 两个候选 topology，先做模型状态估算。

## 资源与租卡

- 最优：Day 18 双卡实例继续 2–4 小时抓 trace、修改重跑和 checkpoint audit，随后关机。
- 若双卡窗口已结束：1×H100 1–2 小时做单卡 trace，多卡 trace 使用 Day 18 日志。
- Capacity planning 用 CPU，不在 GPU 上写报告。

## 资料

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

## 验收

- [ ] 至少一个 bottleneck 有 trace 证据。
- [ ] Megatron 端到端图与 runtime/stack 对齐，所有未解边有具体 owner/question。
- [ ] 有一份最小源码/插桩 diff，以及修改前后 5-step 结果。
- [ ] 理论/实测时间差有分解。
- [ ] 32B 三种集群规模各有两个候选 topology、模型状态/卡与主要风险。
- [ ] 最终方案留到 Day 30，不在今天假装精确。

## Daily Log

### Trace bottleneck

### Megatron codepath / local diff

### 32B 候选 topology

### Day 20 Reading 问题
