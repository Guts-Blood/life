# Day 30 — 30B+ Design、Clean Reproduction 与模拟汇报

日期：`2026-08-25`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

把一个月的结果收敛成可交接的 30B+ post-training design，并证明最小训练/Eval 闭环能从保存的环境与文档复现。

## 理论（45 分钟）

精读清单：[Day 30 — 30B+ design review](../SCALING-BOOK-READING-GUIDE.md#day-30)。把 Part 12 Quiz 5 Q2 改写成自己的 Qwen 题，并区分 `known / estimated / must benchmark`。

- 回顾 dense/MoE、SFT/DPO/GRPO、train/rollout/eval 的资源差异。
- 为设计中的每个 parallel degree、GPU pool 和 Eval gate 写出选择理由。

## Coding（60 分钟）

- 从保存镜像/新 shell 执行 clean-room 最小流程：环境检查、5–10 step SFT、checkpoint load、固定小 Eval。
- 修复 README/commands 中缺失的隐式步骤。
- 运行关键 scripts/tests：data validator、scorers、capacity plan。

## 训练 / 实验（60 分钟执行/验证）

- 不重复主训练，只做最小 reproducibility smoke。
- 检查 ms-swift SFT、Megatron SFT、slime RL 的 selected artifacts、commits/diffs、manifest、logs 和 predictions 均可定位。
- Megatron/slime 未完成时，诚实记录具体 blocker、静态验证到哪条边和下一次最小实验，不伪造“通读/完整闭环”。

## 30B+ Design Doc（90 分钟）

选择 Qwen3-32B dense 或 Qwen3.5-35B-A3B-Base MoE；若选 MoE，再用 Qwen3.6-35B-A3B 做 current-state 对照。设计包含：

- GPU/节点、DP/FSDP/TP/PP/CP/EP、precision、sequence、micro/global batch。
- SFT model state/activation/checkpoint 预算与预计吞吐范围。
- RL actor/reference/rollout/reward 的 GPU 分配、vLLM/SGLang、weight sync、staleness。
- 数据 contract、checkpoint/recovery、NCCL/OOM/reward collapse runbooks。
- Public/frozen/pairwise/safety/latency Eval gate。
- 已知、假设、必须用 pilot benchmark 验证的未知量。

## 资源与租卡

- Clean smoke：1×H100 80GB，预计 2–4 小时；使用已保存镜像减少安装时间。
- Design/report：CPU only。完成 smoke 后立即关机，不在 H100 上写汇报。

## 模拟汇报（45 分钟）

准备并录一次 15 分钟说明：

1. 跑通了什么。
2. SFT 数据/loss 如何工作。
3. Scaling 的主要约束。
4. Eval 如何避免错误结论。
5. DPO/GRPO 工程差异。
6. 如何扩到 30B+。
7. 仍不知道什么、入职第一周验证什么。

## 最终验收

- [ ] Clean smoke 无隐式手工步骤。
- [ ] Megatron 与 slime 各有 `file:function` 调用链、runtime evidence 和一次受控修改重跑。
- [ ] 30B+ design 可由他人评审，而不是术语列表。
- [ ] 所有主要结论链接到 config/log/prediction/report。
- [ ] 完成 [`../PROGRESS.md`](../PROGRESS.md) 与最后 weekly review。
- [ ] 写出入职后第一周的 5 个高价值问题。

## Final Log

### 实际完成的闭环

### 最强证据

### 最大缺口

### 入职第一周问题

1. 
2. 
3. 
4. 
5. 
