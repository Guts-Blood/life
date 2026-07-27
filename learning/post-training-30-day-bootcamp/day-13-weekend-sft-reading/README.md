# Day 13 — 周末 Reading：Tülu 3 × Qwen3 Post-training

日期：`2026-08-08`

状态：`not_started`

强度：1 小时，仅阅读

## 主要目标

用一个完整开放 recipe（Tülu 3）和一个强模型技术报告（Qwen3）校准自己的小型 SFT：理解阶段、数据、目标与评测如何衔接，不把 production recipe 简化成框架命令列表。

## 理论（60 分钟）

精读清单：[Day 13 — Tülu 3 与 Qwen3 post-training 对读](../SCALING-BOOK-READING-GUIDE.md#day-13)。

- 30 分钟：[Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 pipeline、SFT mixture、preference data、RLVR、evaluation/decontamination 与明确报告的 negative results。
- 25 分钟：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 的 post-training stage diagram、thinking/non-thinking 训练与小模型构建路径；benchmark 大表不是 Core。
- 5 分钟：完成对照表：

| 维度 | Tülu 3 | Qwen3 | Day 12 小实验 |
|---|---|---|---|
| 上游 checkpoint | | | |
| data signal | | | |
| objective/stage | | | |
| eval/gate | | | |
| 可复现信息与缺口 | | | |

必须回答：

1. 哪些能力问题适合先改 SFT data，哪些需要 preference 或可验证 reward signal？
2. Tülu 3 的 openness 让哪些因果判断更可信？Qwen3 报告中哪些细节仍不足以复现？
3. Day 12 的结果最多支持什么结论，离 production post-training 还缺哪些环节？

## Coding

无。

## 训练 / 实验

无；只确认 Day 12 artifacts 已同步，不开启新 run。

## 资源与租卡

阅读不开 GPU。若 Day 12 任务仍在运行，只做健康检查和到点停止，不临时改参。

## 验收

- [ ] 完成对照表并写 3 个带适用边界的 takeaway。
- [ ] 至少记录一个公开 recipe 的 negative result 或无效方向。
- [ ] 能把自己的 SFT 放回 `Base -> SFT -> preference -> RL/RLVR -> eval` 链路。

### 三个 takeaway

-
-
-
