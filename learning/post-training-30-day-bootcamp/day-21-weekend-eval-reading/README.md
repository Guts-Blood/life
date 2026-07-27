# Day 21 — 周末 Reading：Eval 与 Checkpoint Selection 可靠性

日期：`2026-08-16`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

预注册一个不会被 final train loss、反复试验和数据泄漏误导的 checkpoint selection policy。

## 理论与复盘（60 分钟）

精读清单：[Day 21 — Reliable checkpoint selection](../SCALING-BOOK-READING-GUIDE.md#day-21)。

- 15 分钟：区分 train loss、validation loss、task metric、guardrail 和人工 bad case。
- 15 分钟：检查 Day 10/12 的 frozen eval 是否在训练前冻结，是否存在 prompt/source leakage。
- 15 分钟：理解多 checkpoint/多 metric 带来的 multiple-comparison 与 winner's curse。
- 15 分钟：填写 selection policy。

Selection policy 必须提前写明：

1. 候选 checkpoint 集合和评测配置。
2. 一个 primary metric、若干 guardrails、最小有意义差异。
3. slice、置信区间和 tie-breaker。
4. 何时结论是 `inconclusive`，而不是强行选一个 winner。
5. 选完后的独立 held-out confirmation，不再用它反复调参。

## Coding

无。

## 训练 / 实验

无；只用已有 checkpoint metadata 与 frozen-eval 样例做纸面 selection rehearsal。

## 资源与租卡

CPU only；严格 60 分钟，不启动 eval GPU。

## Evidence-first 产物

- `../artifacts/eval/checkpoint-selection-policy.md`
- 一次盲化 selection rehearsal 与泄漏检查

## 验收

- [ ] selection rule 在看候选结果前冻结。
- [ ] primary metric、guardrail、CI 和 inconclusive 条件明确。
- [ ] held-out confirmation 不参与反复调参。
- [ ] 能解释为什么 lowest train loss 不一定是最佳 checkpoint。

### 五类选择风险

1. 
2. 
3. 
4. 
5. 
