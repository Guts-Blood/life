# Day 34 — 8B Controlled SFT 与 T1 Selection

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；GPU wall time 按 Day 33 实测预算

## 主要目标

使用 Day 33 通过的唯一配置训练 8B SFT trajectory，并只用 `capstone_dev` 和预注册规则选择 `T1`。今天不运行 domain RL，不揭盲 frozen。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 34 — 8B SFT Selection](../SCALING-BOOK-READING-GUIDE.md#day-34)。

## 训练 / 实验

- 从 exact `T0` 和 frozen SFT manifest 开始；
- 固定 supervised label-token budget，不以 epochs/steps 模糊预算；
- 保存 early/mid/final 与完整 resume state；
- 每个 candidate 使用相同 generation/scorer 跑 capstone dev；
- 报告 domain slice、general guardrails、parse/system errors、paired deltas 与 uncertainty；
- 用 Day 31 selection policy 选择 `T1`，不因 final loss 最低自动选 final。

至少执行一次预定中断/恢复，证明长 run 使用了 Day 33 的 checkpoint contract。

## Evidence-first 产物

- `../artifacts/logs/capstone/day34-8b-sft/`
- `../artifacts/reports/capstone/day34-8b-sft-selection.md`
- `../artifacts/checkpoints/capstone/T1-manifest.json`
- candidate predictions、selection record、resume evidence

## 验收

- [ ] T0→T1 的 data/config/code/hardware/checkpoint lineage 完整。
- [ ] candidate 比较使用相同 dev manifest 与 comparison key。
- [ ] T1 通过 domain minimum、general guardrails 和 error-rate gate，或明确返回 `inconclusive`。
- [ ] `capstone_frozen` 未运行、未评分、未人工查看。
- [ ] T1 既有可推理权重，也有可继续 domain RL 的完整训练交接清单。

## Stop Conditions

若 dev 增益不存在、TP/resume 漂移或 general guardrail 失败，停止 teacher 路线并先修 SFT；不能靠 RL 挽救未通过的 SFT 起点。
