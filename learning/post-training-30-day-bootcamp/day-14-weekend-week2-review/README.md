# Day 14 — 周末 Review：Week 2 Evidence Review

日期：`2026-08-09`

状态：`not_started`

强度：1 小时，仅阅读/复盘

## 主要目标

审计 Day 08–13 的证据链，判断哪些结论可以保留、哪些实验无效、下周只该验证哪个最高优先级假设。今天不补跑训练。

## 理论 / 复盘（60 分钟）

精读清单：[Day 14 — Week 2 evidence review](../SCALING-BOOK-READING-GUIDE.md#day-14)。

- 15 分钟：数据证据——contract/template hash、有效 label、lineage、质量抽样、mixture 与 decontamination。
- 15 分钟：训练证据——tiny-overfit step trace、A/B config diff、token budget、curves、resume 与 checkpoint 完整性。
- 15 分钟：评测证据——Base/selected checkpoints 的同协议逐样本结果、slice 变化、退化和不确定性。
- 10 分钟：每项结论写成 `Claim -> Evidence -> Alternative explanation -> Limit`，删除只由 train loss 支持的能力结论。
- 5 分钟：只选一个 Week 3 实验，写 `假设 -> 单一变量 -> 指标 -> 停止/作废条件`。

## Coding

无。发现缺失 artifact 时记录缺口，不在周末临时补脚本。

## 训练 / 实验

无；CPU only，不启动 GPU。

## 资源与租卡

- Day 08–13 的 manifests、logs、predictions、curves 与 reports。
- [Weekly review template](../templates/weekly-review.md)。
- CPU only；今天不新增数据、不改 scorer、不启动训练。

## 产物

- `../artifacts/reports/week2-evidence-review.md`
- `../artifacts/reports/week2-claim-evidence-table.md`

## Week 2 Gate

- [ ] SFT data contract、template 和 loss mask 有逐 token 证据。
- [ ] 训练数据可追溯，mixture 用 supervised tokens 报告，并完成 train/eval 去污染。
- [ ] Frozen Base baseline 早于训练且协议未静默变化。
- [ ] tiny overfit 证明 step pipeline 可学习并能恢复，但未被误写为能力提升。
- [ ] Controlled A/B 唯一变量成立；否则实验明确标记 invalid。
- [ ] selected checkpoint 的改善与退化都有逐样本证据和适用边界。
- [ ] Week 3 只有一个已预注册的最高优先级实验，不用堆 run 掩盖未知。

### 本周最强证据

### 被否决/降级的结论

### Week 3 唯一优先实验
