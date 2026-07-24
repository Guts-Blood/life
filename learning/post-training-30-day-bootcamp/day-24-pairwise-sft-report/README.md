# Day 24 — Pairwise、统计与 SFT Eval Report

日期：`2026-08-19`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

用盲化 pairwise、人类 anchor、置信区间和 bad cases 完成 SFT 是否值得 promotion 的正式结论。

## 理论（60 分钟）

精读清单：[Day 24 — Pairwise/Judge/统计](../SCALING-BOOK-READING-GUIDE.md#day-24)。今天暂停 Scaling Book 新章节，建立 `bias 风险 -> control -> evidence` 表。

- Win/tie/loss、position/length/style bias。
- Human anchor、judge calibration、bootstrap CI。
- Guardrail 与 promotion gate。

## Coding（90 分钟）

- 实现 blind pair generation：隐藏 model、随机 A/B、保存 permutation。
- 实现 `bootstrap_ci.py`，输出 overall/slice 95% CI。
- 可选：保存固定 judge prompt、model version、raw/parsed result。

## 训练 / 实验（120 分钟人工 Eval/分析）

- 人工标注 50–100 对 Base vs SFT；rubric 先固定。
- 对至少 50 个 SFT bad cases 做 taxonomy。
- 可选固定 judge 对同样 anchor 评分，测 agreement 和 position flip。
- 合并 public、deterministic、pairwise、length、latency、regression。

## 资源与租卡

- CPU only；人工/统计/报告不开 GPU。
- Judge 优先用已有可访问且版本固定的 API；不为了低质量本地 judge 租卡。

## Eval Report 必答

1. Overall/哪些 slices 提升？
2. 哪些退化，是否越过 guardrail？
3. CI 是否支持“真的提升”？
4. 长度、格式、拒答、延迟如何变化？
5. 失败更像数据、训练还是 eval 问题？
6. Promotion、reject 还是 inconclusive？

## 产物

- `../artifacts/eval/day24-human-anchor.jsonl`
- `../artifacts/scripts/bootstrap_ci.py`
- `../artifacts/reports/sft-eval-report.md`
- `../artifacts/eval/promotion-gate.json`

## 验收

- [ ] 模型身份盲化、A/B 顺序随机。
- [ ] 结论包含 CI 和逐样本 evidence。
- [ ] 允许结论为不 promotion/inconclusive。

## Daily Log

### Human/Judge agreement

### Promotion decision

### Day 25 第一动作
