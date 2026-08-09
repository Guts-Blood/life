# Day 21 — Qwen3.5 v2 SFT Candidate Selection Audit

日期：`2026-08-16`  
状态：`not_started`  
强度：1 小时，仅阅读/复盘

## 主要目标

只审计并最终晋级 Day 16/17 的 Qwen3.5 v2 SFT candidates。Day 10/12 的 Qwen3-0.6B artifacts 保持历史只读，不进入 candidate set、tie-breaker 或 promotion decision。

## Candidate Boundary

- Base baseline：Day 15 exact Qwen3.5 Base，只作 delta 起点，不是 SFT candidate。
- Candidates：Day 16 预注册的 early/mid/final LoRA checkpoints；Day 17 resumed checkpoint 只有通过 resume equivalence 后才可映射回同一 candidate。
- Excluded：所有 Day 01–12 checkpoints、failed/invalid runs、临时 debug exports、只加载成功但无 resumable state 的路径。
- Downstream：只有本日最终 promoted SFT anchor 可作为 Day 23 DPO policy/reference anchor 和 Day 25 GRPO policy parent。

## Selection Audit（60 分钟）

精读清单：[Day 21 — Reliable checkpoint selection](../SCALING-BOOK-READING-GUIDE.md#day-21)。

- 10 分钟：验证 candidate IDs、Base/adapter/resumable/export hashes、processor/template/data/config 和累计 supervised tokens。
- 10 分钟：确认 eligibility、primary coding metric、general/math/format guardrails、最小有意义差异、tie 与 `inconclusive` 条件在结果揭盲前已冻结。
- 15 分钟：检查逐样本 predictions、length/truncation、source slices、bootstrap/paired uncertainty 和 bad cases。
- 10 分钟：核对 Day 17 uninterrupted/resumed equivalence，防止同一逻辑 checkpoint 重复计为独立试验。
- 10 分钟：盲化应用 selection rule，生成 promotion manifest 或 `no_eligible_qwen35_sft_anchor`。
- 5 分钟：封存所有 candidate evidence；新 v2 confirmation 继续未消费，留给候选锁定后的单次确认。旧 v1 frozen test 也保持未消费、只读。

## Coding / 训练

无。今天不改 scorer、不新增 candidate、不重跑 eval GPU，也不以 final train loss 推翻预注册规则。

## Evidence-first 产物

- `../artifacts/eval/day21-qwen35-sft-selection-policy.md`
- `../artifacts/eval/day21-qwen35-blinded-selection.json`
- `../artifacts/reports/day21-qwen35-sft-promotion-manifest.json`

Promotion manifest 至少记录：Base revision/hash、adapter/resumable checkpoint、merged inference export、processor/template/data/config/runtime hashes、selection evidence、resume/export parity 与 downstream key。

## 验收

- [ ] candidate set 只含 v2 Qwen3.5 SFT checkpoints，v1 权重和指标未参与选择。
- [ ] selection rule 在揭盲前冻结，primary metric、guardrails、uncertainty、tie 与 inconclusive 条件明确。
- [ ] resumed run 没有被重复计算为独立 candidate。
- [ ] 生成唯一 promoted SFT anchor，或诚实记录 no-eligible；没有 winner 时 Day 23/25 blocked。
- [ ] 新 v2 confirmation 未消费；旧 v1 frozen test 仍为 sealed/unconsumed history。

## Daily Log

### Candidate set / excluded set

### Blinded decision

### Promoted SFT anchor or blocker

### Remaining selection risks
