# Day 21 — Qwen3.5 v2 SFT Candidate Selection Audit

日期：`2026-08-16`  
状态：`blocked_no_day16_candidate`
强度：1 小时，仅阅读/复盘

## 当前阻塞

Day 16 已生成 machine-readable [`no_eligible_qwen35_sft_anchor`](../artifacts/reports/day16-qwen35-provisional-anchor.json)；main、early/mid/final 与 resumable candidate 均不存在。本日没有 candidate set 可做 blinded promotion audit，因此暂不生成空壳 selection/promotion artifacts。

原审计流程保留，等待新批准的 SFT charter 产生候选后再执行；Base 和失败 probe 都不能填入 S1 slot。

## 主要目标

只审计并最终晋级新 SFT charter 产生的 Qwen3.5 v2 SFT candidates。Day 10/12 的 Qwen3-0.6B artifacts 保持历史只读，不进入 candidate set、tie-breaker 或 promotion decision。

## Candidate Boundary

- Base baseline：Day 15 exact Qwen3.5 Base，只作 delta 起点，不是 SFT candidate。
- Candidates：新 SFT charter 预注册的 early/mid/final LoRA checkpoints；若顺序课程 Day 20 Optional R 实际运行，resumed path 永远映射回原逻辑 candidate，不能成为独立候选。
- Excluded：所有 Day 01–12 checkpoints、failed/invalid runs、临时 debug exports、只加载成功但无 resumable state 的路径。
- Downstream：只有本日最终 promoted SFT anchor 可作为 Day 23 DPO policy/reference anchor 和 Day 25 GRPO policy parent。

## Selection Audit（60 分钟）

精读清单：[Day 21 — Reliable checkpoint selection](../SCALING-BOOK-READING-GUIDE.md#day-21)。

- 10 分钟：验证 candidate IDs、Base/adapter/resumable/export hashes、processor/template/data/config 和累计 supervised tokens。
- 10 分钟：确认 eligibility、primary coding metric、general/math/format guardrails、最小有意义差异、tie 与 `inconclusive` 条件在结果揭盲前已冻结。
- 15 分钟：检查逐样本 predictions、length/truncation、source slices、bootstrap/paired uncertainty 和 bad cases。
- 10 分钟：核对 checkpoint integrity；若 Day 20 Optional R 有证据，再核对 uninterrupted/resumed equivalence。未运行 Optional R 不构成淘汰理由，但任何 resumed path 都不能重复计为独立试验。
- 10 分钟：盲化应用 selection rule，生成 promotion manifest 或 `no_eligible_qwen35_sft_anchor`。
- 5 分钟：封存所有 candidate evidence；新 v2 confirmation 继续未消费，留给候选锁定后的单次确认。旧 v1 frozen test 也保持未消费、只读。

## Coding / 训练

无。今天不改 scorer、不新增 candidate、不重跑 eval GPU，也不以 final train loss 推翻预注册规则。

## Evidence-first 产物

- `../artifacts/eval/day21-qwen35-sft-selection-policy.md`
- `../artifacts/eval/day21-qwen35-blinded-selection.json`
- `../artifacts/reports/day21-qwen35-sft-promotion-manifest.json`

Promotion manifest 至少记录：Base revision/hash、adapter/checkpoint integrity、merged inference export、processor/template/data/config/runtime hashes、selection evidence、export parity 与 downstream key；Day 20 Optional R 的 exact-resume evidence 若存在则作为 supplemental field 记录，不是必填 promotion gate。

## 验收

- [ ] candidate set 只含 v2 Qwen3.5 SFT checkpoints，v1 权重和指标未参与选择。
- [ ] selection rule 在揭盲前冻结，primary metric、guardrails、uncertainty、tie 与 inconclusive 条件明确。
- [ ] 若存在 resumed run，它没有被重复计算为独立 candidate；没有 Optional R 证据时也没有伪造 resume parity。
- [ ] 生成唯一 promoted SFT anchor，或诚实记录 no-eligible；没有 winner 时 Day 23/25 blocked。
- [ ] 新 v2 confirmation 未消费；旧 v1 frozen test 仍为 sealed/unconsumed history。

## Daily Log

### Candidate set / excluded set

### Blinded decision

### Promoted SFT anchor or blocker

### Remaining selection risks
