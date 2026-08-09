# Day 22 — Coding Preference Data：Provenance、Processor 与 Sandbox Evidence

日期：`2026-08-17`
状态：`not_started`
强度：4–5 小时

## 主要目标

为 Qwen3.5 v2 构建一份可审计的 coding preference dataset。重点不是先训练 DPO，而是确保 prompt/chosen/rejected、processor/template、代码测试、偏好理由、长度差与 train/dev/held-out 边界可信。

Day 01–12 的 raw bad cases 可以作为选题线索，但旧 rendered/tokenized records、0.6B generations 和 model-specific hashes 不进入 v2 pair manifest。

## 理论（60 分钟）

精读清单：[Day 22 — Preference data reliability](../SCALING-BOOK-READING-GUIDE.md#day-22)。

- Pairwise preference 表达相对偏好，不天然等于代码正确、安全或可执行。
- Unit-test/verifier、human review、LLM judge、best-of-N 和 synthetic repair pairs 的偏差不同。
- Length/style/position bias、test overfitting、invalid sandbox result、tie/ambiguous pair 与 prompt contamination。
- Random row split 为什么会让同一 problem/test/template/source family 跨 train/held-out。

## Schema 与 Coding（120 分钟）

Manifest 至少包含：

- `pair_id`、problem/prompt hash、chosen/rejected raw text；
- Qwen3.5 model key、processor/tokenizer/template revision/hash、rendered prompt hash；
- chosen/rejected response token spans、length、truncation/status；
- programming language、task/source/license、creation method；
- test/sandbox image or environment digest、test manifest/hash、timeout、stdout/stderr/exit code；
- annotator/judge/verifier name/version、rubric、timestamp、quality flags；
- prompt/problem/test/source-family split key。

实现 validator：空值、chosen=rejected、语义/近重复、template mismatch、response span 错位、异常长度差、不可复现测试、timeout/error 被当成 wrong answer、prompt/test/source-family 泄漏。

使用 Day 15 冻结的 Qwen3.5 processor/template 重新 tokenize chosen/rejected；断言 prompt 部分完全一致且只计算 response log-prob。禁止复用 v1 token IDs 或 assistant spans。

## 数据实验（105–120 分钟）

- 从来源清楚的 coding tasks、Day 15/16 v2 bad cases 或可授权数据中整理 200–500 pairs；规模不足时宁可少而可审计。
- 在固定 CPU sandbox 中重放 chosen/rejected tests，保存 raw evidence；测试不足以区分时标记 `ambiguous`，不伪造偏好。
- 人工盲审至少 50 pairs，允许 `tie/ambiguous/reject`，记录 rubric disagreement。
- 构造 length-matched、large-Δlength、source-held-out、test-family-held-out 和 timeout/error slices。
- 交换 chosen/rejected 展示顺序做 position-bias 检查；不修改语义来“平衡”数据。
- 按 problem/test/source family 切 train/dev/held-out，并在 Day 23 前冻结 IDs 与 checksums。

## 资源与租卡

- CPU only；processor/tokenization、validator、sandbox replay 和 split 不需要 GPU。
- 没有 Day 21 promoted SFT anchor 时仍可完成数据审计，但 Day 23 必须保持 blocked。
- 今天不启动 DPO，不修改 Day 15/16 scorer 或 frozen eval。

## Evidence-first 产物

- `../artifacts/data/day22-qwen35-coding-preference-manifest.json`
- `../artifacts/scripts/validate_day22_coding_preference.py`
- `../artifacts/reports/day22-qwen35-coding-preference-audit.md`
- train/dev/held-out IDs、processor/render hashes、sandbox/test evidence 与 checksums

## 验收

- [ ] 每个 pair 可追溯到 source、creation method、test evidence 和偏好依据。
- [ ] Qwen3.5 processor/template 与 response masks 有逐 token audit，v1 tokens 未复用。
- [ ] 50 对盲审完成，ambiguous/tie/error 没被强行变成 preference。
- [ ] length/position/source/test-family/timeout bias 有单独 slice。
- [ ] held-out 按 problem/test/source family 隔离并在训练前冻结。

## Daily Log

### Provenance / processor coverage

### Sandbox replay / ambiguous pairs

### Length / source / test bias

### Held-out leakage check

### Day 23 第一动作
