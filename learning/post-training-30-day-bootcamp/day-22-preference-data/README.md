# Day 22 — Preference Data：Provenance、Length Bias 与 Held-out

日期：`2026-08-17`
状态：`not_started`
强度：4–5 小时

## 主要目标

构建一份可审计的 preference dataset。重点不是先训练 DPO，而是确保 chosen/rejected 的来源、偏好理由、长度差和 train/held-out 边界可信。

## 理论（75 分钟）

精读清单：[Day 22 — Preference data reliability](../SCALING-BOOK-READING-GUIDE.md#day-22)。

- Pairwise preference 表达的是相对偏好，不天然等于绝对正确。
- Human、LLM judge、rule/verifier、best-of-N 和 synthetic pair 的偏差不同。
- Length/style/position bias、tie/ambiguous pair、同源泄漏和 prompt contamination。
- Random row split 为什么会让同一 prompt/template/source family 跨 train/held-out。

## Coding（105 分钟）

- 定义 manifest/schema：`pair_id`、prompt hash、chosen/rejected、source、license、creation method、annotator/judge/version、rubric、timestamp、quality flags。
- 写 validator：空值、chosen=rejected、重复/近重复、template mismatch、异常长度差、prompt/source-family 泄漏。
- 生成 chosen/rejected token length、`Δlength`、来源、类别、质量标签分布。
- 实现 group split：按 prompt/source family 切 train/dev/held-out，而非按 row 随机切。

## 数据实验（105–120 分钟）

- 从已有 SFT bad cases 或一份来源清楚的数据中整理 200–500 pairs；规模不足时宁可少而可审计。
- 人工盲审至少 50 pairs，允许 `tie/ambiguous/reject`，记录 rubric disagreement。
- 构造 length-matched slice、large-Δlength slice 和 source-held-out slice。
- 交换 chosen/rejected 显示顺序做 position-bias 检查；不修改语义来“平衡”数据。
- 冻结 Day 23 使用的 train/dev/held-out manifest 与 checksum。

## 资源与租卡

- CPU only；tokenize/validator/split 不需要 GPU。
- 不在今天启动 DPO；发现 provenance 不明或泄漏时优先丢弃数据。

## Evidence-first 产物

- `../artifacts/data/preference-manifest.json`
- `../artifacts/scripts/validate_preference_data.py`
- `../artifacts/reports/day22-preference-audit.md`
- train/dev/held-out IDs 与 checksums

## 验收

- [ ] 每一 pair 可追溯到创建方式、source 和偏好依据。
- [ ] 50 对盲审完成，ambiguous/tie 没被强行变成 chosen/rejected。
- [ ] length/position/source bias 有单独 slice。
- [ ] held-out 按 prompt/source family 隔离并在训练前冻结。

## Daily Log

### Provenance coverage

### Length / source bias

### Held-out leakage check

### Day 23 第一动作
