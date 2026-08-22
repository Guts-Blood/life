# Day 09 — 数据质量、Mixture、Lineage 与去污染

日期：`2026-08-04`

状态：`complete_with_gate_c_waiver`

强度：工作日 4–5 小时

## 主要目标

把“有一个 JSONL”升级为可追溯、可审计、可复现实验的数据资产；为 Day 12 冻结两个只改变 mixture 比例的训练 manifest。

## 理论（75 分钟）

精读清单：[Day 09 — Data quality、mixture 与 lineage](../SCALING-BOOK-READING-GUIDE.md#day-09)。

- 质量维度：格式正确、回答可用、任务难度、语言/领域覆盖、长度、拒答与安全、合成数据偏差。
- mixture 的单位：按 example、raw token 或 supervised token 采样会产生不同训练分布。
- lineage：source、license、revision、获取时间、变换链、过滤规则、parent ID、内容 hash。
- exact/near duplicate 的作用域：数据集内部、跨 source、train 与 eval。
- decontamination 与 dedup 不等价；明确 normalization、matching threshold、被删除记录和误杀抽样。

## Coding（135 分钟）

- 建立 dataset manifest，至少包含 `sample_id/source/license/revision/split/skill/language/content_hash/transform_chain`。
- 生成 source、skill、language、长度和 supervised-token 分布；每个 source 人工抽样 10 条并记录 accept/reject reason。
- 做 exact dedup；对 near-duplicate 和 eval overlap 先输出候选及相似度，再人工确认，不静默删除。
- 冻结 Day 12 的两个 manifest：
  - `mix-A-balanced`：general/math/code/finance 四个 slice 各占 25% supervised tokens；
  - `mix-B-targeted`：finance 固定 25%，code 从 25% 提高到 50%。

  两组使用相同 source pool、质量过滤、总 supervised tokens 和 preprocessing；唯一实验变量是 mixture 比例。

## 训练 / 实验（45–60 分钟）

- 不训练模型。
- 原计划对 A/B manifest 各抽样 30 条；用户明确豁免 Gate C，因此实际为 0/60，不声称人工 QA 通过。
- 将 Day 10 的候选 eval pool 与两个 train manifest 做 exact + near-overlap 检查；今天发现 overlap 就修训练 manifest 并重新计算 hash，Day 10 再冻结通过检查的 eval split。

## 资源与租卡

- [Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 data curation、mixture、evaluation/decontamination 部分
- [Tülu 3 Decontamination](https://github.com/allenai/open-instruct/tree/main/decontamination)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- CPU only；不要为数据审计租 GPU。

## 产物

- [Day 09 完整 Pipeline（SVG）](../artifacts/reports/day09-pipeline.svg)
- [Day 09 完整 Pipeline（PNG）](../artifacts/reports/day09-pipeline.png)
- `../artifacts/data/day09-dataset-manifest.json`
- `../artifacts/reports/day09-data-quality-audit.md`
- `../artifacts/data/day09-mix-A-balanced.json`
- `../artifacts/data/day09-mix-B-targeted.json`
- `../artifacts/reports/day09-decontamination-report.md`

## 验收

- [x] 任意训练 sample 都能追溯到 source、revision 和变换链。
- [x] 报告 example 数与 supervised-token 数两种分布，不用样本数代替训练权重。
- [x] exact duplicate、near-duplicate candidate、train/eval overlap 分开报告。
- [x] A/B 的总 supervised tokens 相同，除 mixture 比例外的训练输入条件一致。
- [x] 所有过滤和去污染都有 before/after count 与可复查的删除记录。

复现性验收：11/11 checks、44 tests 通过。Gate C 是显式 waiver，不是 passed review。

## Daily Log

### 最大的数据质量风险

- Gate C 0/60 reviewed，是最终产物最大的残余质量风险。
- Near matcher 的 0 contamination 结论只适用于冻结的 normalization、字段和 threshold。
- A/B supervised tokens 相等，但 input tokens 为 728,317 vs 833,607；Day 12 仍需报告 compute 差异。

### A/B 实际 supervised-token 比例

- Mix A：general/math/code/finance = 61,734/61,734/61,734/61,734，合计 246,936。
- Mix B：general/math/code/finance = 30,867/30,867/123,468/61,734，合计 246,936。

### Day 10 第一动作

冻结通过 Day 09 overlap audit 的 160 条 eval candidates 的完整 decoder/scorer/protocol，并在训练前运行 Base checkpoint per-sample baseline。
