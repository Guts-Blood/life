# Day 09 — 数据质量、Mixture、Lineage 与去污染

日期：`2026-08-04`

状态：`not_started`

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
  - `mix-A-balanced`：general/math/code 三个 slice 等 supervised-token 配比；
  - `mix-B-targeted`：提高一个预注册目标 slice 的配比。

  两组使用相同 source pool、质量过滤、总 supervised tokens 和 preprocessing；唯一实验变量是 mixture 比例。

## 训练 / 实验（45–60 分钟）

- 不训练模型。
- 对 A/B manifest 各抽样 30 条，验证内容质量和实际 supervised-token 比例。
- 将 Day 10 的候选 eval pool 与两个 train manifest 做 exact + near-overlap 检查；今天发现 overlap 就修训练 manifest 并重新计算 hash，Day 10 再冻结通过检查的 eval split。

## 资源与租卡

- [Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 data curation、mixture、evaluation/decontamination 部分
- [Tülu 3 Decontamination](https://github.com/allenai/open-instruct/tree/main/decontamination)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- CPU only；不要为数据审计租 GPU。

## 产物

- `../artifacts/data/day09-dataset-manifest.json`
- `../artifacts/reports/day09-data-quality-audit.md`
- `../artifacts/data/day09-mix-A-balanced.json`
- `../artifacts/data/day09-mix-B-targeted.json`
- `../artifacts/reports/day09-decontamination-report.md`

## 验收

- [ ] 任意训练 sample 都能追溯到 source、revision 和变换链。
- [ ] 报告 example 数与 supervised-token 数两种分布，不用样本数代替训练权重。
- [ ] exact duplicate、near-duplicate candidate、train/eval overlap 分开报告。
- [ ] A/B 的总 supervised tokens 相同，除 mixture 比例外的训练输入条件一致。
- [ ] 所有过滤和去污染都有 before/after count 与可复查的删除记录。

## Daily Log

### 最大的数据质量风险

### A/B 实际 supervised-token 比例

### Day 10 第一动作
