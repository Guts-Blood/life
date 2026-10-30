---
id: AV2-W09
title: 用户 query 挖掘到复杂化 pipeline
status: completed / feeding benchmark
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-18
updated_at: 2026-05-27
---

# AV2-W09 用户 Query 挖掘到复杂化 Pipeline

## Scope

梳理一个可执行看板，用来把真实用户 query 挖掘出来，筛选、去重、分类、复杂化，再回流到 agent_v2 eval、skill、prompt 或产品需求。

## Why It Matters

真实 query 是 agent_v2 后续提升的燃料。复杂化不是单纯把题写难，而是把真实需求扩展成覆盖长链路、多约束、跨工具、失败恢复和 artifact 质量的可复现 case。

## Pipeline Link

主表见：[Query complexification pipeline](../runs/query-complexification-pipeline.md)

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Intake quality | query source / logs / user feedback | query 有来源、意图和上下文，不只是孤立句子 |
| Complexity rubric | review notes | 复杂化后的 case 有明确新增约束和验收标准 |
| Dedup / clustering | board labels | 相似 query 能聚类，不重复堆 case |
| Eval reuse | eval / workstream links | 高价值 case 能回流 GLM eval、skill run 或 prompt/tool 修改 |
| Product signal | PM review | 能区分工程问题、产品缺口和模型能力缺口 |

## Pipeline Stages

| Stage | Purpose | Exit Criteria |
| --- | --- | --- |
| Intake | 收集用户 query、上下文和来源 | 有 source、raw query、场景、用户意图 |
| Normalize | 清洗、去重、聚类、打标签 | 有 canonical query、cluster、domain、能力标签 |
| Complexity Design | 增加真实约束、多步骤、artifact、错误恢复 | 有复杂化版本、复杂度理由、验收标准 |
| Review | 判断是否值得进入 eval / skill / product | 有 owner、priority、risk、目标 workstream |
| Run / Evaluate | 跑 Sonnet baseline、GLM candidate 或 skill case | 有 pass/fail、issue type、artifact、trace |
| Backlog / Ship | 回流到 eval、prompt、skill、tool、产品需求 | 有 follow-up、链接和决策 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27 | recent-master, tab-web | 完成 query 挖掘、用户画像分析 -> 生成 benchmark query 的 pipeline | 把用户画像和真实 query 挖掘结合起来，可以产出更贴近真实需求的 benchmark query | pipeline 已完成，开始回流 AV2-W13 benchmark | completed | feed benchmark | 持续把高价值 query / persona cluster 回流 benchmark deep cases |
| 2026-05-18 | recent-master, tab-web | 新增 query 挖掘 -> 复杂化 pipeline workstream | 结构化看板能把真实 query 变成可复用 eval / skill / 产品输入 | 待建立第一批 query 样本和复杂化规则 | pending | scope board | 定义字段、stage、质量门槛，补第一批样本入口 |

## Open Risks

- query 来源如果不清楚，复杂化后的 case 可能脱离真实用户需求。
- 复杂化标准如果只追求难度，会让 eval 失真，不能指导 GLM 替换或 skill 改进。
- 没有 owner / reviewer 时，case 会卡在看板里，不能回流到工程动作。

## Next Actions

- [x] 确认 query 来源：用户日志、产品反馈、手工收集、实习生 skill run 输出。
- [x] 定义看板字段：raw query、source、domain、ability、complexity delta、acceptance、owner、reviewer、target workstream。
- [x] 定义复杂化 rubric：多步骤、多约束、跨工具、长上下文、artifact、异常恢复。
- [x] 补第一批 query 样本入口。
- [x] 确认哪些 case 回流 AV2-W13 benchmark / AV2-W12 Gemini eval / AV2-W06 GLM eval matrix。
