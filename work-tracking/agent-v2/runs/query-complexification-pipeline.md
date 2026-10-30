---
id: query-complexification-pipeline
title: 用户 query 挖掘到复杂化 pipeline
created_at: 2026-05-18
updated_at: 2026-05-27
---

# Query Complexification Pipeline

## Board

| ID | Stage | Source | Raw Query / Signal | Cluster | Complexity Delta | Target Workstream | Owner | Reviewer | State | Next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QC-20260527-001 | Backlog / Ship | query mining + user profile analysis | 生成 benchmark query pipeline | persona-grounded benchmark | 用户画像 + query 挖掘 -> benchmark query | AV2-W13 / AV2-W12 | 我 | TBD | completed / feeding benchmark | 继续扩充更多更深度 case |
| QC-20260518-001 | Intake | TBD | TBD | TBD | TBD | AV2-W06 / AV2-W07 / AV2-W09 | 我 | TBD | planned | 补第一批真实 query 来源和字段 |
| QC-20260521-001 | Intake | benchmark planning | 更深度 agent v2 case 来源待补 | benchmark / deep cases | 多步骤、多约束、跨工具、长上下文、artifact 质量 | AV2-W13 | 我 | TBD | planned | 回流到 benchmark deep cases board |

## Stage Definitions

| Stage | Required Fields | Quality Gate |
| --- | --- | --- |
| Intake | source, raw query, context, date | 能解释用户真实意图和场景 |
| Normalize | canonical query, cluster, domain, ability labels | 去重后仍有代表性 |
| Complexity Design | complexified query, complexity delta, acceptance criteria | 新增约束真实、可执行、可验证 |
| Review | priority, target workstream, owner, reviewer | 明确回流对象和验收人 |
| Run / Evaluate | model / skill, trace, artifact, pass/fail, issue type | 能支持 GLM / Sonnet / skill 对比 |
| Backlog / Ship | follow-up, PR / issue / eval link, decision | 有工程动作或明确丢弃理由 |

## Complexity Rubric

| Dimension | Meaning | Examples |
| --- | --- | --- |
| Multi-step | 需要规划和阶段性执行 | 先收集、再整理、最后生成 artifact |
| Multi-constraint | 同时满足格式、内容、时间、偏好等限制 | 指定输出格式、字数、数据口径 |
| Tool / skill use | 需要调用工具或 skill | 文档、表格、浏览器、文件处理 |
| Long-context | 依赖已有材料或多轮上下文 | 从历史记录、PR、日志里综合判断 |
| Recovery | 需要处理失败、缺字段、模糊需求 | 缺数据时提出假设或降级方案 |

## Review Checklist

- [x] 每条 query 有真实来源或明确假设来源。
- [x] 每条复杂化 case 有验收标准。
- [x] 每条 case 标注目标能力和目标 workstream。
- [x] 可跑 case 能进入 GLM eval matrix、Gemini eval matrix、benchmark deep cases 或 skill run ledger。
- [ ] 不采用的 query 有丢弃原因。
