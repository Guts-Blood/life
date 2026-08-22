---
id: AV2-W01
title: SP prompt
status: validating
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-05-19
---

# AV2-W01 SP Prompt

## Scope

追踪 agent_v2 的 system prompt / SP prompt 改动，包括 baseline、diff、调优假设、eval 结果、上线结论和回归风险。

## Why It Matters

SP prompt 是 agent 行为的最大杠杆之一。它会影响任务分解、工具选择、用户沟通、失败恢复、长任务坚持度和安全边界。

## Current State

- Baseline: TBD
- Latest prompt branch / PR: 2026-05-19 新增平台 -> 工具映射 prompt 调优，待补具体 PR / branch
- Main target behavior: agent_v2 在常规 case 中保持原有路径不偏离，并能根据用户前端、E2B、用户本地文件选择正确工具面
- Known regression risk: SP prompt 与 tool description / skill router / model prompt 同时变化，尤其要防止平台工具映射造成路径串台

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Task success rate | eval / manual cases | agent_v2 相对 baseline 提升或不回退 |
| Tool selection quality | traces / reviewer notes | 少错选、少漏选、少无意义调用 |
| Recovery behavior | failure cases | 出错后能解释、重试或换路径 |
| User-facing quality | manual review | 更清楚、更主动、更少空话 |
| Regression cases | golden cases | 老能力不被 prompt 改动破坏 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-19 | recent-master, tab-web | 新增平台 -> 工具映射 prompt 调优：用户前端 / E2B / 用户本地文件 | 明确平台和工具边界后，模型能更稳定地选择正确工具面，减少前端、E2B、本地文件上下文混用 | 待回归：三类平台正例、负例、混合上下文、多轮切换 | pending | scope prompt diff | 写平台工具矩阵和回归 case，链接 AV2-W10 |
| 2026-05-12 | recent-master, tab-web | 主要 prompt 相关改动已推 | 新版 prompt 应该支持 agent_v2 上线目标，同时不破坏常规 case 路径 | 待回归：平常 case 路径偏离、分数、轮次、时间 | pending | 进入 validating | 跑常规 case before/after，记录偏离点 |
| 2026-05-11 | TBD | Tracking created | 先建立 SP prompt 追踪口径 | TBD | TBD | TBD | 补 baseline prompt、最新 diff、目标 case |

## Open Risks

- Prompt 和 tool description / model 同时改时，效果变化难归因。
- Prompt 变强可能带来更长响应或更多 tool call。
- 平台工具映射如果过宽，会让模型在用户前端、E2B、本地文件之间错误迁移操作。
- 少数 benchmark 变好不代表真实任务变好，需要覆盖长任务和失败恢复。

## Next Actions

- [ ] 贴当前 baseline prompt 或链接。
- [ ] 贴 agent_v2 最新 prompt diff 或 PR。
- [x] 将本阶段状态切到回归验证。
- [ ] 跑常规 case，记录路径是否偏离。
- [ ] 写平台 -> 工具映射 prompt 矩阵和负例。
- [ ] 回归用户前端、E2B、用户本地文件三类平台工具选择。
- [ ] 记录最终评估分数、轮次、时间。
