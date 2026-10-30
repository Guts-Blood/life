---
id: AV2-W04
title: 老 tool 点击 feedback 增加
status: validating
owner: 我
repos:
  - tab-web
  - recent-master
created_at: 2026-05-11
updated_at: 2026-05-12
---

# AV2-W04 老 Tool 点击 Feedback 增加

## Scope

追踪老 tool 的点击 feedback 增加：前端事件、payload、后端消费、数据质量、调优使用方式和上线验证。

## Why It Matters

点击 feedback 只有进入效果调优闭环才有价值。这个 workstream 的核心是确认 feedback 不只是被记录，而是能帮助判断 tool 行为是否正确。

## Feedback Contract

| Field / Event | Source | Consumer | Required? | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Tool id | tab-web | recent-master | yes | TBD | TBD |
| Click action | tab-web | recent-master | yes | TBD | 点击、取消、重试、展开等需要确认 |
| Context / task id | tab-web | recent-master | yes | TBD | 用于关联 trace / session |
| Outcome | tab-web / backend | recent-master | TBD | TBD | 成功、失败、timeout、user ignored |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Feedback coverage | event logs | 老 tool 关键点击都有记录 |
| Payload completeness | logs / schema checks | 能关联 task、tool、session、结果 |
| Signal usability | tuning review | 能反向解释 agent/tool 行为 |
| Event noise | analytics | 噪音可控，不误导优化 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-12 | tab-web, recent-master | 老 tool click feedback 相关主要改动已推 | feedback 能形成调优信号，同时不改变常规 tool 路径 | 待回归：事件可用性、常规 case path drift、tool 成功/失败归因 | pending | 进入 validating | 用真实 case 验证 feedback 能关联 trace/session |
| 2026-05-11 | TBD | Tracking created | 点击 feedback 需要从事件走到调优信号 | TBD | TBD | TBD | 补事件 schema 和消费逻辑 |

## Open Risks

- 只记录 click，不记录 context，后续无法归因。
- 前端有事件，后端没有消费，形成断链。
- feedback 语义不清，容易把用户探索行为误判成 tool 效果问题。

## Next Actions

- [ ] 确认老 tool 范围。
- [ ] 写清楚 feedback event schema。
- [ ] 补 tab-web PR / recent-master PR 链接。
- [ ] 找一条真实 trace 验证 feedback 能被使用。
- [ ] 确认 feedback 改动没有造成常规 case 路径偏离。
