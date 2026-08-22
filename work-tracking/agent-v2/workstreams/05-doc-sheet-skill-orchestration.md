---
id: AV2-W05
title: doc/sheet skill 合入和 orchestration
status: validating
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-05-18
---

# AV2-W05 Doc/Sheet Skill 合入和 Orchestration

## Scope

追踪 document / sheet skills 的合入、触发策略、orchestration、UI 状态、回归 case 和上线风险。

## Why It Matters

doc/sheet skill 是高价值长任务能力。关键不是只把 skill 合进去，而是 agent 能在合适时机触发、跟踪进度、处理失败，并把结果交付清楚。

## Integration Board

| Skill | Repo | Trigger Strategy | UI / State Impact | State | Link |
| --- | --- | --- | --- | --- | --- |
| Document skill | recent-master, tab-web | 新 SOP + 更多场景 router + contract tightening | 待回归 live-state verification、heading semantics、retry budgets | validating | feat(agent-v2): tighten GLM workflows and artifact paste handling |
| Sheet skill | recent-master, tab-web | 新 SOP + 更多场景 router + exact range / live-state contract | 待回归 exact ranges、spreadsheet artifact quality、retry budgets | validating | feat(agent-v2): tighten GLM workflows and artifact paste handling |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Correct trigger | traces / eval | 目标任务触发，普通任务不误触 |
| Orchestration quality | traces | 能分阶段执行和恢复 |
| Artifact quality | manual review | 文档/表格输出可用 |
| UX clarity | tab-web review | 用户能看到进度、结果、失败原因 |
| Regression | golden cases | 原本非 doc/sheet 任务不受干扰 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-18 | recent-master, tab-web | clarify doc/sheet skill contracts around live-state verification, exact ranges, heading semantics, and retry budgets | 更明确的 skill contract 能减少虚假完成、范围误判、heading 语义混乱和无效重试 | 待回归 doc/sheet 正例、边界 case、失败恢复和 artifact quality | pending | keep validating | 跑 live-state、exact range、heading、retry budget case |
| 2026-05-14 | recent-master, tab-web | 表格 skill 昨日更新版初步效果不错；测试同学回归未跑完 | 新 SOP / router 改动可能提升表格场景触发和产物质量 | 初步人工信号 positive；缺正式回归分数、轮次、时间 | partial positive | keep validating | 等测试回归结果，补成功率、产物质量、轮次、时间 |
| 2026-05-14 | recent-master, tab-web | skill 新改动通过 tabhobor 上传，待 merge 到 master branch | merge 后才能稳定进入 master 回归和后续上线验证 | 待验证：master 上 doc/sheet 场景、常规路径、轮次、时间 | pending | merge required | merge 到 master，并补 PR / commit 链接 |
| 2026-05-12 | recent-master, tab-web | doc/sheet skill 更新：新 SOP + 更多场景 router | 新 SOP 和 router 应提升文档/表格场景触发与执行效果 | 待回归：表格/文档场景成功率、产物质量、轮次、时间 | pending | 进入 validating | 重点跑 doc/sheet 场景并记录误触发/漏触发 |
| 2026-05-11 | TBD | Tracking created | skill 合入需要同时追 agent 触发、执行和 UI 体验 | TBD | TBD | TBD | 补合入 PR 和回归 case |

## Open Risks

- Skill 被误触发，导致普通任务变慢或跑偏。
- Skill 触发正确但 orchestration 状态不清，用户体验不好。
- Artifact 生成成功但质量不足，eval 需要人工维度。

## Next Actions

- [x] 记录 skill 已更新新 SOP 和更多场景 router。
- [ ] Merge 通过 tabhobor 上传的 skill 新改动到 master branch。
- [ ] 补 doc/sheet skill 最新 PR / branch / commit。
- [ ] 回归表格/文档场景正例、负例、边界 case。
- [ ] 记录成功率、产物质量、轮次、时间。
- [ ] 等测试同学完成回归后，补表格 skill 最新效果结论。
- [ ] 确认 tab-web 是否有进度/结果/失败展示需求。
- [ ] 回归 live-state verification、exact ranges、heading semantics、retry budgets。
