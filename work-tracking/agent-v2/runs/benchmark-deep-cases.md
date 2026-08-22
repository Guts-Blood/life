---
id: benchmark-deep-cases
title: Benchmark 深度 Case
created_at: 2026-05-21
updated_at: 2026-06-16
---

# Benchmark Deep Cases

## Case Board

| Case ID | Source | Domain | Depth Type | Target Capability | Models / Skills | State | Owner | Reviewer | Next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BDC-20260615-001 | 小红书 skill iteration | xiaohongshu / strong site control | site-control stress case | site-control detection, fallback, browser skill boundary | Xiaohongshu skill / GLM / Gemini 3.5 / Sonnet | planned | 我 | TBD | 记录封控触发样式、可做/不可做任务、失败态和降级策略 |
| BDC-20260608-001 | URL 高频站点挖掘 pipeline | url/site taxonomy | site-frequency -> task-category -> skill mapping | benchmark case mining, skill routing, auto-skill scenario taxonomy | skill TBD / GLM / Gemini 3.5 / Sonnet | planned | 老板 | 我 | 产出高频站点、任务类别、skill candidate、coverage gap 和第一批 eval case |
| BDC-20260527-001 | query mining + user profile analysis pipeline | benchmark query | persona-grounded / deep query | model quality, tool use, long task handling | Gemini 3.5 Flash / GLM / Sonnet | active input | 我 | TBD | 继续扩充并记录跑分结果 |
| BDC-20260521-001 | TBD | TBD | TBD | TBD | GLM / Gemini 3.5 / Sonnet / skill TBD | planned | 我 | TBD | 补第一批更深度 case |

## Case Schema

| Field | Meaning |
| --- | --- |
| Source | query pipeline、实习生 skill run、真实用户反馈、手工构造等 |
| URL / Site | 高频 URL、站点、域名或产品 surface，用于聚类真实任务来源 |
| Task Category | 从 URL / site 抽出的任务类别，用于后续 skill 映射和 benchmark 分层 |
| Task | 原始任务和复杂化后的任务 |
| Depth | 多步骤、多约束、跨工具、长上下文、失败恢复、artifact 质量等 |
| Site Control | 登录、验证码、频控、访问限制、内容不可见、站点强封控等 |
| Expected Artifact | 文档、表格、研究结论、浏览器操作结果等 |
| Eval Signals | score、rounds、time、tool path、issue type、artifact quality |
| Decision | keep / iterate / rollback / add to regression |

## Review Checklist

- [x] 每个 case 有来源和目标能力。
- [ ] 每个 case 有成功标准，不只是一句 query。
- [ ] 每个 case 能记录 score、rounds、time、tool path、artifact quality。
- [ ] case 能用于 GLM / Gemini 3.5 / Sonnet 或 skill 对比。
- [ ] 深度 case 和红线回归 case分开标注。
- [ ] URL 高频站点 case 有任务类别、skill candidate 和 coverage gap。
- [ ] 强封控站点 case 有 block type、可执行边界和降级策略。
