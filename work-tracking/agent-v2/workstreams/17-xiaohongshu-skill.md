---
id: AV2-W17
title: 小红书 skill / 站点封控
status: iterating / blocked-by-site-control
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-06-16
updated_at: 2026-06-16
---

# AV2-W17 小红书 Skill / 站点封控

## Scope

追踪小红书 skill 的迭代、站点封控样式、可执行任务边界、失败态、降级策略和可回流 benchmark case。2026-06-15 最新结论：迭代后发现小红书这个网站封控非常强，需要把它当作高风险站点单独管理，而不是只作为普通浏览 skill case。

## Why It Matters

小红书这类强封控站点很适合作为 URL 高频站点 -> 任务类别 -> skill pipeline 的边界样本。如果 skill 不知道什么时候可做、什么时候该降级，就会在登录、风控、验证码、访问限制和内容不可见里消耗大量轮次，还可能误导用户以为 agent 能稳定完成站内任务。

## Site-Control Board

| Area | Current State | Needed For Usable Skill | Notes |
| --- | --- | --- | --- |
| Access stability | blocked / unstable | 记录登录、验证码、频控、访问限制、内容不可见等触发样式 | 2026-06-15 发现封控非常强 |
| Task boundary | scoping | 区分可执行、部分可执行、不可执行任务 | 需要正负例 |
| Failure detection | scoping | 能识别封控态，并停止无效重试 | 避免轮次浪费 |
| User-facing fallback | scoping | 能解释站点限制，并给出替代路径或人工介入点 | 需要文案/状态 |
| Benchmark case | planned | 把强封控样本回流 AV2-W13，作为 site-control stress case | 和 URL 高频站点 taxonomy 对齐 |
| Skill strategy | scoping | 判断是继续做 skill、做只读/半自动 skill，还是标记为高风险站点 | 待评估 |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Block trigger rate | traces / manual runs | 哪些动作最容易触发封控 |
| Detectability | trace review | agent 能否识别登录、验证码、访问限制、内容不可见 |
| Recovery value | manual review | 重试是否有效；如果无效，是否应该快速降级 |
| Task coverage | benchmark cases | 小红书任务中哪些可稳定做、哪些只能人工介入 |
| User clarity | UX / final answer review | 用户能理解是站点限制，而不是 agent 无故失败 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-15 | recent-master, tab-web | 迭代小红书 skill，发现网站封控非常强 | 小红书不能按普通浏览 skill 乐观处理，需要单独记录站点封控、任务边界和降级策略 | manual iteration; strong site control | blocked / high-risk site | track separately | 收集封控样式，定义可做/不可做任务，补 benchmark stress case 和降级文案 |

## Open Risks

- 如果不识别封控态，agent 会在无效点击、刷新、重试中消耗大量轮次。
- 如果 skill 对小红书能力描述过强，用户会形成错误预期。
- 站点封控样式可能随时间变化，case 需要记录日期、登录态、网络环境和失败截图/trace。
- 高频站点不等于高可自动化站点；小红书可能需要被标记为高价值但高封控风险。

## Next Actions

- [ ] 收集小红书封控触发样式：登录、验证码、频控、访问限制、内容不可见、页面跳转。
- [ ] 定义小红书 skill 任务边界：可执行、部分可执行、不可执行。
- [ ] 设计封控态检测和快速降级策略，避免无效重试。
- [ ] 写用户态失败/降级文案：说明站点限制、可替代路径、需要用户提供的信息。
- [ ] 把小红书加入 AV2-W13 benchmark：site-control stress case、成功标准、失败分类。
