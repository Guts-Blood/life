---
id: AV2-W06
title: GLM 换模型和 prompt
status: shipping
owner: 我
repos:
  - recent-master
created_at: 2026-05-11
updated_at: 2026-06-05
---

# AV2-W06 GLM 换模型和 Prompt

## Scope

追踪 GLM 模型切换、模型配置、prompt 适配、对比实验、成本/延迟/质量变化和上线决策。

## Model Board

| Item | Baseline | Candidate | State | Evidence |
| --- | --- | --- | --- | --- |
| Model | Sonnet agent v2 mainline | GLM system prompt config | active | executor/fallback defaults 已切到 GLM system prompt config，待 before/after 回归 |
| Prompt | Sonnet mainline prompt | tightened GLM browser-use prompts | active | 强化 action cadence、snapshot/file search、recovery、completion ledgers |
| Routing / config | 主链路 Sonnet 配置 | executor/fallback defaults -> GLM | shipping | 2026-05-21 今日目标：合入 master 并发版 |
| Eval set | Sonnet mainline regression cases | GLM release smoke + before/after matrix | shipping | 覆盖常规 case、doc/sheet、UID / 输出格式、tool calling、action cadence、recovery、platform tool mapping、release smoke |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Quality | eval / manual review | 关键任务不回退 |
| Tool use | traces | tool call 行为稳定或变好 |
| Latency | logs | 延迟可接受 |
| Cost | billing / estimate | 成本可接受 |
| Style / instruction following | review | 和 agent_v2 预期一致 |
| Regression | golden cases | 老能力不被模型切换破坏 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-05 | recent-master | SQL 模型配置和 settings 配置到线上，支撑 AV2-W14 生图上线收口 | 把模型/settings 配置走线上配置后，生图默认模型/参数可以灰度、回滚和验证，减少硬编码或环境漂移风险 | 待验证 settings 生效、SQL config、生图 smoke、回滚路径 | todo | configure online | 记录配置版本、生效范围、灰度/回滚和 smoke 结果 |
| 2026-05-21 | recent-master | GLM 替代合入 master 并发版 | GLM 替代进入主链路后，可以作为下一阶段 Gemini / benchmark / skill 迭代的稳定 baseline | 待记录 master merge、release、smoke、rollback note | pending | ship today | 合入 master、发版、记录 release evidence 和观察项 |
| 2026-05-19 | recent-master, tab-web | browser-use 迭代方向：按 UID 查询 APC 细节，支持渐进式网页内容披露 | GLM 在网页操作前如果能按需查看少量 UID 的 APC 细节，应该能提高点击/输入置信度，同时避免全量 snapshot 爆上下文 | 待设计工具和回归 case | idea | include in workflow eval | 加入 snapshot -> UID detail -> action 的 GLM workflow 回归 |
| 2026-05-19 | recent-master, tab-web | GLM prompt 需要纳入平台 -> 工具映射：用户前端 / E2B / 用户本地文件 | GLM 在 tool choice 上需要显式平台上下文，否则容易把前端工具、E2B 工具、本地文件工具混用 | 待加入 GLM before/after case | pending | include in GLM eval | 把 AV2-W10 的平台矩阵纳入 GLM 回归 |
| 2026-05-18 | recent-master | feat(agent-v2): tighten GLM workflows and artifact paste handling | 切 executor/fallback defaults 到 GLM system prompt config，并强化 browser-use prompt 后，GLM 主链路执行节奏、检索、恢复和 completion ledger 应更稳定 | 已补 prompt/workflow 改动；待 GLM before/after 回归 | pending | keep iterating | 跑 GLM 回归：action cadence、snapshot/file search、recovery、completion ledger、UID / 输出格式 |
| 2026-05-18 | recent-master | Sonnet agent v2 已合入主链路；21 号前主要任务转为迭代和替换模型为 GLM | 以 Sonnet mainline 作 baseline，可以更清楚判断 GLM 的质量、格式稳定性、tool calling 和延迟差异 | 待建立 GLM eval matrix：常规 case、doc/sheet、UID / 输出格式、tool calling、轮次、时间 | pending | prioritize GLM replacement | 明确 baseline / candidate / eval set / rollout guardrail |
| 2026-05-14 | recent-master | 修复 UID 模型幻觉和输出格式 bug，并加工程兜底 | 格式约束和工程兜底可以降低模型幻觉导致的错误输出 | 测试同学回归未跑完 | pending | keep validating | 等回归结果，重点看 UID 稳定性、输出格式和兜底触发 |
| 2026-05-12 | recent-master | 模型/prompt 相关主要改动已推 | GLM 和 prompt 适配应在目标场景保持或提升质量 | 待回归：分数、轮次、时间、tool calling 稳定性 | pending | 进入 validating | 跑常规 case 与 doc/sheet case，拆分模型/prompt 风险 |
| 2026-05-11 | TBD | Tracking created | 模型切换需要和 prompt 适配一起追，但验证时尽量拆因子 | TBD | TBD | TBD | 补模型版本、prompt diff、eval 结果 |

## Open Risks

- 模型和 prompt 同时变化，效果难归因。
- GLM 在 tool calling、长上下文、格式稳定性上可能有特定行为，需要 case 覆盖。
- UID 或结构化字段被模型幻觉时，需要同时依赖 prompt 约束、解析校验和工程兜底。
- Sonnet 已经进入主链路后，GLM 替换需要明确 guardrail，否则线上主链路问题会被误归因到其他 agent_v2 改动。
- GLM 发版如果缺少 release evidence、冒烟结论和 rollback note，会影响后续问题归因。
- 质量提升如果伴随延迟或成本明显上升，需要上线策略。
- SQL 模型配置/settings 如果没有记录线上生效范围和回滚路径，生图默认模型/参数问题会被误归因到 skill 或 runtime。

## Next Actions

- [ ] 记录 Sonnet mainline baseline 和 candidate GLM 版本。
- [x] 记录 GLM prompt / config / routing 适配 diff。
- [ ] 建 GLM before/after eval matrix。
- [ ] GLM 替代合入 master。
- [ ] GLM 替代发版。
- [ ] 记录 release smoke、观察项和 rollback note。
- [ ] 标注必须通过的红线 case。
- [ ] 记录最终评估分数、轮次、时间。
- [ ] 回归 UID 幻觉和输出格式 bug 的修复效果。
- [ ] 回归平台 -> 工具映射 prompt 是否改善 GLM tool choice。
- [ ] 回归 snapshot -> UID APC detail -> action 是否提升网页操作置信度。
- [ ] 决定 GLM 替换策略：全量、灰度、开关或实验分流。
- [ ] 记录生图相关 SQL 模型配置/settings 的线上版本、生效范围、验证和回滚路径。
