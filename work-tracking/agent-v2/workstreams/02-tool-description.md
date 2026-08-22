---
id: AV2-W02
title: Tool description 修改
status: validating
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-05-19
---

# AV2-W02 Tool Description 修改

## Scope

追踪工具 description、参数说明、使用边界、失败提示、选择策略等变更。

## Why It Matters

Tool description 会直接影响 agent 是否知道该不该用工具、何时用、如何填参数，以及失败时是否能恢复。

## Tool Inventory

| Tool | Repo | Change Type | Expected Behavior | Risk | Link |
| --- | --- | --- | --- | --- | --- |
| platform tool mapping | recent-master, tab-web | selection boundary / examples | 模型知道用户前端、E2B、用户本地文件分别对应哪些工具和不可跨用的边界 | 写得太宽会导致跨平台误用，写得太窄会导致漏用可用工具 | [AV2-W10](10-platform-tool-mapping-prompt.md) |
| search_in_file | recent-master | query normalization | 支持 stringified JSON query arrays，减少模型把数组字符串化后的搜索失败 | 仍需回归正常 string、array、stringified JSON array 三类输入 | feat(agent-v2): tighten GLM workflows and artifact paste handling |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Correct tool selection | trace review / eval | 该用时用，不该用时不用 |
| Argument validity | tool logs | 参数错误率下降 |
| Redundant calls | traces | 重复/无效调用减少 |
| Failure recovery | traces | tool fail 后能换策略 |
| UI consistency | tab-web review | 用户看到的行为和 tool 语义一致 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-19 | recent-master, tab-web | 平台 -> 工具映射 prompt / tool boundary 梳理 | 工具选择不仅取决于任务，还取决于运行平台；显式说明能减少前端、E2B、本地文件之间的错选 | 待回归平台正例、负例、跨平台混合任务 | pending | scope mapping | 建 platform x tool matrix，补 tool path drift case |
| 2026-05-18 | recent-master | `search_in_file` handle stringified JSON query arrays | GLM / agent 可能把 query array 以字符串形式传入，normalize 后能降低 file search 参数错误率 | 已扩展 file search query normalization 单测 | unit covered | keep validating | 回归正常 string、array、stringified JSON array 和异常输入 |
| 2026-05-12 | recent-master, tab-web | Tool description 相关主要改动已推 | 更清晰的 description 应减少错选、漏选和无效参数 | 待回归：tool 选择路径、参数稳定性、常规 case 是否偏离 | pending | 进入 validating | 和 SP prompt 一起看路径差异，必要时拆分归因 |
| 2026-05-11 | TBD | Tracking created | 先建立 tool description 追踪口径 | TBD | TBD | TBD | 补 tool 清单和最新 PR |

## Open Risks

- Description 写得太宽会导致滥用 tool。
- Description 写得太窄会导致 agent 漏用关键 tool。
- 参数语义如果和前端/后端实现不一致，会造成调优假象。

## Next Actions

- [x] 列出这版涉及修改的 tool。
- [x] 每个 tool 写一句「希望 agent 行为怎么变」。
- [ ] 回归该调用、不要调用、调用失败后恢复三类 case。
- [ ] 补平台 x 工具矩阵：用户前端 / E2B / 用户本地文件。
- [ ] 记录分数、轮次、时间和 tool path drift。
