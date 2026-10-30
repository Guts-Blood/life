---
id: AV2-W03
title: 新加 tool 修改
status: validating
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-05-19
---

# AV2-W03 新加 Tool 修改

## Scope

追踪 agent_v2 新增 tool 的定义、接入、前端展示、权限/安全、验收 case 和上线状态。

## New Tool Board

| Tool | Repo | Owner | State | Purpose | Launch Criteria | Link |
| --- | --- | --- | --- | --- | --- | --- |
| UID -> APC detail lookup tool | recent-master, tab-web | 我 | scoping | 在 `take_snapshot` 只返回 UID 和简略内容后，允许模型按少量 UID 拉取对应 APC 细节，渐进式披露网页内容 | 输入少量 UID 能返回可操作 APC；上下文预算可控；UID 过期/不存在有清晰失败态；能提升点击/输入前置信度 | iteration idea 2026-05-19 |
| 复制产物->粘贴一体化工具 | recent-master, tab-web | 我 | validating | 把复制产物和粘贴动作串成一体化工具链路，减少手动断点 | 目标场景成功；非目标场景不误触；spreadsheet paste payload 不注入额外标题/表头；merged-cell clipped value 保留；CSV/TSV cap 有 warning；失败可恢复 | feat(agent-v2): tighten GLM workflows and artifact paste handling |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Adoption in target cases | traces | 目标场景中能被正确调用 |
| False positive calls | traces | 非目标场景不误触发 |
| Tool success rate | logs | 成功率可接受，失败可解释 |
| UX clarity | tab-web review | 用户能理解 tool 状态和结果 |
| Rollback readiness | PR / config | 出问题时可关闭或回滚 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-19 | recent-master, tab-web | 迭代方向：新增按 UID 查看具体 APC 内容的工具 | `take_snapshot` 全量返回 APC 会爆上下文，只返回 UID/简略内容又会降低操作置信度；按需展开少量 UID 可以渐进式披露网页内容 | 待设计 tool contract：输入 UID 列表、返回 APC 细节、预算上限、失败态 | idea | scope tool | 设计 `get_apc_by_uid` / `inspect_uid_apc` 类工具，补正负例和回归 case |
| 2026-05-18 | recent-master, tab-web | 改进 spreadsheet artifact paste payloads：避免注入 header/title，保留 clipped merged-cell values，校验显式 spreadsheet types，CSV/TSV capped data 给 warning | 更干净的 paste payload 能减少用户可见 artifact 污染和 spreadsheet 场景误判 | 已扩展 clipboard paste payload 单测 | unit covered | keep validating | 回归 spreadsheet paste 正例、merged cell、非 spreadsheet type、CSV/TSV cap warning |
| 2026-05-12 | recent-master, tab-web | 新增复制产物->粘贴一体化工具 | 一体化工具能减少复制/粘贴链路中的中断和手动成本 | 待回归：目标路径成功率、非目标误触发、轮次、耗时 | pending | 进入 validating | 补 PR 链接并跑复制/粘贴相关 case |
| 2026-05-11 | TBD | Tracking created | 新 tool 需要单独验收，避免混在 prompt 调优里 | TBD | TBD | TBD | 补新 tool 清单 |

## Open Risks

- 新 tool 缺少负例 case，会在非目标任务中被误用。
- 新 tool 前后端状态不一致，会影响用户信任和调优判断。
- 新 tool 和旧 tool overlap 时，需要明确优先级。
- UID -> APC detail lookup 如果没有数量/大小上限，仍然会变成上下文爆炸；如果失败态不清，会让模型误以为 UID 对应内容不存在。

## Next Actions

- [x] 列出本版新增 tool：复制产物->粘贴一体化工具。
- [ ] 设计 UID -> APC detail lookup 的输入输出 contract、批量上限和失败态。
- [ ] 回归 `take_snapshot` -> 选择 UID -> 拉 APC 细节 -> 再操作 的渐进式链路。
- [ ] 补正例、负例、失败恢复 case。
- [ ] 回归 spreadsheet artifact paste payload 的 header/title、merged cell、type validation、CSV/TSV cap warning。
- [ ] 确认 tab-web 交互、状态和失败态是否完整。
- [ ] 回归轮次、耗时和误触发。
