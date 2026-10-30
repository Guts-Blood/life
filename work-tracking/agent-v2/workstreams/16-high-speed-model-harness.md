---
id: AV2-W16
title: 高速模型 harness / 换模型
status: postponed / blocked-by-qwen-support
owner: 我
repos:
  - recent-master
created_at: 2026-06-08
updated_at: 2026-06-16
---

# AV2-W16 高速模型 Harness / 换模型

## Scope

换模型调一个统一 harness，候选包括 Qwen 3.7 Plus 或其他高速模型。目标是用同一口径比较质量、延迟、成本、tool calling、长任务稳定性和 rollout 风险，而不是只看单次跑分。2026-06-16 最新优先级：公司侧暂时不支持 Qwen，因此 auto harness / 高速模型切换先往后放；当前只保留设计草稿，等模型支持或替代候选明确后再恢复。

## Why It Matters

模型切换会同时影响效果、速度、成本、tool 调用、artifact 质量和用户可见体验。高速模型如果能在关键场景不回退，同时显著改善延迟，就可以作为 agent v2 后续主链路或分场景路由候选；但如果没有 harness，很容易把模型问题、prompt 问题和 skill/runtime 问题混在一起。

## Harness Board

| Area | Current State | Needed For Ship | Notes |
| --- | --- | --- | --- |
| Candidate models | blocked | Qwen 3.7 Plus / 其他高速模型候选列表 | 公司侧暂不支持 Qwen，候选未具备推进条件 |
| Eval matrix | parked | 覆盖 benchmark、tool calling、artifact、long task、auto-skill、image/script skill | 回流 AV2-W13，等恢复后继续 |
| Metrics | parked | quality / latency / cost / rounds / tool errors / artifact quality | 保留口径草稿 |
| Harness runner | postponed | 可批量跑候选模型并记录结果 | 暂不作为当前 P0 |
| Rollout strategy | postponed | 实验开关、灰度、回滚、分场景路由 | 等模型支持后再细化 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-16 | recent-master | 优先级调整：公司侧暂时不支持 Qwen，auto harness 往后放 | 在模型平台支持不足时继续推进 harness 会卡在前置条件上；应把 P0 集中到 auto-skill pipeline | company Qwen support unavailable | postponed | park until support | 保留设计草稿；等 Qwen 支持或替代高速模型候选明确后恢复 |
| 2026-06-08 | recent-master | 新主要 priority：换模型调高速模型 harness，候选 Qwen 3.7 Plus 或其他高速模型 | 用统一 harness 比较高速模型，可以更快找到质量不回退且延迟更优的 agent v2 候选 | 待建 harness、候选模型矩阵和首批 eval | first-priority | build harness | 定义候选、case、指标、runner、灰度/回滚策略 |

## Open Risks

- 只看平均质量分会掩盖 tool calling、artifact、长任务和延迟问题。
- 候选模型如果和 prompt/config 同时变化，结果会难归因。
- 高速模型可能在简单任务表现好，但复杂 tool / skill / artifact 场景回退。
- 没有灰度和回滚策略时，模型切换会放大线上风险。
- 当前最大阻塞不是 harness 设计，而是公司侧暂时不支持 Qwen；继续作为 P0 会挤占 auto-skill pipeline。

## Next Actions

- [x] 记录优先级后置原因：公司侧暂时不支持 Qwen。
- [ ] 保留候选模型和 harness eval matrix 草稿，暂不投入实现。
- [ ] 等 Qwen 支持或替代高速模型候选明确后，恢复候选、runner、指标和灰度/回滚设计。
