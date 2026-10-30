---
id: AV2-W11
title: DeepResearch skill 调研+合并
status: doing
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-21
updated_at: 2026-05-27
---

# AV2-W11 DeepResearch Skill 调研+合并

## Scope

调研 deepresearch skill 的能力边界、触发契约、产物形态、失败恢复和合并路径，并决定如何合入 agent v2 主链路。

## Why It Matters

DeepResearch 属于高价值长任务 skill。合入时不仅要看是否能完成 demo，还要看信息来源、过程可追踪性、产物质量、预算/轮次、失败态和用户可理解性。

## Integration Board

| Area | Current State | Needed For Merge | Notes |
| --- | --- | --- | --- |
| Skill contract | assigned to intern | 触发条件、输入输出、产物格式、失败态 | 已给实习生 assign deep research skill task |
| Orchestration | assigned to intern | 能分阶段执行、恢复和总结 | 等实习生调研输出 |
| UI / artifact | TBD | 用户能理解进度、引用、结果和失败原因 | 待对齐 tab-web |
| Eval / benchmark | TBD | 至少有正例、边界、失败恢复 case | 回流 AV2-W13 |
| Merge path | TBD | PR / branch / owner / rollback | 待补 |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Trigger precision | traces / manual review | 需要 deepresearch 时触发，不需要时不误触 |
| Source quality | artifact review | 引用和证据可追踪 |
| Synthesis quality | reviewer notes | 能形成有用结论，不只是堆材料 |
| Cycle budget | logs | 轮次、时间、token 可接受 |
| Failure recovery | failure cases | 搜索失败、来源不足、冲突信息能处理 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27 | recent-master, tab-web | 给实习生 assign deep research skill task；后续顺手融合 deepresearch skill | 实习生先调研 contract / case / 风险，可以降低直接融合的未知成本 | 待收实习生输出 | assigned | keep tracking | review 调研输出，确定最小融合路径和 benchmark case |
| 2026-05-21 | recent-master, tab-web | 新增 deepresearch skill 调研+合并 workstream | 先明确 contract 和验收，再合并 skill，可以降低长任务上线风险 | 待调研 skill contract、merge path、回归 case | pending | scope | 补调研结论、合并计划和 benchmark case |

## Open Risks

- DeepResearch 结果看起来完整，但来源、引用和推理过程不可验证。
- 长任务耗时/轮次过高，影响 agent v2 主链路体验。
- Skill 合入后误触发，导致普通 query 变慢或跑偏。

## Next Actions

- [ ] 收实习生 deep research skill 调研输出。
- [ ] 调研 deepresearch skill 当前能力、输入输出和依赖。
- [ ] 写合并计划：branch / PR / owner / rollback。
- [ ] 定义 deepresearch 正例、边界 case、失败恢复 case。
- [ ] 和 tab-web 对齐进度、引用、artifact、失败态展示。
- [ ] 将高价值 case 回流 AV2-W13 benchmark。
