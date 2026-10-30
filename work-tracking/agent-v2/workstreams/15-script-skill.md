---
id: AV2-W15
title: 脚本 skill
status: online
owner: 我
repos:
  - recent-master
created_at: 2026-05-27
updated_at: 2026-06-08
---

# AV2-W15 脚本 Skill

## Scope

新增脚本 skill，追踪脚本类任务的识别、规划、生成/执行边界、产物交付、失败恢复、安全约束和 benchmark case。2026-06-08 最新状态：脚本 skill 已上线，后续补线上 smoke、典型 case 和安全/执行边界记录。

## Why It Matters

脚本能力可以把很多重复、批处理、文件处理、数据转换和自动化任务标准化。风险也更高：需要明确什么时候只是生成脚本，什么时候可以执行脚本，执行环境、文件权限、副作用和可复现性如何管理。

## Skill Board

| Area | Current State | Needed For Ship | Notes |
| --- | --- | --- | --- |
| Trigger | online / needs smoke | 能识别脚本生成 / 脚本执行 / 批处理任务 | 补典型线上 case |
| Execution boundary | online / needs backfill | 明确可执行环境、权限、读写边界、安全策略 | 补边界记录 |
| Artifact | online / needs smoke | 脚本文件、运行日志、输出结果可交付 | 补线上 smoke |
| Recovery | online / needs cases | 运行失败能诊断、修改、重试 | 补失败恢复 case |
| Benchmark | needs backfill | 覆盖生成、执行、失败恢复和文件副作用 case | 回流 AV2-W13 |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Correct trigger | traces / review | 脚本类任务能被识别，非脚本任务不误触 |
| Safety boundary | logs / policy review | 不越权读写，不执行高风险操作 |
| Output usefulness | artifact review | 脚本和输出能被用户复用 |
| Reproducibility | run logs | 输入、命令、输出可复查 |
| Failure recovery | failure cases | 报错后能定位、修复、重跑 |
| Online health | online smoke / traces | 脚本 skill 在线上可触发、产物可交付、失败态可解释 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-08 | recent-master | 脚本 skill 已上线 | 脚本类任务已从 first-priority 进入线上验证阶段，接下来重点是补 smoke、典型 case 和边界证据 | online; smoke 待补 | online | move to follow-up | 记录线上 smoke、典型 case、执行/安全边界和 benchmark 回流 |
| 2026-05-27 | recent-master | 新增 first priority：脚本 skill | 把脚本类任务沉淀成 skill，可以提升批处理/自动化/文件处理任务稳定性 | 待设计 skill contract 和安全边界 | pending | first priority | 定义 trigger、执行边界、artifact、失败恢复和 benchmark case |

## Open Risks

- 生成脚本和执行脚本的边界不清，会带来权限和副作用风险。
- 如果没有运行日志和输出 artifact，结果不可复现。
- 失败恢复不标准化时，模型会在 debug 中消耗过多轮次。
- 已上线后如果不补线上 smoke、典型 case 和失败样式记录，后续效果或安全问题会难以归因。

## Next Actions

- [x] 脚本 skill 已上线。
- [ ] 补线上 smoke：触发、生成/执行、日志、产物、失败态。
- [ ] 回填脚本 skill 的适用任务和非适用任务。
- [ ] 回填生成脚本 vs 执行脚本的边界。
- [ ] 回填输入输出：脚本文件、命令、日志、产物、错误。
- [ ] 回填安全约束：文件读写、网络、环境变量、危险命令。
- [ ] 补生成、执行、失败恢复、权限边界 benchmark case。
