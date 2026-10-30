---
id: AV2-W07
title: 实习生跑 skill tracking
status: doing
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-11
updated_at: 2026-05-27
---

# AV2-W07 实习生跑 Skill Tracking

## Scope

追踪实习生跑 skill 的任务分配、batch、case、输出、review 结论和回流动作。

## Why It Matters

实习生 run 的价值取决于结果能不能被 review、分类、复用。如果记录不统一，很快会变成散落的聊天和日志。

## Batch Link

主表见：[Intern skill runs](../runs/intern-skill-runs.md)

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Batch completion | run ledger | 每批 case 有明确完成状态 |
| Review coverage | reviewer notes | 关键输出都被看过 |
| Issue classification | run ledger | 失败原因能归类 |
| Reuse rate | eval / workstream links | 高价值 case 回流到 eval 或 prompt/tool 修改 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27 | recent-master, tab-web | 面试新实习生；给实习生 assign deep research skill task | 新实习生可以承担 deepresearch skill 调研和 case 整理，帮助后续 skill 融合 | 已完成面试和任务分配；输出待收 | assigned | keep tracking | 收 deepresearch skill 调研输出、阻塞、下一步和可回流 benchmark case |
| 2026-05-18 | recent-master, tab-web | 21 号前持续 track 实习生 skill 工作 | 稳定收集 skill run 输出和工程卡点，可以给 GLM 替换、query 复杂化、skill 流程改造提供真实样本 | 待收今日进度、输出、阻塞、下一步 | pending | keep tracking | 更新 run ledger，按工程卡点 / artifact / follow-up 归类 |
| 2026-05-14 | recent-master, tab-web | 实习生工作基本 align：跑 skill 找工程卡点；今日目标 24 个，节奏 30 分钟 4 个 | 对齐工作边界和节奏后，输出更容易被 review、分类、回流 | 已确认工作定位和今日节奏；case 输出待收集 | aligned / running | keep tracking | 收集 24 个 case 输出，按工程卡点分类，评估 URL -> create skill 后端流程 |
| 2026-05-12 | recent-master, tab-web | 已给实习生交代 skill run 任务 | 统一 batch 任务能帮助收集更多 skill 场景反馈 | 待收集：case 输出、失败分类、是否回流 workstream | pending | keep running | 收集 batch 输出并 review |
| 2026-05-11 | TBD | Tracking created | 统一记录格式能降低 review 和复盘成本 | TBD | TBD | TBD | 建第一批 batch |

## Open Risks

- 只记录 pass/fail，不记录原因，无法回流。
- Runner 输出和 reviewer 结论混在一起，后续难复盘。
- Case 没有链接到具体 workstream，调优动作断掉。
- 产品沟通占用较多 effort，需要把结论尽快转成工程卡点和可执行 follow-up。

## Next Actions

- [x] 给实习生交代 skill run 任务。
- [x] 记录 2026-05-14 实习生进度、输出、阻塞、下一步。
- [x] 定义/补齐第一批 skill run batch 详情。
- [ ] 给每个 batch 指 owner / reviewer。
- [ ] 规定输出 artifact 格式。
- [ ] 把失败 case 链接回 prompt/tool/model/skill workstream。
- [ ] 收集今日 24 个 skill run 的具体 case 输出和工程卡点分类。
- [ ] 评估后端直接给 URL -> create skill 流程的开发时间和工程影响。
- [ ] 记录 2026-05-18 实习生 skill 工作进度、输出、阻塞、下一步。
- [ ] 将可用 case 回流到 AV2-W09 query 复杂化 pipeline 或 AV2-W06 GLM eval matrix。
- [x] 面试新实习生。
- [x] 给实习生 assign deep research skill task。
- [ ] 收 deepresearch skill 调研输出并回流 AV2-W11 / AV2-W13。
