---
id: AV2-W10
title: 平台到工具映射 Prompt
status: scoping
owner: 我
repos:
  - recent-master
  - tab-web
created_at: 2026-05-19
updated_at: 2026-06-05
---

# AV2-W10 平台到工具映射 Prompt

## Scope

调 prompt，明确告诉模型在不同平台/环境下应该使用哪类工具：用户的前端、E2B、用户的本地文件。目标是让模型先判断当前操作面，再选择工具，而不是只根据任务语义盲选工具。

## Why It Matters

同一个用户意图在不同平台上对应的可用工具和副作用不同。前端操作、E2B 沙箱、用户本地文件如果被模型混用，会造成路径偏离、文件上下文错误、不可见状态误判，甚至让调优结果失真。

## Platform Tool Matrix

| Platform / Context | Tool Surface | Model Should Do | Model Should Avoid | Examples / Cases |
| --- | --- | --- | --- | --- |
| 用户的前端 | TBD | 使用面向用户 UI / 前端状态的工具，关注用户可见状态和交互反馈 | 不把前端可见状态当成本地文件或 E2B 文件状态 | TBD |
| E2B | TBD | 使用沙箱执行 / 远端环境相关工具，关注 sandbox 内的文件、命令和 artifact | 不假设 E2B 文件就是用户本地文件 | TBD |
| 用户的本地文件 | TBD | 使用本地文件 / workspace 相关工具，关注真实本地路径、读写边界和已有文件 | 不通过前端或 E2B 路径替代本地文件读写 | TBD |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Correct platform detection | traces / reviewer notes | 模型能先识别当前任务所在平台 |
| Correct tool choice | traces | 工具面和平台一致，不串台 |
| Boundary compliance | negative cases | 明确不可跨用时不会强行调用错误工具 |
| Recovery | failure cases | 发现平台不匹配后能改用正确路径 |
| Regression | golden cases | 常规 tool use 不因为新增提示而变慢或过度询问 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-05 | recent-master, tab-web | 生图放到 E2B 链路，作为 AV2-W14 最新上线收口 todo | 生图明确走 E2B 执行链路，可以减少前端状态、E2B 文件和用户本地文件之间的混用 | 待验证 E2B route、artifact 回传、失败态和前端展示 | todo | route image gen through E2B | 补生图 E2B 正负例，并回流 W14 smoke |
| 2026-05-19 | recent-master, tab-web | 新增平台 -> 工具映射 prompt 工作 | 显式 platform x tool matrix 能减少用户前端、E2B、用户本地文件之间的工具错选 | 待建立 prompt diff 和回归 case | pending | scope prompt | 补工具矩阵、正负例、混合场景，并回流 AV2-W01 / AV2-W02 / AV2-W06 |

## Open Risks

- 平台定义如果不够具体，模型仍然可能只按任务语义选择工具。
- 映射过硬可能导致模型在跨平台任务中不敢切换工具。
- prompt、tool description、GLM 替换同时变化时，需要用 trace 拆分归因。
- 生图放到 E2B 链路后，如果 artifact 回传和用户可见 surface 没有明确，会出现 E2B 成功但前端/agent 看不到结果。

## Next Actions

- [ ] 列出用户前端、E2B、用户本地文件分别可用的工具面。
- [ ] 写平台判断规则：模型如何从任务上下文识别当前平台。
- [ ] 写正例：每个平台应该调用什么工具。
- [ ] 写负例：每个平台不应该调用什么工具。
- [ ] 加入 GLM before/after 回归矩阵。
- [ ] 回归混合场景：前端触发、E2B 执行、本地文件产物读取。
- [ ] 补生图 E2B 链路 case：agent 触发、E2B 生成、artifact 回传、前端展示。
