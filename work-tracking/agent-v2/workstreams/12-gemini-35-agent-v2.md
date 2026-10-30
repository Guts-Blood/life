---
id: AV2-W12
title: Gemini 3.5 Flash 调效果+合入 Agent V2
status: validating / SOTA
owner: 我
repos:
  - recent-master
created_at: 2026-05-21
updated_at: 2026-05-27
---

# AV2-W12 Gemini 3.5 Flash 调效果+合入 Agent V2

## Scope

追踪 Gemini 3.5 Flash 在 agent v2 链路里的效果调优、prompt/config 适配、eval 对比、合入策略和 rollback 方案。

## Why It Matters

GLM 替代发版后，需要继续探索更强模型候选。Gemini 3.5 Flash 当前跑分达到 SOTA，但缺点是不输出 content thinking，用户体验需要单独补救或权衡。

## Model Board

| Item | Baseline | Candidate | State | Evidence |
| --- | --- | --- | --- | --- |
| Model | GLM release / Sonnet mainline | Gemini 3.5 Flash | validating / SOTA | 已调试并跑分，目前 SOTA |
| Prompt / config | GLM current prompt/config | Gemini 3.5 Flash adapted prompt/config | validating | 已完成一轮调效果，待补 diff / config 链接 |
| Eval set | AV2-W13 benchmark | Gemini 3.5 Flash score matrix | validating | 跑分当前 SOTA；待补具体结果链接 |
| Rollout | GLM master release | Gemini 3.5 Flash agent v2 integration | scoping | 待确定合入链路、content thinking UX 处理和 rollback |

## Metrics / Signals

| Signal | Source | Target / Interpretation |
| --- | --- | --- |
| Task quality | benchmark / manual review | 相比 GLM/Sonnet 提升或不回退 |
| Content thinking UX | product / manual review | 即使模型不输出 content thinking，也要有可接受的用户可见进度/解释 |
| Tool calling | traces | 工具选择、参数、恢复稳定 |
| Long task handling | traces / reviewer notes | 多步任务不丢目标 |
| Artifact quality | artifact review | 文档、表格、研究类产物可用 |
| Latency / cost | logs / estimate | 上线可接受 |

## Change Ledger

| Date | Repo | Change / PR | Hypothesis | Eval / Signal | Result | Decision | Next |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-27 | recent-master | Gemini 3.5 Flash 调试和跑分完成，目前 SOTA | Gemini 3.5 Flash 在当前 benchmark 上能成为新效果上界候选 | 跑分 SOTA；缺点是不输出 content thinking，用户体验风险存在 | SOTA / UX gap | keep validating | 设计 content thinking / progress UX 补救方案，并准备合入 agent v2 链路 |
| 2026-05-21 | recent-master | 新增 Gemini 3.5 调效果+合入 agent v2 workstream | 用 GLM release 作为近期 baseline，再对 Gemini 3.5 做同口径 eval，可以判断是否值得合入 | 待建 eval matrix 和接入计划 | pending | scope | 补版本/config、prompt diff、benchmark 结果、接入策略 |

## Open Risks

- Gemini 3.5 Flash 的强项/弱项如果不拆场景，会被平均分掩盖。
- 不输出 content thinking 可能降低用户对长任务进度、推理过程和可靠性的感知。
- Prompt/config 与模型同时变化，难以归因。
- 接入 agent v2 链路后可能影响 tool calling、artifact 或长任务稳定性。

## Next Actions

- [x] 确认 Gemini 3.5 Flash 作为候选模型并完成调试/跑分。
- [x] 建 Gemini 3.5 Flash vs GLM / Sonnet eval matrix 初版。
- [x] 调整 prompt/config，并记录为待补链接。
- [x] 跑 AV2-W13 benchmark 的核心 case，当前 SOTA。
- [ ] 设计 content thinking / 用户体验补救方案。
- [ ] 明确合入策略：实验开关、灰度、master merge、rollback。
