---
period: 2026-W22
date_range: 2026-05-25 to 2026-05-31
created_at: 2026-05-27
---

# Weekly Review: Agent V2 / 2026-W22

## Completed / Moved

| Workstream | Progress | Evidence / Notes |
| --- | --- | --- |
| AV2-W09 Query pipeline | query 挖掘 + 用户画像分析 -> benchmark query pipeline 完成 | 回流 AV2-W13 |
| AV2-W12 Gemini 3.5 Flash | 调试和跑分完成，目前 SOTA | 风险：不输出 content thinking，用户体验待处理 |
| AV2-W07 Intern skill runs | 面试新实习生，已 assign deep research skill | 待收调研输出 |
| AV2-W11 DeepResearch skill | 进入实习生调研和后续融合阶段 | 回流 benchmark case |
| AV2-W14 Agent 生图 | 生图 skill/mode scope 收敛 | chat 不做生图；agent 通过 skill runtime 加载生图 tool；第一版不做模型选择 |
| AV2-W14 Agent 生图 | 代码已合进 daily | 审核链路有问题，`poresche` 偶发失败，待收集样本和日志 |

## First Priority

| Priority | Focus | Owner | Success Signal |
| --- | --- | --- | --- |
| P0 | Agent 生图 skill/runtime | 我 | daily 可用；审核/`poresche` 偶发失败被定位；产物展示/复用 UX、失败态、回归 case 明确 |
| P0 | 脚本 skill | 我 | 生成/执行边界、安全约束、产物交付、失败恢复明确 |
| P1 | DeepResearch skill 融合 | 我 + 实习生 | 调研输出被 review，最小融合路径明确 |
| P1 | Gemini 3.5 Flash 合入 agent v2 | 我 | SOTA 结果可复现，content thinking UX gap 有方案 |

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Image generation tool 误触发或产物展示弱 | chat 入口误触发、agent 任务被打断，或用户看不到/复用不了生成结果 | 明确 chat exclusion、agent 触发条件、负例、artifact contract 和结果预览/保存/继续编辑路径 |
| Agent 生图 runtime 契约不清 | tool 加载、schema、错误码和日志不稳定，问题难定位 | 先定义 skill runtime -> image tool 的加载契约和失败态 |
| 审核/`poresche` 偶发失败 | daily 上真实生图请求可能被不稳定阻断，用户体验不连续 | 收集请求 id、输入、审核返回、重试状态和最终用户态，明确兜底/修复 |
| Script skill 安全边界不清 | 可能带来本地副作用、权限和不可复现风险 | 区分生成脚本 vs 执行脚本，记录日志和读写边界 |
| Gemini 3.5 Flash 不输出 content thinking | 用户看不到长任务思考/进度，信任感下降 | 设计 progress / summary / visible reasoning substitute |
| DeepResearch skill 融合过早 | 长任务质量、引用、失败恢复不稳定 | 等实习生调研输出，先补 benchmark case |

## Checklist

- [ ] W14 agent image generation skill/runtime 进入 daily 后稳定性验证。
- [ ] W15 script skill 进入设计/实现。
- [ ] W11 deepresearch skill 调研输出被 review。
- [ ] W12 Gemini 3.5 Flash UX gap 有方案。
- [ ] W13 benchmark deep cases 继续扩充。
